import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import dataset
from .evaluation import np_round, div

from .hrnet.hrnet import get_hrnet
from .evaluation import formatting_output_port2head, get_result_matrix

import sys
sys.path.append('..')
from dataset.transform import PCBEVTransform
from utils import utils as u
from prettytable import PrettyTable
from .losses import LovaszLoss
from configs import IMG_SIZE, NUM_CLASSES, CLASSES_NAMES, CLASSES_WEIGHT

class PCBEVModel(LightningModule):

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.backbone = get_hrnet(config['backbone_cfg'])
        self.transform = PCBEVTransform()#GeneralizedRCNNTransform()
        #self.sem_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.multiple_classes = config['multiple_classes']
        if not self.multiple_classes:
            self.sem_criterion = LovaszLoss(loss_weight=1.0, reduction='none', class_weight=CLASSES_WEIGHT)
        else:
            self.sem_criterion = LovaszLoss(loss_weight=1.0, reduction='none', class_weight=CLASSES_WEIGHT)


    
    def load_weights(self, model_path, strict=True):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        # remove some key if needed
        checkpoint = u.remove_key(checkpoint, ['backbone.conv1.weight'])
        self.load_state_dict(checkpoint, strict=False)#strict)


    def forward(self, x):
        x, _ = self.transform(x, None)
        features = self.backbone(x)
        return features
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['init_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_step'], gamma=0.1)
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # skip the first 500 steps
        if self.config['warm_up'] and self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config['init_lr']

        # update params
        optimizer.step(closure=optimizer_closure)
    
    
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        
        #losses = {}
        h, w = IMG_SIZE[0], IMG_SIZE[1]#200, 200#1200, 1920  #loss will compute on this size

        out = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.sem_criterion(out, targets.long())


        #loss = losses['Drivable_Area'] + 10 * losses['Port_Lane']
        
        # logging  
        self.logging(loss)

        return {'loss': loss}

    def logging(self, loss):
        # logging
        #self.log('Train_Drivable', losses['Drivable_Area'],  on_step=True, on_epoch=True)
        #self.log('Train_Lane', losses['Port_Lane'],  on_step=True, on_epoch=True)
        self.log('Train_loss', loss,  on_step=True, on_epoch=True)
        self.log('Learning Rate', self.optimizers().param_groups[0]['lr'], on_step=True)
    
    def validation_step(self, batch, batch_idx):
        return None
    
    def evaluate(self, data_loader_test, key=None):
        classes = NUM_CLASSES#2

        self.eval()
        confmat = u.ConfusionMatrix(classes)
        #port_accu_matrix = []
        with torch.no_grad():
            for img, target in tqdm(data_loader_test):
                target = target.cuda()
                img = img.cuda()
                prediction = self(img)
                sem_out = torch.argmax(prediction, dim=1)

                confmat.update(target.flatten(), sem_out.flatten())

        confusion_matrix = confmat.mat.float()
        confusion_matrix = np.array(confusion_matrix.cpu())

        return confusion_matrix

    def formatting_output_port1head(self, h_drivable):
        "given confusion matrix, compute intrested info"
        # Drivable
        recall = np_round(div(np.diag(h_drivable), h_drivable.sum(1)))
        precision = np_round(div(np.diag(h_drivable), h_drivable.sum(0)))
        f1_drivable = np_round(div(2, (div(1, precision) + div(1, recall))))
        iou = np_round(div(np.diag(h_drivable), (h_drivable.sum(1) + h_drivable.sum(0) - np.diag(h_drivable))))

        x = PrettyTable()
        x.field_names = ['class', 'iou', 'Recall', 'precision', 'f1_score']
        if NUM_CLASSES == 2:
            x.add_row(['background', iou[0], recall[0], precision[0], f1_drivable[0]])
            x.add_row(['obstacle', iou[1], recall[1], precision[1], f1_drivable[1]])
            accu_final = f1_drivable[1]
        else:
            accu_final = 0
            for idx, name in enumerate(CLASSES_NAMES):
                x.add_row([name, iou[idx], recall[idx], precision[idx], f1_drivable[idx]])
                if idx != 0:
                    accu_final += f1_drivable[idx]
            accu_final /= (len(CLASSES_NAMES) -1)


        return np_round(accu_final, 4), x.get_string()


    def validation_epoch_end(self, outs):
        # if self.trainer.total_batch_idx > 0:
        if self.trainer.global_step > 0:
            confusion_matrix_drivable = self.evaluate(self.dataset.val_dataloader())

            accu, td = self.formatting_output_port1head(confusion_matrix_drivable)

            #self.print('\n'*2, td, '\n', tl)
            self.print('\n' * 2, td)
            self.print("EPOCH: {} --> Accu: {}".format(self.current_epoch, accu))
            self.log('Accu', accu)
            return accu
        else:
            # for first iter
            self.log('Accu', torch.tensor(0.))