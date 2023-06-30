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
from dataset.transform import POINTPILLARTransform
from utils import utils as u
from prettytable import PrettyTable
from .losses import LovaszLoss, CrossEntropyLoss
from configs import IMG_SIZE, NUM_CLASSES, CLASSES_NAMES, CLASSES_WEIGHT, voxel_size, AREA_EXTENTS
from dataset.pointpillar import point_cloud_range


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.
    :param actual_num:
    :param max_num:
    :param axis:
    :return: [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num : [N, M, 1]
    # tiled_actual_num : [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # title_max_num : [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape : [batch_size, max_num]
    return paddings_indicator

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        """
        对于PointPillars而言,它只用到了一层PFNLayer
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels
        # if use_norm:
        #     BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        #     Linear = change_default_args(bias=False)(nn.Linear)
        # else:
        #     BatchNorm1d = Empty
        #     Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = nn.Linear(self.in_channels, self.units, bias=False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        # kernel=>1x1x9x64
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1)
        self.t_conv = nn.ConvTranspose2d(100, 1, (1, 8), stride=(1, 7))
        """
        这个操作nb,这里不是用的max pooling,而是用的带孔卷积,
        34 + 33*2 = 100
        用了一个w方向kernel size为34,w方向dilation为3的带孔卷及
        """
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 34), stride=(1, 1), dilation=(1, 3))

    def forward(self, input):
        """
        (Pdb) masked_features.shape
        torch.Size([1, 9, 6815, 100])
        """
        """
        (Pdb) input.shape
        torch.Size([1, 9, 6815, 100])
        """
        x = self.conv1(input)  # kernel=>1x1x9x64?
        x = self.norm(x)  # =>shape[1,64,6815,100]
        x = F.relu(x)  #
        x = self.conv3(x)  # =>shape[1,64,6815,1]
        return x


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.5, 0.3, 3),#(0.2, 0.2, 4),
                 pc_range=(-50, -50, -1, 100, 50, 2)):#(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)  # [9,64]
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]  # 输入通道,9
            out_filters = num_filters[i + 1]  # 输出通道,64
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
    '''
    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()
    '''
    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i,
                num_voxels,  # num_points,名字为啥总是在变呢??
                x_sub_shaped, y_sub_shaped,
                mask):
        # Find distance of x, y, and z from cluster center
        # pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 3)

        pillar_xyz = torch.cat((pillar_x, pillar_y, pillar_z), 1)

        # points_mean = pillar_xyz.sum(dim=2, keepdim=True) / num_voxels.view(1,-1, 1, 1)

        # 算pillar算术均值
        points_mean = pillar_xyz.sum(dim=3, keepdim=True) / num_voxels.view(1, 1, -1, 1)
        f_cluster = pillar_xyz - points_mean
        # Find distance of x, y, and z from pillar center

        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped
        """
        (Pdb) f_center_concat.shape
        torch.Size([1, 2, 6815, 100])
        """
        f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 1)

        pillar_xyzi = torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 1)
        features_list = [pillar_xyzi, f_cluster, f_center_concat]

        features = torch.cat(features_list, dim=1)
        masked_features = features * mask

        pillar_feature = self.pfn_layers[0](masked_features)
        # return=>shape[1,64,6815,1]
        return pillar_feature


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 #output_shape,
                 num_input_features=64,
                 batch_size=2):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        #self.output_shape = output_shape
        self.ny = IMG_SIZE[0]#output_shape[2]
        self.nx = IMG_SIZE[1]#output_shape[3]
        self.nchannels = num_input_features
        self.batch_size = batch_size
        #self.dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]

    # def forward(self, voxel_features, coords, batch_size):
    def forward(self, voxel_features, coords):
        # batch_canvas will be the final output.
        batch_canvas = []

        if self.batch_size == 1:
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)
            indices = coords[:, 2] * self.nx + coords[:, 3]
            indices = indices.type(torch.float64)
            transposed_voxel_features = voxel_features.t()

            # Now scatter the blob back to the canvas.
            indices_2d = indices.view(1, -1)
            ones = torch.ones([self.nchannels, 1], dtype=torch.float64, device=voxel_features.device)
            indices_num_channel = torch.mm(ones, indices_2d)
            indices_num_channel = indices_num_channel.type(torch.int64)
            scattered_canvas = canvas.scatter_(1, indices_num_channel, transposed_voxel_features)

            # Append to a list for later stacking.
            batch_canvas.append(scattered_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(1, self.nchannels, self.ny, self.nx)
            return batch_canvas
        elif self.batch_size == 2:
            first_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                       device=voxel_features.device)
            # Only include non-empty pillars
            first_batch_mask = coords[:, 0] == 0
            first_this_coords = coords[first_batch_mask, :]
            first_indices = first_this_coords[:, 2] * self.nx + first_this_coords[:, 3]
            first_indices = first_indices.type(torch.long)
            first_voxels = voxel_features[first_batch_mask, :]
            first_voxels = first_voxels.t()

            # Now scatter the blob back to the canvas.
            first_canvas[:, first_indices] = first_voxels

            # Append to a list for later stacking.
            batch_canvas.append(first_canvas)

            # Create the canvas for this sample
            second_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                        device=voxel_features.device)

            second_batch_mask = coords[:, 0] == 1
            second_this_coords = coords[second_batch_mask, :]
            second_indices = second_this_coords[:, 2] * self.nx + second_this_coords[:, 3]
            second_indices = second_indices.type(torch.long)
            second_voxels = voxel_features[second_batch_mask, :]
            second_voxels = second_voxels.t()

            # Now scatter the blob back to the canvas.
            second_canvas[:, second_indices] = second_voxels

            # Append to a list for later stacking.
            batch_canvas.append(second_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(2, self.nchannels, self.ny, self.nx)
            return batch_canvas
        else:
            print("Expecting batch size less than 2")
            return 0

class POINTPILLARModel(LightningModule):

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

        self.voxel_feature_extractor = PillarFeatureNet(num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=voxel_size,#(0.5, 0.3, 3),#(0.2, 0.2, 4),
                 pc_range=point_cloud_range)#(-50, -50, -1, 100, 50, 2))

        self.middle_feature_extractor = PointPillarsScatter(#output_shape=output_shape,
                                                            num_input_features=64,#vfe_num_filters[-1],
                                                            batch_size=2)



        self.backbone = get_hrnet(config['backbone_cfg'])
        self.transform = POINTPILLARTransform()#GeneralizedRCNNTransform()
        #self.sem_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.multiple_classes = config['multiple_classes']
        if not self.multiple_classes:
            self.sem_criterion = CrossEntropyLoss(class_weight=CLASSES_WEIGHT,loss_weight=10)
            #LovaszLoss(loss_weight=1.0, reduction='none', class_weight=[0.1, 10])
        else:
            self.sem_criterion = CrossEntropyLoss(class_weight=CLASSES_WEIGHT,loss_weight=10)
            #LovaszLoss(loss_weight=1.0, reduction='none', class_weight=CLASSES_WEIGHT)


    
    def load_weights(self, model_path, strict=True):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        # remove some key if needed
        checkpoint = u.remove_key(checkpoint, ['backbone.conv1.weight'])
        self.load_state_dict(checkpoint, strict=False)#strict)

    def data_preprocess(self, example):
        """
        pillar_x = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        pillar_y = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        pillar_z = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        pillar_i = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        num_points_per_pillar = torch.ones([1, 12000], dtype=torch.float32, device=pillar_x.device)
        x_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        y_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        mask = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
        """
        #print('*****',len(example.keys()))
        #for k in example.keys():
        #    print(k, example[k].shape)
        # Training Input form example
        pillar_x = example['voxels'][:, :, 0].unsqueeze(0).unsqueeze(0)
        pillar_y = example['voxels'][:, :, 1].unsqueeze(0).unsqueeze(0)
        pillar_z = example['voxels'][:, :, 2].unsqueeze(0).unsqueeze(0)
        pillar_i = example['voxels'][:, :, 3].unsqueeze(0).unsqueeze(0)
        num_points_per_pillar = example['num_points'].float().unsqueeze(0)

        ################################################################
        # Find distance of x, y, z from pillar center
        # assume config_file xyres_16.proto
        coors_x = example['coordinates'][:, 3].float()
        coors_y = example['coordinates'][:, 2].float()
        # self.x_offset = self.vx / 2 + pc_range[0]
        # self.y_offset = self.vy / 2 + pc_range[1]
        # this assumes xyres 20
        # x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
        # y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
        ################################################################

        # assumes xyres_16
        x_sub = coors_x.unsqueeze(1) * voxel_size[0] + AREA_EXTENTS[0][0] + voxel_size[0]/2 #* 0.16 + 0.08
        y_sub = coors_y.unsqueeze(1) * voxel_size[1] + AREA_EXTENTS[1][0] + voxel_size[1]/2 #* 0.16 - 39.6
        ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
        #print('--------------------------',coors_x.shape,x_sub.shape)
        x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
        y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

        num_points_for_a_pillar = pillar_x.size()[3]
        mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
        mask = mask.permute(0, 2, 1)
        mask = mask.unsqueeze(1)
        mask = mask.type_as(pillar_x)

        coors = example['coordinates']


        return [pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar,
                 x_sub_shaped, y_sub_shaped, mask, coors]

    def forward(self, x):
        #x, _ = self.transform(x, None)
        for k in x.keys():
            x[k] = torch.as_tensor(x[k]).to("cuda")
        example = self.data_preprocess(x)

        pillar_x = example[0]
        pillar_y = example[1]
        pillar_z = example[2]
        pillar_i = example[3]
        num_points = example[4]
        x_sub_shaped = example[5]
        y_sub_shaped = example[6]
        mask = example[7]

        #print(pillar_x[:50],x_sub_shaped[:50])
        voxel_features = self.voxel_feature_extractor(pillar_x, pillar_y, pillar_z, pillar_i,
                                                      num_points, x_sub_shaped, y_sub_shaped, mask)

        ###################################################################################
        # return voxel_features ### onnx voxel_features export
        # middle_feature_extractor for trim shape
        voxel_features = voxel_features.squeeze()
        voxel_features = voxel_features.permute(1, 0)

        coors = example[8]

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        # spatial_features input size is : [1, 64, 496, 432]



        features = self.backbone(spatial_features)
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
        #imgs, targets = batch
        #print('00000000000000000',batch.keys())
        targets = torch.as_tensor(batch['targets']).to("cuda")
        preds = self(batch)
        
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
            for batch in tqdm(data_loader_test):
                target = torch.as_tensor(batch['targets']).cuda()#target.cuda()
                #img = img.cuda()
                prediction = self(batch)
                sem_out = torch.argmax(prediction, dim=1)
                #print('--------------',target.shape, sem_out.shape)

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