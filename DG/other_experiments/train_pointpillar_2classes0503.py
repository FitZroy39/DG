import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


from model.pointpillar_model import POINTPILLARModel
from dataset.data_module import POINTPILLARDataModule

config = {
    'gpu_list': [4,5,6,7],
    'exp_name': "test_run_pointpillar_binclass05x03_lovaszloss_lr5e4_cw_jitterda",    #exp name
    
    # train
    'max_epoch': 20,#32,
    'dev_mode': None,  # change to a small int for dev
    'seed': 0,
    'warm_up': True,
    'init_lr': 5e-4,#5e-3,#1e-2,#3,
    'lr_step': [10, 16],#[18, 26],

    # monitor
    'use_wandb': False,

    # backbone
    'backbone_cfg': './model/hrnet/conf_file/m1112_b2_c12.yaml',
    'using_init_weights': True,
    'resume_training': False,
    'init_weights': './model/imagenet_pretrained_weights.pth',
    
    # dataloader
    'train_batch_size': 2,#16,#32,
    'train_num_workers': 2,#16,#32,#4,
    'val_batch_size': 2,#4,#1,
    'val_num_workers': 2,#4,#2,
    'det_val_batch_size': 2,
    'reload_dataloaders_every_epoch': False,
    
    # dataset
    # 'root_path': '/ssd/Port_Dataset/',
    'root_path': '/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_rain/',#'/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_rain/',
    'train_dir': 'images/training',
    #'lane_train_list': 'Lane_Port/train_list.txt',
    'val_dir': 'images/validation',
    #'lane_val_list': 'Lane_Port/val_list.txt',
    'exclude_list': '',  # add a exclude img list if needed

    # eval
    'eval_before_training': False,
    'eval_and_vis': False,#True,

    #binary or multiple classes
    'multiple_classes': False
}

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config['gpu_list']))
seed_everything(seed=config['seed'])

# logger
tb_logger = pl_loggers.TensorBoardLogger('logs/'+config['exp_name'])
logger = [tb_logger]

if config['use_wandb']:
    wandb_logger = WandbLogger(
        name = config['exp_name'], 
        save_dir = 'logs/'+config['exp_name'],
        offline=False,
        project='Port_Visual_2Head',
        )
    logger.append(wandb_logger)

checkpoint_callback = ModelCheckpoint(
    monitor='Accu',
    dirpath='./logs/'+config['exp_name'],
    filename='{epoch:02d}-{Accu:.2f}',
    save_top_k=1,
    verbose=True,
    mode='max',
    save_last=True,
)

# train
dataset = POINTPILLARDataModule(config=config)
model = POINTPILLARModel(config, dataset)

if config['using_init_weights']:
    model.load_weights(config['init_weights'], strict=False)

resume_from_checkpoint = config['init_weights'] if config['resume_training'] else None
num_sanity_val_steps = -1 if config['eval_before_training'] else 6

trainer = pl.Trainer(
                resume_from_checkpoint=resume_from_checkpoint,
                logger=logger,
                max_epochs=config['max_epoch'],
                progress_bar_refresh_rate=100,#50,
                gpus=list(range(len(config['gpu_list']))),
                reload_dataloaders_every_epoch=config['reload_dataloaders_every_epoch'],
                num_sanity_val_steps=num_sanity_val_steps,
                callbacks=[checkpoint_callback],
                distributed_backend='ddp',
                sync_batchnorm=True,
                precision=16
                )

trainer.fit(model, dataset)

# record best model's path
best_model_path = trainer.checkpoint_callback.best_model_path
with open('results/best_model_path.txt', 'w+') as f:
    f.write(best_model_path)
    
if config['eval_and_vis']:
    # eval model & save visualization
    from eval_bev import eval_and_vis
    eval_and_vis(best_model_path, to_video=False)
