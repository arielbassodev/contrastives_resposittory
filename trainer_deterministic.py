from datetime import datetime
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from contrastive_approaches import CLRLightningModule
from datamodule import CassavaDataModule
import argparse
import os
import re
from pathlib import Path
from app_logger import logger
from utils import BackBonesType, BACKBONES, CONTRASTIVES_APPROACHES, OPTIMIZERS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir_path',
        type=str,
        help='Path to the directory containing images',
        required=True
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        help='Path to the train CSV file',
        required=False
    )
    parser.add_argument(
        '--json_file',
        type=str,
        help='Path to the label mapping JSON file',
        required=False
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=BACKBONES,
        help='The chosen backbone',
        required=True
    )
    parser.add_argument(
        '--contrastive_approach',
        type=str,
        choices=CONTRASTIVES_APPROACHES,
        help='The chosen contrastive approach',
        required=True
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=OPTIMIZERS,
        default='sgd',
        help='The name of the optimizer'
    )
    parser.add_argument(
        '--value_to_optimize',
        type=str,
        choices=['train_loss', 'val_loss'],
        default='train_loss',
        help='The value we want to optimize for the best model'
    )
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of training epochs (default: 10)')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of workers for data loading (default: 12)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation proportion (default: 0.2)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test proportion (default: 0.2)')
    args = parser.parse_args()
    for p in [args.img_dir_path, args.csv_file, args.json_file]:
        if not Path(p).exists():
            print(f"Error: {p} does not exist!")
            exit(0)
    return args

def train(args, checkpoints_subdir='clr', logger_save_subdir='clr_tb_logs'):
    random_seed = 42
    seed_everything(random_seed, workers=True)
    torch.use_deterministic_algorithms(True)
    logger.info("Argument: %s", args)
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir=args.img_dir_path if getattr(args, 'csv_file', None) is None  else (args.img_dir_path, args.csv_file, args.json_file)
    dm = CassavaDataModule(data_dir=data_dir,
                           batch_size=args.batch_size,
                           val_split=args.val_split, test_split=args.test_split,
                           n_transforms_to_choose=0, # 0 because during contrastive learning we do not want data agmentation
                           base_transform_to_use=args.backbone,
                           num_workers=args.num_workers,
                           random_seed=random_seed)
    dm.setup(stage="fit")  # calling in the fit stage to initialize the train and validation dataloaders



    l_model = CLRLightningModule(clr_model_or_backbone_name=args.backbone,
                                 contrastive_approach=args.contrastive_approach,
                                 optimizer_name=args.optimizer,
                                 lr=args.learning_rate)

    checkpoints_dir='./checkpoints'
    backbone_cat = re.match(r"^[^0-9_]+", args.backbone).group()
    checkpoints_dir = os.path.join(checkpoints_dir, checkpoints_subdir, backbone_cat)

    # Will save logs into the given subfolder
    logger_save_dir='./my_logs'
    logger_save_dir = os.path.join(logger_save_dir, logger_save_subdir)
    os.makedirs(logger_save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=logger_save_dir, name=backbone_cat)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor='train_loss',
        mode="min",
        save_top_k=1,
        filename=f'{args.backbone}--{args.contrastive_approach}--{current_datetime}--version_{tb_logger.version}' + '--best-model-{epoch:02d}--{train_loss:.3f}--{val_loss:.3f}'
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        limit_val_batches=0, #validation disabled during training
        deterministic=True)

    trainer.fit(model=l_model, datamodule=dm)
    best_loss = trainer.checkpoint_callback.best_model_score
    best_metrics = checkpoint_callback.best_k_models[checkpoint_callback.best_model_path]
    logger.info("Best metrics: %s", best_metrics)

    trainer.limit_val_batches = 1.0
    validation_metrics = trainer.validate(l_model,  datamodule=dm, ckpt_path="best")
    return  {
      'validation_metrics': validation_metrics[0],
      'best_model_score': best_loss.item(),
      'model': l_model,
      'datamodule': dm,
      'trainer': trainer,
      'best_model_path': checkpoint_callback.best_model_path
    }

if __name__ == "__main__":
    logger.add_file_handler_to_logger(file_name='clr_log_output')
    logger.info('============== [Starting] =======================')
    args = parse_args()
    ret = train(args)
    logger.info('Validation metrics: %s', ret['validation_metrics'])
    logger.info('Best train loss: %s', ret['best_model_score'])
    logger.info('Best Model Path: %s', ret['best_model_path'])
    logger.info('==================================================')




