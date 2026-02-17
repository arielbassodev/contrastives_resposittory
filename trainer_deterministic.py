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
        help='Path to the directory containing images'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        help='Path to the train CSV file'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        help='Path to the label mapping JSON file'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=BACKBONES,
        help='The chosen backbone'
    )
    parser.add_argument(
        '--contrastive_approach',
        type=str,
        choices=CONTRASTIVES_APPROACHES,
        help='The chosen contrastive approach'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=OPTIMIZERS,
        default='sgd',
        help='The name of the optimizer'
    )
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of training epochs (default: 10)')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of workers for data loading (default: 12)')
    args = parser.parse_args()
    for p in [args.img_dir_path, args.csv_file, args.json_file]:
        if not Path(p).exists():
            print(f"Error: {p} does not exist!")
            exit(0)
    return args

def main():
    random_seed = 42
    seed_everything(random_seed)
    seed_everything(random_seed, workers=True)
    torch.use_deterministic_algorithms(True)
    args = parse_args()
    logger.info("Argument: %s", args)
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    dm = CassavaDataModule(data_dir=(args.img_dir_path, args.csv_file, args.json_file),
                           batch_size=args.batch_size,
                           val_split=0.2, test_split=0.2,
                           n_transforms_to_choose=0, # 0 because during contrastive learning we do not want data agmentation
                           base_transform_to_use=args.backbone,
                           num_workers=args.num_workers,
                           random_seed=random_seed)
    dm.setup(stage="fit")  # calling in the fit stage to initialize the train and validation dataloaders



    l_model = CLRLightningModule(clr_model_or_backbone_name=args.backbone,
                                 contrastive_approach=args.contrastive_approach,
                                 optimizer_name=args.optimizer,
                                 lr=args.lr,
                                 active_groups=["rotations"])

    logger_save_dir='./my_logs'
    backbone_cat = re.match(r"^[^0-9_]+", args.backbone).group()
    checkpoints_dir = os.path.join('checkpoints', 'clr', backbone_cat)

    # Will save logs into the given subfolder
    logger_save_dir = os.path.join(logger_save_dir, 'clr_tb_logs')
    os.makedirs(logger_save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=logger_save_dir, name=backbone_cat)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        filename=f'{args.backbone}--{args.contrastive_approach}--{current_datetime}--version_{tb_logger.version}' + '--best-model-{epoch:02d}--{train_loss:.2f}--{val_loss:.2f}'
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        deterministic=True)

    trainer.fit(model=l_model, datamodule=dm)
    validation_metrics = trainer.validate(l_model, datamodule=dm, ckpt_path="best")
    # return {'trainer': trainer, 'validation_metrics': validation_metrics}
    best_loss = trainer.checkpoint_callback.best_model_score


    return  {
      'validation_metrics': validation_metrics[0],
      'best_train_loss': best_loss,
      'model': l_model,
      'datamodule': dm,
      'trainer': trainer,
      'best_model_path': checkpoint_callback.best_model_path}

if __name__ == "__main__":
    logger.add_file_handler_to_logger(file_name='clr_log_output')
    logger.info('============== [Starting] =======================')
    ret = main()
    logger.info('Validation metrics: %s', ret['validation_metrics'])
    logger.info('Best train loss: %s', ret['best_train_loss'])
    logger.info('Best Model Path: %s', ret['best_model_path'])
    logger.info('==================================================')




