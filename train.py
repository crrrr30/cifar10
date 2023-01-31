from pytorch_lightning.tuner import lr_finder

import argparse
import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl

from fourier_net import *

from data_loader import get_train_dataloader, get_test_dataloader


def parse_args():

    parser = argparse.ArgumentParser(description = 'Train CIFAR-10')
    
    parser.add_argument('--num-epochs', help = 'Number of epochs to train the model for', default = 500, type = int)
    parser.add_argument('--batch-size', help = 'Batch size', default = 256, type = int)
    parser.add_argument('--log', help = 'Directory to save logs to', default = 'logs', type = str)
    parser.add_argument('--resume', help = 'Directory of the checkpoint to resume from or the path to the checkpoint', default = None, type = str)
    parser.add_argument('--checkpoint-interval', help = 'Frequency of saving checkpoints', default = 5, type = str)
    parser.add_argument('--half', help = 'Whether to use mixed precision training', default = True, type = bool)
    
    return parser.parse_args()


class LitClassifier(pl.LightningModule):
        def __init__(self, model_config, lr=1e-4):
            super().__init__()
            self.learning_rate = lr
            self.automatic_optimization = False
            self.num_epochs = model_config["num_epochs"]
            self.model = FMLPS_4()
            self.criterion = nn.CrossEntropyLoss()
        def forward(self, x):
            return self.model(x)
        def configure_optimizers(self):
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=.1)
            scheduler = LinearLR(
                optimizer,
                start_factor=0.0015,
                end_factor=0.0195,
                total_iters=self.trainer.estimated_stepping_batches * self.num_epochs
            )
            return [optimizer], [scheduler]
        def training_step(self, data, idx):
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()
            x, y = data
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            self.log("train_loss", loss, prog_bar=True)
            norm = torch.nn.utils.clip_grad.clip_grad_norm(self.model.parameters(), max_norm=1.)
            self.log("grad_norm", norm.item(), prog_bar=True)
            optimizer.step()
            scheduler.step()
            return {
                "loss": loss,
                "progress_bar": {
                    "loss": loss.item(),
                    "grad_norm": norm.item(),
                }
            }
        def validation_step(self, data, idx):
            x, y = data
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            _, predicted = y_hat.max(1)
            total = y.size(0)
            correct = predicted.eq(y).sum().item()
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", correct / total, prog_bar=True, on_epoch=True)


if __name__ == '__main__':

    args = parse_args()
    
    print('==> Prepping data...')
    train_dataloader = get_train_dataloader(args.batch_size)
    test_dataloader = get_test_dataloader(args.batch_size)

    print('==> Building CNN...')
    
    tb_logger = pl.loggers.TensorBoardLogger(".")

    model = LitClassifier(vars(args))
    trainer = pl.Trainer(
        default_root_dir=".",
        enable_progress_bar=True,
        max_epochs=args.num_epochs,
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        callbacks=[],
        logger=tb_logger,
        check_val_every_n_epoch=args.checkpoint_interval,
        precision=16 if args.half else 32,
    )

    print("==> Optimizing LR")
    trainer.tune(model, train_dataloader)

    if args.resume:
        model.load_from_checkpoint(args.resume)
        
    print(f'Number of total params: {sum([np.prod(p.shape) for p in model.parameters()]):,}')

    trainer.fit(model, train_dataloader, test_dataloader)