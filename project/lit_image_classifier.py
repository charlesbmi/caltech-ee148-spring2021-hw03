from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class LitImageClassifier(pl.LightningModule):
    def __init__(self, dropout: float = 0.5, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Based on https://github.com/elijahcole/caltech-ee148-spring2020-hw03/blob/master/main.py
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(self.hparams.dropout)
        self.dropout2 = torch.nn.Dropout2d(self.hparams.dropout)
        self.fc1 = torch.nn.Linear(200, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        """Inference / prediction"""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='passed to train_test_split')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitImageClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    return args


def cli_main():
    """Command-line interface for training/validating model."""

    # ------------
    # args
    # ------------
    args = parse_args()

    # Seed generator
    pl.seed_everything(args.random_seed)

    # ------------
    # data
    # ------------
    mnist_train_val = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # Stratified train/val split
    train_idx, val_idx = train_test_split(
            np.arange(len(mnist_train_val)),
            test_size=args.val_size,
            stratify=mnist_train_val.targets)
    mnist_train = Subset(mnist_train_val, train_idx)
    mnist_val = Subset(mnist_train_val, val_idx)

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    model = LitImageClassifier(learning_rate=args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    print(trainer.callback_metrics)

    # ------------
    # testing
    # ------------
    if args.evaluate:
        result = trainer.test(test_dataloaders=test_loader)
        print(result)


if __name__ == '__main__':
    cli_main()
