from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class LitImageClassifier(pl.LightningModule):
    def __init__(
        self, num_classes: int, dropout: float = 0.1, learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Based on https://github.com/elijahcole/caltech-ee148-spring2020-hw03/blob/master/main.py
        self.model = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Convolutional layer 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Convolutional -> fully-connected
            nn.Flatten(),
            # Fully-connected layer 1
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            # Fully-connected output layer
            nn.Linear(128, num_classes),
        )
        # For converting logits to probabilities
        self.softmax = nn.Softmax(dim=1)

        # Use cross-entropy loss on logits (raw linear output, no [log-]softmax)
        self.loss = nn.CrossEntropyLoss()

        # Metrics: can extend with MetricCollection
        metrics = torchmetrics.Accuracy()
        self.train_acc = metrics.clone()
        self.valid_acc = metrics.clone()
        self.test_acc = metrics.clone()

    def forward(self, x):
        """Inference / prediction"""
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self(x)

        loss = self.loss(y_logits, y)
        self.log('train_loss', loss, on_epoch=True)

        # Metrics take probs/classes (not logits), so convert them first
        y_prob = self.softmax(y_logits)
        self.log('train_acc', self.train_acc(y_prob, y), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self(x)

        loss = self.loss(y_logits, y)
        self.log('valid_loss', loss, on_epoch=True)

        # Metrics take probs/classes (not logits), so convert them first
        y_prob = self.softmax(y_logits)
        self.log('valid_acc', self.valid_acc(y_prob, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self(x)

        # Metrics take probs/classes (not logits), so convert them first
        y_prob = self.softmax(y_logits)
        self.log('test_acc', self.test_acc(y_prob, y), on_epoch=True)

        # Return predictions/targets for summary
        y_pred = torch.argmax(y_prob, 1)
        return {'prediction': y_pred, 'target': y}

    def test_epoch_end(self, outputs):
        predictions = torch.cat([out['prediction'] for out in outputs])
        targets = torch.cat([out['target'] for out in outputs])

        confusion_matrix = pl.metrics.functional.confusion_matrix(
            predictions, targets, num_classes=self.hparams.num_classes
        )

        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(),
            index=range(self.hparams.num_classes),
            columns=range(self.hparams.num_classes),
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='viridis', ax=ax)
        plt.close(fig)

        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)

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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument(
        '--val-size', type=float, default=0.15, help='passed to train_test_split'
    )
    parser.add_argument(
        '--subsample_train',
        type=float,
        default=1,
        help='ratio of training data to use, useful for learning curves',
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
        help='evaluate your model on the official test set',
    )
    parser.add_argument(
        '--visualize_errors',
        action='store_true',
        default=False,
        help='visualize example errors and confusion matrix',
    )
    parser.add_argument(
        '--visualize_filters',
        action='store_true',
        default=False,
        help='visualize layer-1 kernels',
    )

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
    # Augment/normalize training set
    pre_process = transforms.ToTensor()  # Also scales to [0, 1]
    prenorm_augment = transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),  # fraction
        scale=(0.9, 1.1),  # factor
        shear=5,  # degrees
    )
    postnorm_augment = transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    mnist_train_val = MNIST(
        '',
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                prenorm_augment,  # Include data-augmentation only for train dataset
                pre_process,
                postnorm_augment,
            ]
        ),
    )
    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(mnist_train_val)),
        test_size=args.val_size,
        stratify=mnist_train_val.targets,
    )
    # Sub-sample training data for learning curve
    train_idx = np.random.choice(
        train_idx, int(args.subsample_train * len(train_idx)), replace=False
    )
    mnist_train = Subset(mnist_train_val, train_idx)
    train_loader = DataLoader(
        mnist_train, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Re-instantiate the validation data, but without data-augmentation
    mnist_train_val = MNIST('', train=True, download=True, transform=pre_process)
    mnist_val = Subset(mnist_train_val, val_idx)
    val_loader = DataLoader(
        mnist_val, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.evaluate:
        mnist_test = MNIST('', train=False, download=True, transform=pre_process)
        test_loader = DataLoader(
            mnist_test, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # ------------
    # model
    # ------------
    num_classes = len(np.unique(mnist_train_val.targets))
    model = LitImageClassifier(
        num_classes=num_classes, learning_rate=args.learning_rate
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    print('Saving logs to:', trainer.log_dir)
    trainer.fit(model, train_loader, val_loader)
    print('callback_metrics:', trainer.callback_metrics)
    print('logged_metrics:', trainer.logged_metrics)

    # ------------
    # testing
    # ------------
    if args.evaluate:
        result = trainer.test(model, test_dataloaders=test_loader)
        print('test_results:', result)

    if args.evaluate and args.visualize_errors:
        # Convert to CPU for evaluation / printing
        model = model.to(torch.device('cpu'))

        # Visualize example errors
        error_images = []
        err_preds = []
        labels = []
        for batch, targets in test_loader:
            pred_proba = model.softmax(model.forward(batch))
            preds = torch.argmax(pred_proba, 1)
            err_indices = np.argwhere(targets != preds)

            for err_idx in err_indices.numpy().flatten():
                error_images.append(batch[err_idx].numpy().squeeze())
                labels.append(targets[err_idx].numpy().squeeze())
                err_preds.append(preds[err_idx].numpy().squeeze())

            if len(labels) >= 9:
                print(f'error_images={len(error_images)} x {error_images[0].shape}')
                print(f'labels={labels}')
                print(f'err_preds={err_preds}')
                break

        # Plot these all!
        fig = plt.figure(figsize=(12, 12))
        n_rows = 3
        n_cols = 3
        for plot_idx, (img, label, pred) in enumerate(
            zip(error_images, labels, err_preds)
        ):
            ax = fig.add_subplot(
                n_rows, n_cols, plot_idx + 1
            )  # matplotlib is 1-indexed
            ax.imshow(img, cmap=plt.get_cmap('gray'))
            ax.set_title('true: {}; pred: {}'.format(label, pred))
        plt.show()

    if args.visualize_filters:
        # Visualize 1st layer of filters
        fig = plt.figure(figsize=(12, 12))
        n_rows = 4
        n_cols = 4
        n_plots = n_rows * n_cols
        for plot_idx, weight_tensor in enumerate(model.model[0].weight[:n_plots]):
            weight = weight_tensor.detach().numpy().squeeze()

            ax = fig.add_subplot(
                n_rows, n_cols, plot_idx + 1
            )  # matplotlib is 1-indexed
            im = ax.imshow(weight, cmap=plt.get_cmap('gray'))
            fig.colorbar(im, ax=ax)
        plt.suptitle('filters')
        plt.show()


if __name__ == '__main__':
    cli_main()
