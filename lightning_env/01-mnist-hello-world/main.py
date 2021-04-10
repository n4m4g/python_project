import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


class MNISTModel(pl.LightningModule):

    # just like the nn.Module init
    def __init__(self, data_dir='./', h_size=64, lr=2e-4):
        super(MNISTModel, self).__init__()
        self.data_dir = data_dir
        self.h_size = h_size
        self.lr = lr
        self.b_size = 32
        self.n_workers = 8

        self.num_classes = 10
        c, h, w = (1, 28, 28)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c*h*w, h_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_size, self.num_classes)
        )

    # just like the model forward
    def forward(self, x):
        return self.model(x)

    # handle forward and calculate loss
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    # create optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir,
                               train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full,
                                                            [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir,
                                    train=False,
                                    transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          batch_size=self.b_size,
                          num_workers=self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          batch_size=self.b_size,
                          num_workers=self.n_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_train,
                          batch_size=self.b_size,
                          num_workers=self.n_workers)


model = MNISTModel()

# create lightning trainer
# gpu count, epochs and something else
trainer = pl.Trainer(gpus=0,
                     max_epochs=10,
                     progress_bar_refresh_rate=20)

# training
trainer.fit(model)

trainer.test()
