from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ez_torch.data import get_mnist_dl


class DenseGAN(pl.LightningModule):
    def __init__(self, hp: Namespace, z_size: int) -> None:
        super().__init__()
        self.hp = hp

        self.generator = nn.Sequential(
            nn.Linear(z_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 28 * 28),
            nn.Sigmoid(),
        )

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

        self.automatic_optimization = False

    def sample_z(self, bs):
        return torch.randn(bs, device=self.device)  # N(0, 1)

    def forward(self, bs):
        z = self.sample_z(bs)

        fake = self.generator(z)
        pred = self.discriminator(fake)

        return fake, pred

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.hp.lr)
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.hp.lr)
        return g_optim, d_optim

    def training_step(self, batch, _batch_id=0):
        self.train()

        bs = batch.size(0)
        g_optim, d_optim = self.optimizers()
        real_label, fake_label = 1.0, 0.0
        z = self.sample_z(bs)
        loss = 0

        # Train the discriminator
        d_optim.zero_grad()
        real_y = batch
        label = torch.full((bs, 1), real_label, device=self.device)
        real_pred = self.discriminator(real_y)
        real_loss = F.binary_cross_entropy(real_pred, label)
        self.manual_backward(real_loss, retain_graph=True)

        fake_y = self.generator(z)
        label.fill_(fake_label)
        fake_pred = self.discriminator(fake_y)
        fake_loss = F.binary_cross_entropy(fake_pred, label)
        self.manual_backward(fake_loss, retain_graph=True)
        d_loss = real_loss + fake_loss
        self.log("d_loss", d_loss)

        d_optim.step()
        loss += d_loss

        # Train the generator
        g_optim.zero_grad()

        label.fill_(real_label)
        fake_pred = self.discriminator(fake_y)
        fake_loss = F.binary_cross_entropy(fake_pred, label)
        self.manual_backward(fake_loss)

        g_loss = fake_loss
        self.log("g_loss", g_loss)

        g_optim.step()
        loss += g_loss

        self.log("loss", loss)
        return loss


def test_mnist_dataloader():
    train, test = get_mnist_dl(bs_train=32, bs_test=32, shuffle=False)
    X, y = next(iter(train))
    assert list(X.shape) == [32, 1, 28, 28]


def test_dense_gan():
    train, test = get_mnist_dl(bs_train=32, bs_test=32, shuffle=False)
    X, y = next(iter(train))


if __name__ == "__main__":
    test_mnist_dataloader()
