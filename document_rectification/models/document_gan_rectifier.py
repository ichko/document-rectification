import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from document_rectification.models.auto_encoder import AutoEncoder
from document_rectification.models.geometric_transformer import GeometricTransformModel
from ez_torch.vis import Fig
from torch import nn
from torch.functional import Tensor
from torch.nn.modules.activation import Sigmoid


class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 512, 3, stride=2, padding=0),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 32, 3, stride=2, padding=0),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0),
            nn.BatchNorm2d(1, 0.8),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return y


class DocumentGANRectifier(pl.LightningModule):
    def __init__(
        self,
        image_channels,
        transform_res_w,
        transform_res_h,
        plot_dataloader,
        hparams,
    ):
        super().__init__()

        self.hp = hparams
        self.plot_dataloader = plot_dataloader
        self.generator = GeometricTransformModel(
            res_w=transform_res_w,
            res_h=transform_res_h,
        )
        self.discriminator = Discriminator(image_channels)
        self.automatic_optimization = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.generator(x)
        x = self.discriminator(x)
        return x

    def info_forward(self, x: Tensor) -> Tensor:
        geom_out = self.generator(x)
        discriminator_out = self.discriminator(geom_out)
        return {
            "generator_pred": geom_out,
            "discriminator_pred": discriminator_out,
        }

    def configure_optimizers(self):
        id_optim = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return id_optim, g_optim, d_optim

    def training_step(self, batch, _batch_index):
        self.train()

        id_optim, g_optim, d_optim = self.optimizers()
        x, y = batch["x"], batch["y"]
        bs = y.size(0)
        loss = 0
        # Generator retains scanned documents (identity)

        id_optim.zero_grad()
        y_pred = self.generator(y)
        id_loss = F.binary_cross_entropy(y_pred, y)
        self.manual_backward(id_loss, retain_graph=True)
        self.log("id_loss", id_loss)
        id_optim.step()
        loss += id_loss

        # GAN Training
        real_label = 1.0
        fake_label = 0.0

        # Train the discriminator
        d_optim.zero_grad()
        real_y = y
        label = torch.full((bs, 1), real_label, device=self.device)
        real_pred = self.discriminator(real_y)
        real_loss = F.binary_cross_entropy(real_pred, label)
        self.manual_backward(real_loss, retain_graph=True)

        fake_y = self.generator(x)
        label = label.fill_(fake_label)
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

        if d_loss < 0.15:
            g_optim.step()
        loss += g_loss

        self.log("loss", loss)

    def on_epoch_start(self):
        with torch.no_grad():
            self.eval()
            for batch in self.plot_dataloader:
                x, y = batch["x"], batch["y"]
                info = self.info_forward(x)
                pred = info["generator_pred"]

                fig = Fig(nr=1, nc=3, figsize=(15, 10))

                im = x.ez.grid(nr=2).channel_last.np
                fig[0].imshow(im)
                fig[0].ax.set_title("Input")

                im = pred.ez.grid(nr=2).channel_last.np
                fig[1].imshow(im)
                fig[1].ax.set_title("Pred")

                im = y.ez.grid(nr=2).channel_last.np
                fig[2].imshow(im)
                fig[2].ax.set_title("GT")

                plt.tight_layout()
                wandb.log({"chart": plt})
                plt.close()
