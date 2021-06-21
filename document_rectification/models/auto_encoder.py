from typing import Any

import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from document_rectification.common import DEVICE
from document_rectification.data import DocumentsDataModule
from ez_torch.models import Lambda, Reshape
from torch import nn
from torch.functional import Tensor


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.net = torchvision.models.mobilenet.mobilenet_v2(
            pretrained=False,
            progress=True,
            num_classes=latent_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        channel_dim = 1
        if x.size(channel_dim) == 1:
            repeats = [1] * x.ndim
            repeats[channel_dim] = 3
            x = x.repeat(*repeats)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, image_channels, initial_reshape):
        super().__init__()
        self.net = nn.Sequential(
            Reshape(-1, 1, *initial_reshape),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=1.25),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
            Lambda(lambda x: x.repeat(1, image_channels, 1, 1)),  # make 3 channel image
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, image_channels, latent_size, decoder_initial_reshape):
        super().__init__()
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(
            image_channels=image_channels,
            initial_reshape=decoder_initial_reshape,
        )

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Feed image in grayscale form?
        # (maybe not - the background may be easily separable in fullcolor)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, _batch_id):
        self.train()
        x, y = batch["x"], batch["y"]

        y_hat = self(y)

        y = y.mean(dim=1)
        y_hat = y_hat.mean(dim=1)

        loss = self.criterion(y_hat, y)
        self.log("loss", loss)
        return loss

    def on_epoch_start(self):
        self.eval()
        with torch.no_grad():
            if self.plot_dataloader is not None:
                for batch in self.plot_dataloader:
                    x, y = batch["x"], batch["y"]
                    y_hat = self(y)

                    fig = Fig(nr=1, nc=2, figsize=(15, 10))

                    im = y.ez.grid(nr=4).channel_last.np
                    fig[0].imshow(im)
                    fig[0].ax.set_title("Input")

                    im = y_hat.ez.grid(nr=4).channel_last.np
                    fig[1].imshow(im)
                    fig[1].ax.set_title("Pred")

                    plt.tight_layout()
                    wandb.log({"chart": plt})
                    plt.close()


def main():
    dm = DocumentsDataModule(
        train_bs=32,
        val_bs=32,
        plot_bs=16,
        shuffle=False,
        device=DEVICE,
    )
    image_size = [dm.H // 10, dm.W // 10]
    dl = dm.plot_dataloader()
    batch = next(iter(dl))

    model = AutoEncoder(
        image_channels=3,
        latent_size=image_size[0] * image_size[1],
        decoder_initial_reshape=image_size,
        plot_dataloader=dm.plot_dataloader(),
    )
    model.summarize()

    hparams = Namespace(
        lr=1e-4,
    )

    logger = loggers.WandbLogger(
        save_dir=".logs",
        project="document-rectification-ae",
    )
    logger.log_hyperparams(hparams)

    dm = DocumentsDataModule(
        train_bs=6,
        val_bs=8,
        plot_bs=8,
        shuffle=True,
        device=DEVICE,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=".checkpoints/",
        filename="ae-{epoch:02d}-{loss:.2f}",
        save_top_k=5,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=1 if DEVICE == "cuda" else None,
        callbacks=[checkpoint_callback],
        logger=[logger],
        log_every_n_steps=1,
        flush_logs_every_n_steps=3,
        max_epochs=150_000,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
