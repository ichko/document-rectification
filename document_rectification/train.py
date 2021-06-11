import logging
import os
import sys

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from ez_torch.models import SpatialUVOffsetTransformer
from ez_torch.vis import Fig
from pytorch_lightning import loggers
from torch.nn import functional as F

from document_rectification import data

logger = logging.getLogger()

is_debug = "--debug" in sys.argv
if is_debug:
    os.environ["WANDB_MODE"] = "offline"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GeometricTransformModel(pl.LightningModule):
    def __init__(self, res_w, res_h, datamodule):
        super().__init__()
        self.datamodule = datamodule
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.st = SpatialUVOffsetTransformer(
            i=1000,
            uv_resolution_shape=(res_w, res_h),
        )

    def forward(self, x):
        self.features = self.feature_extractor(x)
        x = x.mean(dim=1, keepdim=True)
        y_hat = self.st([self.features, x])
        return y_hat

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-8)

    def training_step(self, batch, _batch_index):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("loss", loss)
        return loss

    def on_epoch_start(self) -> None:
        for batch in self.datamodule.plot_dl():
            x, y = batch["x"], batch["y"]
            y_hat = self(x)
            wandb_exp = self.logger.experiment[0]
            wandb_exp.log({"y_hat": [wandb.Image(i) for i in y_hat]})


def main():
    hparams = {}

    # logger = loggers.TensorBoardLogger(
    #     save_dir=".logs",
    #     name=".checkpoints",
    # )

    logger = loggers.WandbLogger(
        save_dir=".logs",
        project="document-rectification",
    )
    # logger.log_hyperparams(hparams)

    datamodule = data.get_datamodule(train_bs=16, val_bs=16, plot_bs=8, device=DEVICE)
    model = GeometricTransformModel(res_w=2, res_h=2, datamodule=datamodule)
    model = model.to(DEVICE)

    trainer = pl.Trainer(
        gpus=1 if DEVICE == "cuda" else None,
        logger=[logger],
        log_every_n_steps=1,
        flush_logs_every_n_steps=3,
        max_epochs=100,
    )
    trainer.fit(model, datamodule=datamodule)


def sanity_check():
    datamodule = data.get_datamodule(train_bs=16, val_bs=16, plot_bs=8, shuffle=False)
    dl = datamodule.plot_dl()
    batch = next(iter(dl))
    # TODO: Problem with slow plotting!
    model = GeometricTransformModel(res_w=2, res_h=2, datamodule=datamodule).to(DEVICE)
    fig = Fig(nr=1, nc=2, figsize=(15, 15))
    fig[0].imshow(batch["x"].ez.grid(nr=2).channel_last.raw.detach().cpu())
    fig[0].ax.set_title("Input")

    fig[1].imshow(batch["y"].ez.grid(nr=2).channel_last.raw.detach().cpu())
    fig[1].ax.set_title("Output")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sanity_check()
    main()
