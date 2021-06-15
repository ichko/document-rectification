import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from ez_torch.vis import Fig
from pytorch_lightning import loggers

from document_rectification.common import DEVICE
from document_rectification.data import DocumentsDataModule
from document_rectification.models.document_rectifier import DocumentRectifier

logger = logging.getLogger()


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

    datamodule = DocumentsDataModule(
        train_bs=16,
        val_bs=16,
        plot_bs=8,
        device=DEVICE,
    )
    model = DocumentRectifier(res_w=2, res_h=2, datamodule=datamodule)
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
    datamodule = DocumentsDataModule(
        train_bs=16,
        val_bs=16,
        plot_bs=8,
        shuffle=False,
        device=DEVICE,
    )
    dl = datamodule.plot_dataloader()
    batch = next(iter(dl))

    image_size = datamodule.W
    model = DocumentRectifier(
        image_channels=3, image_size=image_size, res_w=2, res_h=2, datamodule=datamodule
    ).to(DEVICE)
    predictions = model(batch["x"])

    fig = Fig(nr=1, nc=3, figsize=(15, 10))

    im = batch["x"].ez.grid(nr=2).channel_last.np
    fig[0].imshow(im)
    fig[0].ax.set_title("Input")

    im = predictions.ez.grid(nr=2).channel_last.np
    fig[1].imshow(im)
    fig[1].ax.set_title("Prediction")

    im = batch["y"].ez.grid(nr=2).channel_last.np
    fig[2].imshow(im)
    fig[2].ax.set_title("GT")

    plt.tight_layout()
    plt.show()
    # plt.savefig("a.png")


if __name__ == "__main__":
    sanity_check()
    main()
