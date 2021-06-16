import logging
from argparse import Namespace

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from ez_torch.vis import Fig
from pytorch_lightning import loggers

from document_rectification.common import DEVICE
from document_rectification.data import DocumentsDataModule
from document_rectification.models.document_ae_rectifier import DocumentAERectifier

logger = logging.getLogger()


def main():
    hparams = Namespace(
        lr=1e-4,
    )

    logger = loggers.WandbLogger(
        save_dir=".logs",
        project="document-rectification",
    )
    logger.log_hyperparams(hparams)

    dm = DocumentsDataModule(
        train_bs=8,
        val_bs=8,
        plot_bs=8,
        shuffle=True,
        device=DEVICE,
    )
    model = DocumentAERectifier(
        image_channels=3,
        ae_latent_size=50 * 38,
        ae_decoder_initial_reshape=[50, 38],
        transform_res_w=5,
        transform_res_h=5,
        plot_dataloader=dm.plot_dataloader(),
        hparams=hparams,
    ).to(DEVICE)

    trainer = pl.Trainer(
        gpus=1 if DEVICE == "cuda" else None,
        logger=[logger],
        log_every_n_steps=1,
        flush_logs_every_n_steps=3,
        max_epochs=100,
    )
    trainer.fit(model, datamodule=dm)


def sanity_check():
    dm = DocumentsDataModule(
        train_bs=16,
        val_bs=16,
        plot_bs=8,
        shuffle=False,
        device=DEVICE,
    )
    dl = dm.plot_dataloader()
    batch = next(iter(dl))

    model = DocumentAERectifier(
        image_channels=3,
        ae_latent_size=50 * 38,
        ae_decoder_initial_reshape=[50, 38],
        transform_res_w=5,
        transform_res_h=5,
        datamodule=dm,
    ).to(DEVICE)
    info = model.info_forward(batch["x"])

    fig = Fig(nr=1, nc=4, figsize=(15, 10))

    im = batch["x"].ez.grid(nr=2).channel_last.np
    fig[0].imshow(im)
    fig[0].ax.set_title("Input")

    im = info["geom_out"].ez.grid(nr=2).channel_last.np
    fig[1].imshow(im)
    fig[1].ax.set_title("Geom Out")

    im = info["ae_out"].ez.grid(nr=2).channel_last.np
    fig[2].imshow(im)
    fig[2].ax.set_title("AE Out")

    im = batch["y"].ez.grid(nr=2).channel_last.np
    fig[3].imshow(im)
    fig[3].ax.set_title("GT")

    plt.tight_layout()
    plt.show()
    # plt.savefig("a.png")


if __name__ == "__main__":
    # sanity_check()
    main()
