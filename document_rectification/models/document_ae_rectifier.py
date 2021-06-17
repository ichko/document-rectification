import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from document_rectification.models.auto_encoder import AutoEncoder
from document_rectification.models.geometric_transformer import GeometricTransformModel
from ez_torch.vis import Fig
from torch.functional import Tensor


class DocumentAERectifier(pl.LightningModule):
    def __init__(
        self,
        image_channels,
        ae_latent_size,
        ae_decoder_initial_reshape,
        transform_res_w,
        transform_res_h,
        plot_dataloader,
        hparams,
    ):
        super().__init__()

        self.hp = hparams
        self.plot_dataloader = plot_dataloader
        self.geom_transform = GeometricTransformModel(
            res_w=transform_res_w,
            res_h=transform_res_h,
        )
        self.ae = AutoEncoder(
            image_channels=image_channels,
            latent_size=ae_latent_size,
            decoder_initial_reshape=ae_decoder_initial_reshape,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.geom_transform(x)
        x = self.ae(x)
        return x

    def info_forward(self, x: Tensor) -> Tensor:
        geom_out = self.geom_transform(x)
        ae_out = self.ae(geom_out)
        return {"geom_out": geom_out, "ae_out": ae_out}

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        y_hat = y_hat.mean(dim=1, keepdim=True)
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        # TODO: Set lr for geom_transformer to something smaller
        # schedule it to increase?
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr)

    def training_step(self, batch, _batch_index):
        self.train()
        x, y = batch["x"], batch["y"]
        y_hat = self(x)

        ae_y_hat = self.ae(batch["y"])
        id_recon_loss = self.ae.criterion(ae_y_hat, batch["y"])
        geom_recon_loss = self.criterion(y_hat, y)
        loss = (geom_recon_loss + id_recon_loss) / 2

        self.log("loss", loss)
        self.log("geom_recon_loss", geom_recon_loss)
        self.log("id_recon_loss", id_recon_loss)

        return loss

    def on_epoch_start(self):
        with torch.no_grad():
            self.eval()
            # TODO: This logs only on the first call
            for batch in self.plot_dataloader:
                info = self.info_forward(batch["x"])

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
                wandb.log({"chart": plt})