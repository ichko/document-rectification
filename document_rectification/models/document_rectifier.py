import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from document_rectification.models.auto_encoder import AutoEncoder
from document_rectification.models.geometric_transformer import GeometricTransformModel
from torch.functional import Tensor


class DocumentRectifier(pl.LightningModule):
    def __init__(
        self,
        image_channels,
        ae_latent_size,
        ae_decoder_initial_reshape,
        transform_res_w,
        transform_res_h,
        datamodule,
    ):
        super().__init__()
        self.geom_transform = GeometricTransformModel(
            res_w=transform_res_w,
            res_h=transform_res_h,
            datamodule=datamodule,
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

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def training_step(self, batch, _batch_index):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("loss", loss)
        return loss

    def on_epoch_start(self) -> None:
        for batch in self.datamodule.plot_dataloader():
            x, y = batch["x"], batch["y"]
            y_hat = self(x)
            wandb_exp = self.logger.experiment[0]
            wandb_exp.log({"y_hat": [wandb.Image(i) for i in y_hat]})
