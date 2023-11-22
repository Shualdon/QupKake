import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import BatchNorm, Linear, TransformerConv
from torch_geometric.nn import global_mean_pool as gap


class Transformer4(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        edge_dim,
        global_size,
        embedding_size,
        conv_layers_num,
        heads,
        linear_layers_num,
        d_energy=False,
        **kwargs,
    ):
        super().__init__()
        torch.manual_seed(42)
        self.d_energy = d_energy
        # GCN layers
        self.initial_conv_d = TransformerConv(
            input_dim, embedding_size, edge_dim=edge_dim, heads=heads
        )
        self.initial_bn_d = BatchNorm(embedding_size * heads)
        self.conv_ds = torch.nn.ModuleList()
        self.batch_norm_ds = torch.nn.ModuleList()
        for i in range(conv_layers_num):
            self.conv_ds.append(
                TransformerConv(
                    embedding_size * heads,
                    embedding_size,
                    edge_dim=edge_dim,
                    heads=heads,
                )
            )
            self.batch_norm_ds.append(BatchNorm(embedding_size * heads))

        self.initial_conv_p = TransformerConv(
            input_dim, embedding_size, edge_dim=edge_dim, heads=heads
        )
        self.initial_bn_p = BatchNorm(embedding_size * heads)
        self.conv_ps = torch.nn.ModuleList()
        self.batch_norm_ps = torch.nn.ModuleList()
        for i in range(conv_layers_num):
            self.conv_ps.append(
                TransformerConv(
                    embedding_size * heads,
                    embedding_size,
                    edge_dim=edge_dim,
                    heads=heads,
                )
            )
            self.batch_norm_ps.append(BatchNorm(embedding_size * heads))

        # Linear Layers
        if self.d_energy:
            self.lin1 = Linear(
                embedding_size * heads * 2 + global_size * 2 + 1, embedding_size
            )
        else:
            self.lin1 = Linear(
                embedding_size * heads * 2 + global_size * 2, embedding_size
            )
        self.lins = torch.nn.ModuleList()
        for i in range(linear_layers_num):
            self.lins.append(Linear(embedding_size, embedding_size))

        # output Layer
        self.out = Linear(embedding_size, 1)

    def forward(
        self,
        x_deprot,
        edge_index_deprot,
        edge_attr_deprot,
        x_prot,
        edge_index_prot,
        edge_attr_prot,
        global_attr_prot,
        global_attr_deprot,
        x_prot_batch,
        x_deprot_batch,
        d_energy=None,
    ):
        if self.d_energy:
            delta_energy = d_energy.reshape(-1, 1)

        hidden_d = self.initial_conv_d(x_deprot, edge_index_deprot, edge_attr_deprot)
        hidden_d = F.elu(hidden_d)
        hidden_d = self.initial_bn_d(hidden_d)
        for bn_d, conv_d in zip(self.batch_norm_ds, self.conv_ds):
            hidden_d = conv_d(hidden_d, edge_index_deprot, edge_attr_deprot)
            hidden_d = F.elu(hidden_d)
            hidden_d = bn_d(hidden_d)

        hidden_p = self.initial_conv_p(x_prot, edge_index_prot, edge_attr_prot)
        hidden_p = F.elu(hidden_p)
        hidden_p = self.initial_bn_p(hidden_p)
        for bn_p, conv_p in zip(self.batch_norm_ps, self.conv_ps):
            hidden_p = conv_p(hidden_p, edge_index_prot, edge_attr_prot)
            hidden_p = F.elu(hidden_p)
            hidden_p = bn_p(hidden_p)

        if self.d_energy:
            hidden = torch.cat(
                [
                    gap(hidden_d, x_deprot_batch),
                    global_attr_deprot,
                    gap(hidden_p, x_prot_batch),
                    global_attr_prot,
                    delta_energy,
                ],
                dim=1,
            )
        else:
            hidden = torch.cat(
                [
                    gap(hidden_d, x_deprot_batch),
                    global_attr_deprot,
                    gap(hidden_p, x_prot_batch),
                    global_attr_prot,
                ],
                dim=1,
            )

        hidden = F.elu(self.lin1(hidden))
        for lin in self.lins:
            hidden = F.elu(lin(hidden))

        out = F.relu(self.out(hidden)).reshape(-1, 1)

        return out


class PredictpKa(pl.LightningModule):
    def __init__(self, lr=0.001, batch_size=64, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.model = Transformer4(*args, **kwargs)
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()
        self.validation_step_outputs = []

    def forward(
        self,
        x_deprot,
        edge_index_deprot,
        edge_attr_deprot,
        x_prot,
        edge_index_prot,
        edge_attr_prot,
        global_attr_prot,
        global_attr_deprot,
        x_prot_batch,
        x_deprot_batch,
        d_energy=None,
    ):
        return self.model(
            x_deprot,
            edge_index_deprot,
            edge_attr_deprot,
            x_prot,
            edge_index_prot,
            edge_attr_prot,
            global_attr_prot,
            global_attr_deprot,
            x_prot_batch,
            x_deprot_batch,
            d_energy,
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            threshold_mode="abs",
        )
        # return optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        preds = self.forward(
            batch.x_deprot,
            batch.edge_index_deprot,
            batch.edge_attr_deprot,
            batch.x_prot,
            batch.edge_index_prot,
            batch.edge_attr_prot,
            batch.global_attr_prot,
            batch.global_attr_deprot,
            batch.x_prot_batch,
            batch.x_deprot_batch,
            batch.d_energy,
        )
        loss = self.loss_fn(preds, batch.y)
        self.log_dict(
            {"train_loss": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(
            batch.x_deprot,
            batch.edge_index_deprot,
            batch.edge_attr_deprot,
            batch.x_prot,
            batch.edge_index_prot,
            batch.edge_attr_prot,
            batch.global_attr_prot,
            batch.global_attr_deprot,
            batch.x_prot_batch,
            batch.x_deprot_batch,
            batch.d_energy,
        )
        loss = self.loss_fn(preds, batch.y)
        self.validation_step_outputs.append(loss)
        self.log_dict(
            {"val_loss": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return preds

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log_dict(
            {"hp_metric": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        preds = self.forward(
            batch.x_deprot,
            batch.edge_index_deprot,
            batch.edge_attr_deprot,
            batch.x_prot,
            batch.edge_index_prot,
            batch.edge_attr_prot,
            batch.global_attr_prot,
            batch.global_attr_deprot,
            batch.x_prot_batch,
            batch.x_deprot_batch,
            batch.d_energy,
        )
        loss = self.loss_fn(preds, batch.y)
        self.log_dict(
            {"test_loss": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self.forward(
            batch.x_deprot,
            batch.edge_index_deprot,
            batch.edge_attr_deprot,
            batch.x_prot,
            batch.edge_index_prot,
            batch.edge_attr_prot,
            batch.global_attr_prot,
            batch.global_attr_deprot,
            batch.x_prot_batch,
            batch.x_deprot_batch,
            batch.d_energy,
        )
        return preds