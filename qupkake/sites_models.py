import torch

torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore

import pytorch_lightning as pl
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    Linear,
    Sequential,
    GCNConv,
    SAGEConv,
    NNConv,
    BatchNorm,
)
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MatthewsCorrCoef


class adder:
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        return torch.add(x1, x2)


class GATNet(torch.nn.Module):
    def __init__(
        self, input_size, edge_size, hidden_layers, layer_size, dropout, heads, **kwargs
    ):
        super().__init__()
        layers = []
        layers.extend(
            [
                (
                    GATConv(input_size, layer_size, heads=heads, edge_dim=edge_size),
                    "x, edge_index, edge_attr -> x1",
                ),
                (nn.Linear(input_size, heads * layer_size), "x -> x2"),
                # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                (adder(), "x1, x2 -> x"),
                (BatchNorm(heads * layer_size), "x -> x"),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        for layer in range(hidden_layers):
            layers.extend(
                [
                    (
                        GATConv(
                            heads * layer_size,
                            layer_size,
                            heads=heads,
                            edge_dim=edge_size,
                        ),
                        "x, edge_index, edge_attr -> x1",
                    ),
                    (nn.Linear(heads * layer_size, heads * layer_size), "x -> x2"),
                    # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                    (BatchNorm(heads * layer_size), "x -> x"),
                    (adder(), "x1, x2 -> x"),
                    nn.ELU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(heads * layer_size, 1))
        self.model = Sequential("x, edge_attr, edge_index", layers)

    def forward(self, x, edge_attr, edge_index, **kwargs):
        return self.model(x, edge_attr, edge_index)


class TransformerNet(torch.nn.Module):
    def __init__(
        self, input_size, edge_size, hidden_layers, layer_size, dropout, heads, **kwargs
    ):
        super().__init__()
        layers = []
        layers.extend(
            [
                (
                    TransformerConv(
                        input_size, layer_size, heads=heads, edge_dim=edge_size
                    ),
                    "x, edge_index, edge_attr -> x1",
                ),
                (nn.Linear(input_size, heads * layer_size), "x -> x2"),
                # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                (adder(), "x1, x2 -> x"),
                (BatchNorm(heads * layer_size), "x -> x"),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        for layer in range(hidden_layers):
            layers.extend(
                [
                    (
                        TransformerConv(
                            heads * layer_size,
                            layer_size,
                            heads=heads,
                            edge_dim=edge_size,
                        ),
                        "x, edge_index, edge_attr -> x1",
                    ),
                    (nn.Linear(heads * layer_size, heads * layer_size), "x -> x2"),
                    # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                    (adder(), "x1, x2 -> x"),
                    (BatchNorm(heads * layer_size), "x -> x"),
                    nn.ELU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(heads * layer_size, 1))
        self.model = Sequential("x, edge_index, edge_attr", layers)

    def forward(self, x, edge_index, edge_attr, **kwargs):
        return self.model(x, edge_index, edge_attr)


class NNNet(torch.nn.Module):
    def __init__(self, input_size, edge_size, layer_size, **kwargs):
        super().__init__()
        self.lin0 = torch.nn.Linear(input_size, layer_size)

        nnet = nn.Sequential(
            nn.Linear(edge_size, layer_size),
            nn.ReLU(),
            Linear(layer_size, layer_size * layer_size),
        )
        self.conv = NNConv(layer_size, layer_size, nnet, aggr="mean")
        self.gru = nn.GRU(layer_size, layer_size)

        # self.set2set = Set2Set(layer_size, processing_steps=3)
        self.lin1 = torch.nn.Linear(layer_size, layer_size)
        self.lin2 = torch.nn.Linear(layer_size, 1)

    def forward(self, x, edge_index, edge_attr, **kwargs):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class GCNNet(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, layer_size, dropout, **kwargs):
        super().__init__()
        layers = []
        layers.extend(
            [
                (GCNConv(input_size, layer_size), "x, edge_index -> x1"),
                (nn.Linear(input_size, layer_size), "x -> x2"),
                # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                (adder(), "x1, x2 -> x"),
                (BatchNorm(heads * layer_size), "x -> x"),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        for layer in range(hidden_layers):
            layers.extend(
                [
                    (GCNConv(layer_size, layer_size), "x, edge_index -> x1"),
                    (nn.Linear(layer_size, layer_size), "x -> x2"),
                    # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                    (adder(), "x1, x2 -> x"),
                    (BatchNorm(heads * layer_size), "x -> x"),
                    nn.ELU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(layer_size, 1))
        self.model = Sequential("x, edge_index", layers)

    def forward(self, x, edge_index, **kwargs):
        return self.model(x, edge_index)


class SAGENet(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, layer_size, dropout, **kwargs):
        super().__init__()
        layers = []
        layers.extend(
            [
                (SAGEConv(input_size, layer_size), "x, edge_index -> x1"),
                (nn.Linear(input_size, layer_size), "x -> x2"),
                # (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
                (adder(), "x1, x2 -> x"),
                (BatchNorm(layer_size), "x -> x"),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        for layer in range(hidden_layers):
            layers.extend(
                [
                    (SAGEConv(layer_size, layer_size), "x, edge_index -> x1"),
                    (nn.Linear(layer_size, layer_size), "x -> x2"),
                    (BatchNorm(layer_size), "x -> x"),
                    (adder, "x1, x2 -> x"),
                    nn.ELU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(layer_size, 1))
        self.model = Sequential("x, edge_index", layers)

    def forward(self, x, edge_index, **kwargs):
        return self.model(x, edge_index)


class SitesPrediction(pl.LightningModule):
    def __init__(self, model_name, lr, pos_weight, batch_size=1024, **config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.validation_step_outputs = []
        self.mcc = MatthewsCorrCoef(task="binary")  # , num_classes=2)

        # self.example_input_array = dict(x=torch.rand(4,61), edge_index=barabasi_albert_graph(num_nodes=4, num_edges=3), edge_attr=torch.rand(3, 11), y=torch.rand(4,1))
        self.model_name = model_name
        if self.model_name == "GATNet":
            self.model = GATNet(**config)
        elif self.model_name == "GCNNet":
            self.model = GCNNet(**config)
        elif self.model_name == "TransformerNet":
            self.model = TransformerNet(**config)
        elif self.model_name == "SAGENet":
            self.model = SAGENet(**config)
        elif self.model_name == "NNNet":
            self.model = NNNet(**config)
        else:
            raise ValueError("No model")
        self.loss_module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float16)
        )

    # def forward(self, data):
    def forward(self, x, edge_index, edge_attr):
        # x, edge_attr, edge_index = data["x"], data["edge_attr"], data["edge_index"]
        return self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # y_hat = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # y = data['y'].detach()
        # #y_tensor = torch.tensor(x.shape[0], dtype=torch.float32)[y]
        # loss = self.loss_module(y_hat, y)
        # preds = (y_hat>0).int()
        # #score = f1_score(data.y.detach().numpy(), preds, average='micro') if preds.sum() > 0 else 0.0
        # #mcc = MatthewsCorrCoef(num_classes=2)
        # score = self.mcc(preds, y.int())
        # #ck = CohenKappa(num_classes=2)
        # #score = ck(preds, data.y.detach().int())
        # return loss, score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))  # type: ignore
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3
        )
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        loss, score = self.calc_loss_score(
            batch.x, batch.edge_index, batch.edge_attr, batch.y
        )
        self.log_dict(
            {"train_loss": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log_dict(
            {"train_score": score, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, score = self.calc_loss_score(
            batch.x, batch.edge_index, batch.edge_attr, batch.y
        )

        self.validation_step_outputs.append([loss, score])
        self.log_dict(
            {"val_loss": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log_dict(
            {"val_score": score, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        loss, score = (
            torch.stack((torch.tensor(self.validation_step_outputs),))
            .mean(dim=1)
            .flatten()
        )

        self.log_dict(
            {"hp_metric": loss, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        _, score = self.calc_loss_score(
            batch.x, batch.edge_index, batch.edge_attr, batch.y
        )
        # self.log("hp_metric", score, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=128)
        self.log_dict(
            {"test_score": score, "step": float(self.current_epoch)},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, edge_attr, edge_index = batch["x"], batch["edge_attr"], batch["edge_index"]
        y_hat = self.model(x=x, edge_attr=edge_attr, edge_index=edge_index)
        y_hat = (y_hat > 0.0).float()
        return y_hat

    def calc_loss_score(self, x, edge_index, edge_attr, y):
        y_hat = self.forward(x, edge_index, edge_attr)
        # y = data["y"]
        # print(len(data), y.shape, y_hat.shape)
        loss = self.loss_module(y_hat, y)
        y_hat = y_hat > 0
        score = self.mcc(y_hat.int(), y.int())

        return loss, score
