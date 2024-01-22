import logging
import os
import warnings
from importlib import resources as impresources
from typing import Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from rdkit.Chem import PandasTools
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures

from .mol_dataset import MolDataset, MolPairDataset
from .pka_models import PredictpKa
from .sites_models import SitesPrediction
from .transforms import IncludeEnergy, ToTensor

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

warnings.filterwarnings("ignore", ".*If your intention is to run Lightning on SLURM*")
warnings.simplefilter(action="ignore", category=FutureWarning)

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


def load_mol_dataset(
    root: str,
    filename: str,
    tautomerize: bool,
    name_col: str,
    mol_col: str,
    mp: Union[bool, int],
) -> MolDataset:
    """Load a dataset of molecules.

    Args:
        root (str): root directory for the project
        filename (str): filename of the dataset
        tautomerize (bool): whether to tautomerize the molecules
        name_col (str): name of the column with the molecule name
        mol_col (str): name of the column with the molecule
        mp (Union[bool, int]): number of processes to use for multiprocessing

    Returns:
        MolDataset: dataset of molecules
    """
    dataset = MolDataset(
        root=root,
        filename=filename,
        tautomerize=tautomerize,
        name_col=name_col,
        mol_col=mol_col,
        mp=mp,
    )
    return dataset


def load_mol_pair_dataset(
    root: str,
    filename: str,
    name_col: str,
    mol_col: str,
    idx_col: str,
    type_col: str,
    mp: Union[bool, int],
) -> MolPairDataset:
    """Load a dataset of molecules and their protonation/deprotonation sites.

    Args:
        root (str): root directory for the project
        filename (str): filename of the dataset
        name_col (str): name of the column with the molecule name
        mol_col (str): name of the column with the RDKit molecule
        idx_col (str): name of the column with the protonation/deprotonation site index
        type_col (str): name of the column with the protonation/deprotonation type
        mp (Union[bool, int]): number of processes to use for multiprocessing

    Returns:
        MolPairDataset: dataset of molecules and their protonation/deprotonation sites
    """

    transforms = Compose(
        [
            ToTensor(["y", "d_energy"]),
            NormalizeFeatures(["global_attr_prot", "global_attr_deprot"]),
            IncludeEnergy(["global_attr_prot", "global_attr_deprot"], "d_energy", -1),
        ]
    )
    pair_dataset = MolPairDataset(
        root=root,
        filename=filename,
        name_col=name_col,
        mol_col=mol_col,
        idx_col=idx_col,
        type_col=type_col,
        type_values=("basic", "acidic"),
        xtb=True,
        mp=mp,
        transform=transforms,
    )
    return pair_dataset


def load_models() -> Tuple[SitesPrediction, SitesPrediction, PredictpKa]:
    """
    Load the trained models.

    Returns:
        Tuple[SitesPrediction, SitesPrediction, PredictpKa]: protonation model, deprotonation model, pKa model
    """
    from . import models

    models_path = impresources.files(models)
    prot_model = SitesPrediction.load_from_checkpoint(
        os.path.join(models_path, "protonation_model.ckpt"),
        map_location=torch.device("cpu"),
    )
    deprot_model = SitesPrediction.load_from_checkpoint(
        os.path.join(models_path, "deprotonation_model.ckpt"),
        map_location=torch.device("cpu"),
    )
    pka_model = PredictpKa.load_from_checkpoint(
        os.path.join(models_path, "pka_model.ckpt"), map_location=torch.device("cpu")
    )
    return prot_model, deprot_model, pka_model


def predict_sites(dataset: MolDataset, model: SitesPrediction) -> list:
    """
    Predict protonation sites for a dataset of molecules.

    Args:
        dataset (MolDataset): dataset of molecules
        model (SitesPrediction): protonation\deprotonation model

    Returns:
        list: list of indices of protonation\deprotonation sites
    """
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )

    sites_predictions = trainer.predict(model, DataLoader(dataset, batch_size=1))
    sites_indices = [
        x.squeeze().nonzero().reshape(-1).tolist() for x in sites_predictions
    ]
    return sites_indices


def predict_pka(dataset: MolPairDataset, model: PredictpKa) -> torch.Tensor:
    """
    Predict pKa values for a dataset of molecules.

    Args:
        dataset (MolPairDataset): dataset of molecules
        model (PredictpKa): pKa model

    Returns:
        torch.Tensor: predicted pKa values
    """
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    pka_predictions = trainer.predict(
        model, DataLoader(dataset, batch_size=1, follow_batch=["x_deprot", "x_prot"])
    )
    pka_predictions = torch.cat(pka_predictions).squeeze()
    return pka_predictions


def make_sites_prediction_files(
    root: str,
    dataset: MolDataset,
    prot_indices: list,
    deprot_indices: list,
    output: str,
) -> None:
    """Make an sdf file with the molecules and their reaction site index.

    Args:
        root (str): root directory for the project
        dataset (MolDataset): dataset of molecules
        prot_indices (list): protonation sites indices
        deprot_indices (list): deprotonation sites indices
        output (str): output file name

    Returns:
        None
    """

    mol_list = []
    for data, prot_idx, deprot_idx in zip(dataset, prot_indices, deprot_indices):
        data_dict = {}
        data_dict["ROMol"] = data.mol
        data_dict["name"] = data.name
        for i in prot_idx:
            prot_dict = data_dict.copy()
            prot_dict["idx"] = i
            prot_dict["pka_type"] = "acidic"
            mol_list.append(prot_dict)
        for i in deprot_idx:
            deprot_dict = data_dict.copy()
            deprot_dict["idx"] = i
            deprot_dict["pka_type"] = "basic"
            mol_list.append(deprot_dict)
    PandasTools.WriteSDF(
        pd.DataFrame(mol_list),
        f"{root}/raw/{output}",
        molColName="ROMol",
        idName="name",
        properties=["idx", "pka_type"],
    )


def run_prediction_pipeline(
    root: str,
    filename: str,
    tautomerize: bool,
    name_col: str,
    mol_col: str,
    mp: Union[bool, int],
    output: str,
    **kwargs,
) -> None:
    """Run the prediction pipeline."""
    dataset = load_mol_dataset(
        root=root,
        filename=filename,
        tautomerize=tautomerize,
        name_col=name_col,
        mol_col=mol_col,
        mp=mp,
    )
    prot_model, deprot_model, pka_model = load_models()
    prot_indices = predict_sites(dataset, prot_model)
    deprot_indices = predict_sites(dataset, deprot_model)
    if not any(prot_indices) and not any(deprot_indices):
        print("No protonation/deprotonation sites were found.")
        print("Output fill will not be created.")
    else:
        make_sites_prediction_files(root, dataset, prot_indices, deprot_indices, output)
        pair_dataset = load_mol_pair_dataset(
            root=root,
            filename=output,
            name_col=name_col,
            mol_col="ROMol",
            idx_col="idx",
            type_col="pka_type",
            mp=mp,
        )

        pka_predictions = predict_pka(pair_dataset, pka_model)
        df = PandasTools.LoadSDF(
            f"{root}/raw/{output}",
            embedProps=True,
            removeHs=False,
            includeFingerprints=False,
            idName=name_col,
            molColName="ROMol",
        )
        df["pka"] = pka_predictions

        PandasTools.WriteSDF(
            df,
            f"{root}/output/{output}",
            molColName="ROMol",
            idName=name_col,
            properties=["idx", "pka_type", "pka"],
        )
        print(f"Predictions saved to {root}/output/{output}")
