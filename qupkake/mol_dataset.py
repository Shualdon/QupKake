import logging
import multiprocessing
import os
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing.pool import Pool
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, PandasTools
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from .featurizer import Featurizer
from .mol_utils import Tautomerize

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


class MolPairData(Data):
    """Representation of a pair of molecules."""

    def __init__(
        self,
        edge_index_prot=None,
        x_prot=None,
        edge_attr_prot=None,
        global_attr_prot=None,
        mol_prot=None,
        edge_index_deprot=None,
        x_deprot=None,
        edge_attr_deprot=None,
        global_attr_deprot=None,
        mol_deprot=None,
        d_energy=None,
        y=None,
        name=None,
    ):
        super().__init__()

        self.x_prot = x_prot
        self.edge_attr_prot = edge_attr_prot
        self.edge_index_prot = edge_index_prot
        self.global_attr_prot = global_attr_prot
        self.mol_deprot = mol_deprot

        self.x_deprot = x_deprot
        self.edge_attr_deprot = edge_attr_deprot
        self.edge_index_deprot = edge_index_deprot
        self.global_attr_deprot = global_attr_deprot
        self.mol_prot = mol_prot

        self.name = name
        self.d_energy = d_energy
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_prot":
            return self.x_prot.size(0)
        if key == "edge_index_deprot":
            return self.x_deprot.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MolDatasetAbstract(Dataset, ABC):
    """Abstract class for MolDataset instances"""

    def __init__(
        self,
        root: str,
        filename: str,
        name_col: str = "name",
        mol_col: str = "ROMol",
        smiles_col: str = "smiles",
        other_cols: Tuple[str, list[str]] = None,
        mp: Union[bool, int] = False,
        transform: Any = None,
        pre_transform: Any = None,
        **kwargs,
    ):
        self.filename = filename
        self.mol_col = mol_col
        self.smiles_col = smiles_col
        self.other_cols = other_cols
        self.mp = mp
        self.kwargs = kwargs
        self.num_processes = (
            multiprocessing.cpu_count() if isinstance(self.mp, bool) else self.mp
        )
        self.data = None
        self.data_list = []

        # logging.basicConfig(
        #     filename=f"{root}/logs/error_log.txt",
        #     level=logging.ERROR,
        #     format="%(asctime)s [%(levelname)s]: %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )

        super().__init__(root, transform, pre_transform, **kwargs)

    def _read_file(self, embed=False):
        _, ext = os.path.splitext(self.filename)
        if ext == ".sdf":
            return self._read_sdf()
        elif ext == ".csv":
            return self._read_csv(embed)
        else:
            raise SyntaxError(
                f"File extension {ext} is not supported. Use .sdf or .csv instead."
            )

    def _save_file(self):
        _, ext = os.path.splitext(self.filename)
        if ext == ".sdf":
            PandasTools.WriteSDF(
                self.data, self.raw_paths[0], properties=list(self.data.columns)
            )
        elif ext == ".csv":
            self.data.to_csv(self.raw_paths[0], index=False)

    def _embed_mol(self, row):
        mol = row[self.mol_col]
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, maxAttempts=1000)
            return mol
        except:
            return mol

    def _read_csv(self, embed):
        datafile = pd.read_csv(
            self.raw_paths[0],
            dtype={
                self.name_col: str,
                self.smiles_col: str,
            },
        ).reset_index(drop=True)
        datafile = datafile.astype(self._col_types(datafile))
        if embed:
            print("Converting and embeding SMILES to Mols.")
            PandasTools.AddMoleculeColumnToFrame(
                datafile, smilesCol=self.smiles_col, molCol=self.mol_col
            )
            datafile[self.mol_col] = datafile.apply(self._embed_mol, axis=1)
        return datafile

    def _read_sdf(self):
        datafile = PandasTools.LoadSDF(
            self.raw_paths[0],
            idName=self.name_col,
            molColName=self.mol_col,
            smilesName=self.smiles_col,
            embedProps=True,
            isomericSmiles=True,
            removeHs=False,
        )
        return datafile.astype(self._col_types(datafile))

    def _check_columns(self) -> None:
        if self.other_cols:
            if isinstance(self.other_cols, str):
                self.other_cols = [self.other_cols]

            for col in self.other_cols:
                if col not in self.data.columns:
                    raise ValueError(f"{col} not in file.")

    def _col_types(self, df) -> dict[str]:
        dtypes = {}
        if self.name_col in df.columns:
            dtypes[self.name_col] = str

        if self.smiles_col in df.columns:
            dtypes[self.smiles_col] = str

        return dtypes

    def _get_data(self, embed: bool):
        if self.data is not None:
            if self.mol_col in self.data.columns and not embed:
                print("returning data from self.data")
                return self.data

        self.data = self._read_file(embed)
        return self.data

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return self.filename

    def download(self):
        pass

    def process(self):
        """Process the dataset to the processed data folder."""
        self.data = self._get_data(embed=True)
        if self.mp:
            L = list(range(self.num_processes))[::-1]
            chunks = np.array_split(self.data, self.num_processes)
            with Pool(self.num_processes) as pool:
                self.data = pd.concat(
                    pool.starmap(self._process_chunk, zip(chunks, L)), ignore_index=True
                )
        else:
            self.data = self._process_chunk(self.data, 0)
        self._save_file()

    def _handle_processing_error(self, row, error):
        """Handle errors during processing"""
        logger.error(f"Error processing row {row.name}")
        logger.error(f"Error: {error}")
        logger.error(traceback.format_exc())

    def _save_processed_data(self, file_name, data):
        """Save processed data to file"""
        file_path = os.path.join(self.processed_dir, file_name)
        torch.save(data, file_path)

    def _load_processed_data(self, file_name):
        """Load processed data from file"""
        file_path = os.path.join(self.processed_dir, file_name)
        return torch.load(file_path)

    @abstractmethod
    def _process_chunk(self, chunk):
        pass


class MolPairDataset(MolDatasetAbstract):
    """A dataset of pairs of molecules

    Arguments:
        root: str
            Root directory.
        filename: str
            Filename in 'root'/raw/ directory.
        name_col: str
            Column name for the molecules names in the file.
        mol_col: str
            Column name for the molecules RDKit mol objects in the file.
        smiles_col: str
            Column name for the molecules SMILES strings in the file.
        type_col: str
            Column name for the molecules pKa type in the file.
        type_values: Tuple[str]
            Values for the pKa type column. The first value is for basic pKa and the second for acidic pKa.
        idx_col: str
            Column name for the molecules atom index in the file.
        y_col: str
            Column name for the pKa column.
        other_cols: str or list of strings
            Column names with other information to include in the graph.
        transform: Any
            Tranformation function on the data after processing.
        pre_transform: Any:
            Tranformation function on the data before processing.
    """

    def __init__(
        self,
        root: str,
        filename: str,
        name_col: str = "name",
        mol_col: str = "ROMol",
        smiles_col: str = "smiles",
        type_col: str = "pka_type",
        type_values: Tuple[str] = ("basic", "acidic"),
        idx_col: str = "idx",
        y_col: str = None,
        other_cols: Tuple[str, list[str]] = None,
        mp: bool = False,
        transform: Any = None,
        pre_transform: Any = None,
        **kwargs,
    ):
        self.filename = filename
        self.name_col = name_col
        self.mol_col = mol_col
        self.smiles_col = smiles_col
        self.y_col = y_col
        self.type_col = type_col
        self.type_values = type_values
        self.idx_col = idx_col
        self.other_cols = other_cols
        self.mp = mp
        self.kwargs = kwargs

        # root: str,
        # filename: str,
        # name_col: str = None,
        # mol_col: str = None,
        # smiles_col: str = None,
        # other_cols: Tuple[str, list[str]] = None,
        # mp: bool = False,
        # transform: Any = None,
        # pre_transform: Any = None,
        # **kwargs,
        super().__init__(
            root,
            filename,
            name_col,
            mol_col,
            smiles_col,
            other_cols,
            mp,
            transform,
            pre_transform,
        )

    def _check_columns(self) -> None:
        if self.other_cols is not None:
            if isinstance(self.other_cols, str):
                self.other_cols = [self.other_cols]
            if isinstance(self.other_cols, list):
                for col in self.other_cols:
                    if col not in self.data.columns:
                        raise ValueError(f"{col} not in file.")

    @property
    def processed_file_names(self) -> list[str]:
        """If these files are found in raw_dir, processing is skipped"""
        self.data = self._get_data(False)
        self._check_columns()
        # return f'{self.data_name}_{self.set}.pt'
        return [
            f"{name}_{idx}_{conjugate}_pair.pt"
            for (name, idx, conjugate) in zip(
                list(self.data[self.name_col]),
                list(self.data[self.idx_col]),
                list(self.data[self.type_col]),
            )
        ]

    def _process_chunk(self, chunk, chunk_pos) -> pd.DataFrame:
        """Processing a chunk of data from the dataframe"""
        bad_idx = []
        pbar = tqdm(chunk.iterrows(), total=len(chunk), position=chunk_pos)
        for index, row in pbar:
            pbar.set_description("Processing %s" % row[self.name_col])

            file_name = (
                f"{row[self.name_col]}_{row[self.idx_col]}_{row[self.type_col]}_pair.pt"
            )
            try:
                if not os.path.exists(
                    os.path.join(
                        self.processed_dir,
                        file_name,
                    )
                ):
                    self._process_row_with_retry(row)
                else:
                    self._load_processed_data(file_name)
            except Exception as e:
                bad_idx.append(index)
                self._handle_processing_error(row, e)
        chunk = chunk.drop(bad_idx).reset_index(drop=True)
        # return the chunk
        return chunk

    def _process_row_with_retry(self, row):
        """Process a row with retry mechanism"""
        file_name = (
            f"{row[self.name_col]}_{row[self.idx_col]}_{row[self.type_col]}_pair.pt"
        )
        tries = 5
        while tries > 0:
            try:
                data = self._make_molpair(row)
                self.data_list.append(data)
                self._save_processed_data(file_name, data)
                break
            except Exception as e:
                tries -= 1
                if tries == 0:
                    self._handle_retry_error(row, e)
                    raise (Exception("Error processing row"))

    def _handle_retry_error(self, row, error):
        """Handle errors during retry attempts"""
        logging.error(f"{row.name} failed all 5 retry attempts")
        logging.error(f"Error: {error}")
        logging.error(traceback.format_exc())

    def _conjugate_mol(self, mol, atom_index, pka_type) -> Chem.Mol:
        """Protonates or deprotonates a molecule"""

        conjugate = deepcopy(mol)
        conjugate = Chem.RemoveHs(conjugate, updateExplicitCount=True)

        # If basic
        if pka_type == self.type_values[0]:
            atom = conjugate.GetAtomWithIdx(atom_index)
            charge = atom.GetFormalCharge()
            exp_hs = atom.GetNumExplicitHs()
            atom.SetFormalCharge(charge + 1)
            atom.SetNumExplicitHs(exp_hs + 1)
        # If acidic
        elif pka_type == self.type_values[1]:
            # Fixing caboxylic acids index
            carb_acid = Chem.MolFromSmarts("[OH]C=O")
            for match in conjugate.GetSubstructMatches(carb_acid):
                if atom_index in match:
                    num_hs = [
                        conjugate.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)
                        for i in match
                    ]
                    if sum(num_hs) == 0:
                        break
                    atom_index = match[num_hs.index(1)]
                    break
            atom = conjugate.GetAtomWithIdx(atom_index)
            charge = atom.GetFormalCharge()
            old_exp_hs = atom.GetNumExplicitHs()
            atom.SetFormalCharge(charge - 1)
            new_exp_hs = atom.GetNumExplicitHs()
            if old_exp_hs == new_exp_hs and new_exp_hs > 0:
                atom.SetNumExplicitHs(new_exp_hs - 1)

        conjugate.UpdatePropertyCache(strict=True)
        conjugate = Chem.AddHs(conjugate, addCoords=True)

        return conjugate

    def _graphs_to_molpair(self, prot, deprot, name, y) -> MolPairData:
        d_energy = None
        if hasattr(prot, "energy") and hasattr(deprot, "energy"):
            d_energy = torch.abs(torch.tensor([prot.energy - deprot.energy]))

        mol_pair = MolPairData(
            edge_index_prot=prot.edge_index,
            x_prot=prot.x,
            edge_attr_prot=prot.edge_attr,
            global_attr_prot=prot.global_attr,
            mol_prot=prot.mol,
            edge_index_deprot=deprot.edge_index,
            x_deprot=deprot.x,
            edge_attr_deprot=deprot.edge_attr,
            global_attr_deprot=deprot.global_attr,
            mol_deprot=deprot.mol,
            d_energy=d_energy,
            y=y,
            name=name,
        )
        return mol_pair

    def _make_molpair(self, row) -> MolPairData:
        """Create a MolPair object from a row in a dataframe."""
        mol = row[self.mol_col]
        name = row[self.name_col]
        atom_index = int(row[self.idx_col])
        pka_type = row[self.type_col]
        y = None
        if self.y_col in row.index:
            y = row[self.y_col]

        # Get mol graph
        try:
            mol_graph = Featurizer(
                mol=mol, name=f"{name}", num_processes=self.num_processes, **self.kwargs
            ).data
            conjugate_mol = self._conjugate_mol(mol, atom_index, pka_type)
            conjugate_graph = Featurizer(
                mol=conjugate_mol,
                name=f"{name}_c{atom_index}",
                y=y,
                num_processes=self.num_processes,
                **self.kwargs,
            ).data
            # If basic
            if pka_type == self.type_values[0]:
                mol_pair = self._graphs_to_molpair(conjugate_graph, mol_graph, name, y)
            # If acidic
            elif pka_type == self.type_values[1]:
                mol_pair = self._graphs_to_molpair(mol_graph, conjugate_graph, name, y)
            else:
                raise ValueError("pKa Type not recognized.")
            return mol_pair
        except Exception as e:
            self._handle_processing_error(row, e)
            raise (Exception("Error processing row"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.data)

    def get(self, idx):
        """- Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        name = self.data.loc[idx, self.name_col]
        atom_idx = self.data.loc[idx, self.idx_col]
        conjugate = self.data.loc[idx, self.type_col]
        data = torch.load(
            os.path.join(self.processed_dir, f"{name}_{atom_idx}_{conjugate}_pair.pt")
        )
        return data


class MolDataset(MolDatasetAbstract):
    """
    Creates a pyg Dataset from a csv file with SMILES.
        Arguments:
            root: str
                Root directory.
            filename: str
                CSV filename in 'root'/raw/ directory.
            tautomerize: bool
                Whether to tzutomerize the molecules.
            name_col: str
                Column name for the molecules names in the csv file.
            smiles_col: str
                Column name for the molecules SMILES strings in the csv file.
            y_col: str
                Column name for the graph output
            other_cols: str or list of strings
                Column names with other information to include in the graph.
            transform: Any
                Tranformation function on the data after processing.
            pre_transform: Any:
                Tranformation function on the data before processing.
    """

    def __init__(
        self,
        root: str,
        filename: str,
        tautomerize: bool = True,
        data_name: str = None,
        name_col: str = "name",
        mol_col: str = "mol",
        smiles_col: str = "smiles",
        y_col: str = None,
        other_cols: Tuple[str, list[str]] = None,
        mp: bool = False,
        transform: Any = None,
        pre_transform: Any = None,
        **kwargs,
    ):
        self.filename = filename
        self.tautomerize = tautomerize
        self.data_name = data_name
        self.name_col = name_col
        self.mol_col = mol_col
        self.smiles_col = smiles_col
        self.y_col = y_col
        self.other_cols = other_cols
        self.mp = mp
        self.kwargs = kwargs
        # self.num_classes = 1
        super().__init__(
            root,
            filename,
            name_col,
            mol_col,
            smiles_col,
            other_cols,
            mp,
            transform,
            pre_transform,
        )

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        self.data = self._get_data(False)
        self._check_columns()
        # print(len(self.data))
        # return f'{self.data_name}_{self.set}.pt'
        if self.data_name:
            return [
                f"{name}_{self.data_name}.pt" for name in list(self.data[self.name_col])
            ]

        return [f"{name}.pt" for name in list(self.data[self.name_col])]

    @property
    def num_classes(self):
        return 1

    def _make_output_dir(self, output_dir):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            return os.path.abspath(output_dir)
        else:
            return None

    def opt_mol(self, mol, addCoords=True):
        """Embeds the molecule"""

        mol = Chem.AddHs(mol, addCoords=True)
        conf = AllChem.EmbedMolecule(
            mol, useRandomCoords=True, ignoreSmoothingFailures=True
        )
        if conf == -1:
            mol = Chem.RemoveHs(mol)
            AllChem.EmbedMolecule(
                mol, useRandomCoords=True, ignoreSmoothingFailures=True
            )
            mol = Chem.AddHs(mol, addCoords=True)

        AllChem.MMFFOptimizeMolecule(mol)
        # mol = Chem.RemoveHs(mol)
        return mol

    def _get_graph(self, row):
        # mol = row[self.mol_col]
        name = row[self.name_col]
        smiles = row[self.smiles_col]

        mol = (
            self.opt_mol(row[self.mol_col])
            if self.mol_col in row
            else self.opt_mol(Chem.MolFromSmiles(smiles))
        )

        other_info = {}
        if self.other_cols:
            for col in self.other_cols:
                other_info[col] = row[col]

        if self.y_col in row:
            y = row[self.y_col]
        else:
            y = None

        if self.tautomerize:
            tautomers = Tautomerize(
                mol=mol,
                name=name,
                check_exists=False,
                keep_mol=False,
                num_processes=self.num_processes,
                # mol_dir=os.path.join(self.root, "molecules"),
                **self.kwargs,
            )
            mol = tautomers.lowest_tautomer
            name = tautomers.lowest_tautomer_name

        mol_data = Featurizer(
            mol=mol,
            name=name,
            y=y,
            num_processes=self.num_processes,
            **{**other_info, **self.kwargs},
        )
        return mol_data.data

    def _process_chunk(self, chunk, chunk_pos) -> pd.DataFrame:
        self.data_list = []
        bad_idx = []
        pbar = tqdm(chunk.iterrows(), total=len(chunk), position=chunk_pos)
        for index, row in pbar:
            pbar.set_description("Processing %s" % row[self.name_col])
            if self.data_name:
                file_name = f"{row[self.name_col]}_{self.data_name}.pt"
            else:
                file_name = f"{row[self.name_col]}.pt"
            try:
                if not os.path.exists(
                    os.path.join(
                        self.processed_dir,
                        file_name,
                    )
                ):
                    self._process_row_with_retry(row)
                else:
                    self._load_processed_data(file_name)
            except Exception as e:
                bad_idx.append(index)
                self._handle_processing_error(row, e)
                pass

        chunk = chunk.drop(bad_idx).reset_index(drop=True)
        return chunk

    def _process_row_with_retry(self, row):
        """Process a row with retry mechanism"""
        if self.data_name:
            file_name = f"{row[self.name_col]}_{self.data_name}.pt"
        else:
            file_name = f"{row[self.name_col]}.pt"
        tries = 5
        while tries > 0:
            try:
                data = self._get_graph(row)
                self.data_list.append(data)
                self._save_processed_data(file_name, data)
                break
            except Exception as e:
                tries -= 1
                if tries == 0:
                    self._handle_retry_error(row, e)
                    raise (Exception("Error processing row"))

    def _handle_retry_error(self, row, error):
        """Handle errors during retry attempts"""
        logging.error(f"{row.name} failed all 5 retry attempts")
        logging.error(f"Error: {error}")
        logging.error(traceback.format_exc())

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.data)

    def get(self, idx):
        """- Equivalent to __getitem__ in pytorch"""
        name = self.data.loc[idx, self.name_col]
        if self.data_name:
            file_name = f"{name}_{self.data_name}.pt"
        else:
            file_name = f"{name}.pt"
        return self._load_processed_data(file_name)
