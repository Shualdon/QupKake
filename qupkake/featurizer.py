import ast
import glob
import logging
import traceback
from abc import ABC
from typing import Any, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, Lipinski
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data

from .xtbp import XTBP, RunXTB

logger = logging.getLogger(__name__)

ATOM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "I",
    "B",
    "K",
    "Li",
    "Unknown",
]
N_HEAVY_NEIGHBORS = [0, 1, 2, 3, 4, "MoreThanFour"]
FORMAL_CHARGE = [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
HYBRIDIZATION = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
CHIRALITY = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]
H_IMPLICIT = [0, 1, 2, 3, 4, "MoreThanFour"]
HBA_SMARTS = "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0]),$([F;!$(F-*-F)])]"
HBD_SMARTS = "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"

ATOM_FEATURES = {
    "AtomType": (True, len(ATOM_LIST)),
    "HImplicit": (True, (len(H_IMPLICIT), 1)),
    "NHeavyNeighbors": (True, len(N_HEAVY_NEIGHBORS)),
    "FormalCharge": (True, len(FORMAL_CHARGE)),
    "Hybridization": (True, len(HYBRIDIZATION)),
    "IsInRing": (True, 1),
    "IsAromatic": (True, 1),
    "AtomicMass": (True, 1),
    "VDWRadius": (True, 1),
    "CovalenRadius": (True, 1),
    "Chirality": (True, len(CHIRALITY)),
    "IsHBA": (True, 1),
    "IsHBD": (True, 1),
    "xTB-q": (True, 1),
    "xTB-cn": (True, 1),
    "xTB-Alpha": (True, 1),
    "xTB-Fukui": (True, 3),
}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREOCHEM = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]

BOND_FEATURES = {
    "BondType": (True, len(BOND_TYPES)),
    "IsConjugated": (True, 1),
    "IsInRing": (True, 1),
    "Stereochemistry": (True, len(STEREOCHEM)),
    "xTB-WBO": (True, 1),
}

MOL_FEATURES = {
    "RadiusGyration": (True, 1),
    "Spherocity": (True, 1),
    "Asphericity": (True, 1),
    "Eccentricity": (True, 1),
    "FractionCSP3": (True, 1),
    "xTB-Energy": (True, 1),
    "xTB-Charge": (True, 1),
}


class FeaturizerAbstract(ABC):
    """Abstract base class for featurizers."""

    def one_hot_encoding(self, x, permitted_list) -> list:
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [
            int(boolean_value)
            for boolean_value in list(map(lambda s: x == s, permitted_list))
        ]
        return binary_encoding


class AtomFeaturizer(FeaturizerAbstract):
    """Generates features for an RDKit atom."""

    def __init__(self, xtb: bool = True):
        self.xtb = xtb

    def get_atom_features(self, atom, attributes) -> torch.FloatTensor:
        """
        Takes an RDKit atom object and xTB attributes dictionary as input and gives a 1d-numpy array of atom features as output.
        """
        atom_feature_vector = []

        if ATOM_FEATURES["AtomType"][0]:
            permitted_list_of_atoms = ATOM_LIST
            if ATOM_FEATURES["HImplicit"][0] == False:
                permitted_list_of_atoms = ["H"] + permitted_list_of_atoms
            atom_feature_vector += self.one_hot_encoding(
                str(atom.GetSymbol()), permitted_list_of_atoms
            )

        if ATOM_FEATURES["NHeavyNeighbors"][0]:
            atom_feature_vector += self.one_hot_encoding(
                int(atom.GetDegree() - atom.GetTotalNumHs(includeNeighbors=True)),
                N_HEAVY_NEIGHBORS,
            )

        if ATOM_FEATURES["FormalCharge"][0]:
            atom_feature_vector += self.one_hot_encoding(
                int(atom.GetFormalCharge()), FORMAL_CHARGE
            )

        if ATOM_FEATURES["Hybridization"][0]:
            atom_feature_vector += self.one_hot_encoding(
                str(atom.GetHybridization()), HYBRIDIZATION
            )

        if ATOM_FEATURES["IsInRing"][0]:
            atom_feature_vector += [int(atom.IsInRing())]

        if ATOM_FEATURES["IsAromatic"][0]:
            atom_feature_vector += [int(atom.GetIsAromatic())]

        if ATOM_FEATURES["AtomicMass"][0]:
            atom_feature_vector += [float((atom.GetMass() - 10.812) / 116.092)]

        if ATOM_FEATURES["VDWRadius"][0]:
            atom_feature_vector += [
                float(
                    (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6
                )
            ]

        if ATOM_FEATURES["CovalenRadius"][0]:
            atom_feature_vector += [
                float(
                    (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)
                    / 0.76
                )
            ]

        if ATOM_FEATURES["Chirality"][0]:
            atom_feature_vector += self.one_hot_encoding(
                str(atom.GetChiralTag()), CHIRALITY
            )

        if ATOM_FEATURES["HImplicit"][0]:
            atom_feature_vector += self.one_hot_encoding(
                int(atom.GetTotalNumHs(includeNeighbors=True)), H_IMPLICIT
            )

        if ATOM_FEATURES["IsHBA"][0]:
            HAcceptorSmarts = Chem.MolFromSmarts(HBA_SMARTS)
            HAcceptors = [
                x[0] for x in atom.GetOwningMol().GetSubstructMatches(HAcceptorSmarts)
            ]
            atom_feature_vector += [1 if atom.GetIdx() in HAcceptors else 0]

        if ATOM_FEATURES["IsHBD"][0]:
            HDonorSmarts = Chem.MolFromSmarts(HBD_SMARTS)
            HDonors = [
                x[0] for x in atom.GetOwningMol().GetSubstructMatches(HDonorSmarts)
            ]
            atom_feature_vector += [1 if atom.GetIdx() in HDonors else 0]

        if self.xtb:
            if ATOM_FEATURES["xTB-q"][0]:
                atom_feature_vector += [attributes["atomprop"]["q"][atom.GetIdx()]]

            if ATOM_FEATURES["xTB-cn"][0]:
                atom_feature_vector += [attributes["atomprop"]["convcn"][atom.GetIdx()]]

            if ATOM_FEATURES["xTB-Alpha"][0]:
                atom_feature_vector += [attributes["atomprop"]["alpha"][atom.GetIdx()]]

            if ATOM_FEATURES["xTB-Fukui"][0]:
                atom_feature_vector += [
                    attributes["atomprop"]["fukui"][0][atom.GetIdx()]
                ]
                atom_feature_vector += [
                    attributes["atomprop"]["fukui"][1][atom.GetIdx()]
                ]
                atom_feature_vector += [
                    attributes["atomprop"]["fukui"][2][atom.GetIdx()]
                ]

        return torch.tensor(atom_feature_vector, dtype=torch.float)


class BondFeaturizer(FeaturizerAbstract):
    """Generates features for an RDKit bond."""

    def __init__(self, xtb: bool = True):
        self.xtb = xtb

    def get_bond_features(self, bond, attributes) -> torch.FloatTensor:
        """
        Takes an RDKit bond object and xTB attributes dictionary as input and gives a 1d-tensor of bond features as output.
        """
        bond_feature_vector = []
        if BOND_FEATURES["BondType"][0]:
            bond_feature_vector += self.one_hot_encoding(bond.GetBondType(), BOND_TYPES)

        if BOND_FEATURES["IsConjugated"][0]:
            bond_feature_vector += [int(bond.GetIsConjugated())]

        if BOND_FEATURES["IsInRing"][0]:
            bond_feature_vector += [int(bond.IsInRing())]

        if BOND_FEATURES["Stereochemistry"][0]:
            bond_feature_vector += self.one_hot_encoding(
                str(bond.GetStereo()), STEREOCHEM
            )

        if self.xtb:
            if BOND_FEATURES["xTB-WBO"]:
                bond_feature_vector += [
                    attributes["bondprop"]["wbo"][bond.GetBeginAtomIdx()][
                        bond.GetEndAtomIdx()
                    ]
                ]

        return torch.tensor(bond_feature_vector, dtype=torch.float)


class MolFeaturizer(FeaturizerAbstract):
    """Generates features for an RDKit mol."""

    def __init__(self, xtb: bool = True):
        self.xtb = xtb

    def get_mol_features(self, mol, attributes) -> torch.FloatTensor:
        """
        Returns a tensor with molecular descriptors.
        """
        mol_feature_vector = []
        if MOL_FEATURES["RadiusGyration"][0]:
            mol_feature_vector += [Descriptors3D.RadiusOfGyration(mol)]

        if MOL_FEATURES["Spherocity"][0]:
            mol_feature_vector += [Descriptors3D.SpherocityIndex(mol)]

        if MOL_FEATURES["Asphericity"][0]:
            mol_feature_vector += [Descriptors3D.Asphericity(mol)]

        if MOL_FEATURES["Eccentricity"][0]:
            mol_feature_vector += [Descriptors3D.Eccentricity(mol)]

        if MOL_FEATURES["FractionCSP3"][0]:
            mol_feature_vector += [Lipinski.FractionCSP3(mol)]

        if self.xtb:
            if MOL_FEATURES["xTB-Energy"][0]:
                mol_feature_vector += [attributes["totalenergy"]]

            if MOL_FEATURES["xTB-Charge"][0]:
                mol_feature_vector += [attributes["charge"]]

        return torch.tensor([mol_feature_vector], dtype=torch.float)


class AIMNet:
    def __init__(self, mol: Chem) -> None:
        self.mol = mol
        self.features = self.get_aimnet_features()

    def get_mult(self):
        """Get the multiplicity of the molecule"""
        num_radical_e = 0
        for atom in self.mol.GetAtoms():
            num_radical_e += atom.GetNumRadicalElectrons()
        spin = num_radical_e / 2
        mult = 2 * spin + 1
        return mult

    def get_charge(self):
        """Get the charge of the molecule"""
        return Chem.GetFormalCharge(self.mol)

    def get_atomic_nums(self):
        """Get the atomic numbers of the atoms in the molecule"""
        return [atom.GetAtomicNum() for atom in self.mol.GetAtoms()]

    def get_coords(self):
        """Get the coordinates of the atoms in the molecule"""
        return self.mol.GetConformer().GetPositions()

    def get_models(self, device):
        """Get the atomic numbers and coordinates of the atoms in the molecule"""
        return [
            torch.jit.load(file, map_location=device)
            for file in glob.glob("/ihome/ghutchison/oda6/aimnetnse/models/*.jpt")
        ]

    def get_aimnet_features(self):
        """Get the AIMNet features of the molecule"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.get_models(device)

        atomic_nums = self.get_atomic_nums()
        coords = self.get_coords()
        charge = self.get_charge()
        mult = self.get_mult()

        atomic_nums = torch.tensor(
            atomic_nums, dtype=torch.long, device=device
        ).reshape(1, -1)
        coords = torch.tensor(coords, dtype=torch.float, device=device).reshape(
            1, -1, 3
        )
        charge = torch.tensor(charge, dtype=torch.float, device=device).reshape(1, -1)
        mult = torch.tensor(mult, dtype=torch.float, device=device).reshape(1, -1)

        _in = dict(coord=coords, numbers=atomic_nums, charge=charge, mult=mult)
        _out = [m(_in) for m in self.models]

        return _out

    def get_aimnet_energy(self):
        """Get the AIMNet energy of the molecule"""
        return torch.tensor([x["energy"][1] for x in self.features]).mean()


class Featurizer:
    """
    Initialize the MolFeaturizer.

    Parameters:
    - smiles (Union[str, None]): SMILES string of the molecule.
    - mol (Union[Chem.Mol, None]): RDKit mol object of the molecule.
    - name (str): Name of the molecule.
    - xtb (bool): Whether to use xTB features.
    - aimnet (bool): Whether to use AIMNet features.
    - y (Any): Target variable.
    - idx_to_list (bool): Whether to convert indices to a list.
    - convert_strings (bool): Whether to convert strings to their appropriate types.
    - num_processes (int): Number of processes.
    - **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        smiles: Union[str, None] = None,
        mol: Union[Chem.Mol, None] = None,
        name: str = "molecule",
        xtb: bool = True,
        aimnet: bool = False,
        y: Any = None,
        idx_to_list: bool = False,
        convert_strings: bool = False,
        num_processes: int = 1,
        **kwargs,
    ):
        self.smiles = smiles
        self.mol = mol
        self.name = name

        if self.smiles and self.mol:
            raise ValueError('Either "smiles" or "mol" can be used, not both.')
        if self.smiles:
            self.mol = self.set_mol()

        self.xtb = xtb
        self.aimnet = aimnet
        self.y = y
        self.idx_to_list = idx_to_list
        self.convert_strings = convert_strings
        self.num_processes = num_processes
        self.kwargs = kwargs
        self.exclude_atom = False

        self.atom_featurizer = AtomFeaturizer(xtb)
        self.bond_featurizer = BondFeaturizer()
        self.mol_featurizer = MolFeaturizer(xtb)
        # self.mol_file = self.get_mol_file()
        self.set_feature_lengths()
        self.data = self.set_graph()

    def __call__(self) -> Data:
        """Returns the pyg data object"""
        return self.data

    def convert_strings_func(self, string) -> Any:
        """Converts strings to their appropriate types"""
        return ast.literal_eval(string)

    def set_mol(self) -> Chem.Mol:
        """
        Create a RDKit mol object from smiles and perform initial MMFF94 optimization.
        Returns the mol object.
        """
        mol = Chem.MolFromSmiles(self.smiles)
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
        return mol

    def get_atom_features(self, atom: Chem.Atom, attributes: dict) -> torch.Tensor:
        """Generates features for a given RDKit atom."""
        return self.atom_featurizer.get_atom_features(atom, attributes)

    def get_bond_features(self, bond: Chem.Bond, attributes: dict) -> torch.Tensor:
        """Generates features for a given RDKit bond."""
        return self.bond_featurizer.get_bond_features(bond, attributes)

    def get_mol_features(self, attributes: dict) -> torch.Tensor:
        """Generates molecular descriptors."""
        return self.mol_featurizer.get_mol_features(self.mol, attributes)

    def set_feature_lengths(self) -> None:
        """Sets features lengths."""
        self.n_node_features = 0
        for key, value in ATOM_FEATURES.items():
            if self.xtb is False and "xTB" in key:
                pass
            else:
                if value[0]:
                    if isinstance(value[1], tuple):
                        self.n_node_features += value[1][0]
                    else:
                        self.n_node_features += value[1]
                else:
                    if isinstance(value[1], tuple):
                        self.n_node_features += value[1][1]

        self.n_edge_features = 0
        for key, value in BOND_FEATURES.items():
            if value[0]:
                self.n_edge_features += value[1]

        self.n_global_features = 0
        for key, value in MOL_FEATURES.items():
            if value[0]:
                self.n_global_features += value[1]

    def get_feature_lengths(self) -> Tuple[int, int]:
        """Returns the node and edge feature vector lengths."""
        return self.n_node_features, self.n_edge_features

    def get_energy(self, attributes):
        """Returns the xTB-GFN2 total energy"""
        return torch.tensor(attributes["totalenergy"], dtype=torch.float)

    def get_aimnet_energy(self, mol):
        """Returns the AIMNet energy"""
        aimnet = AIMNet(mol)
        return aimnet.get_aimnet_energy()

    def get_graph(self) -> Data:
        """Returns the pyg Data graph"""
        return self.data

    def get_xtb_attributes(self) -> dict:
        """Compute xTB-GFN2 attributes."""
        if self.xtb:
            try:
                xtb_out = RunXTB(
                    self.mol, f"--opt --alpb water --lmo -P {self.num_processes}"
                )
                xtbp = XTBP(xtb_out())
                mol_attributes = xtbp()

                # self.mol = xtb_out.get_opt_mol()

                fukui_out = RunXTB(self.mol, "--vfukui")
                fukui = XTBP(fukui_out())
                mol_attributes["atomprop"]["fukui"] = fukui["atomprop"]["fukui"]
            except Exception as e:
                self._handle_processing_error(e)
                mol_attributes = {}
        else:
            mol_attributes = {}
        return mol_attributes

    def construct_node_features(
        self, n_nodes: int, mol_attributes: dict
    ) -> torch.Tensor:
        """Construct node feature matrix X."""
        X = torch.zeros((n_nodes, self.n_node_features), dtype=torch.float)
        for atom in list(self.mol.GetAtoms()):
            X[atom.GetIdx()] = self.get_atom_features(atom, mol_attributes)
        return X

    def construct_edge_index(self) -> torch.Tensor:
        """Construct edge index array E."""
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(self.mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        return torch.stack([torch_rows, torch_cols], dim=0)

    def construct_edge_features(
        self, n_edges: int, mol_attributes: dict
    ) -> torch.Tensor:
        """Construct edge feature array EF."""
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(self.mol))
        EF = torch.zeros((n_edges, 1), dtype=torch.float)
        bond00_features = self.get_bond_features(
            self.mol.GetBondBetweenAtoms(int(rows[0]), int(cols[0])), mol_attributes
        )
        self.n_edge_features = len(bond00_features)
        EF = EF.expand((-1, self.n_edge_features)).clone().detach()
        EF[0] = bond00_features
        for k, (i, j) in enumerate(zip(rows[1:], cols[1:]), start=1):
            EF[k] = self.get_bond_features(
                self.mol.GetBondBetweenAtoms(int(i), int(j)), mol_attributes
            )
        return EF

    def process_target_data(self, data: Data) -> Data:
        """Process target data."""
        if self.y is not None:
            data.y = self.y
        return data

    def process_additional_kwargs(self, data: Data, mol_attributes: dict) -> Data:
        """Process additional keyword arguments."""
        if self.convert_strings:
            for key, value in self.kwargs.items():
                self.kwargs[key] = self.convert_strings_func(value)
        if self.idx_to_list and self.y:
            idx_lst = np.zeros(self.mol.GetNumAtoms())
            if self.exclude_atom:
                if isinstance(self.exclude_atom, int):
                    for x in self.y:
                        if (
                            self.mol.GetAtomWithIdx(x).GetAtomicNum()
                            != self.exclude_atom
                        ):
                            idx_lst[x] = 1.0
                elif isinstance(self.exclude_atom, str):
                    for x in self.y:
                        if (
                            self.mol.GetAtomWithIdx(x).GetSymbol()
                            != self.exclude_atom.title()
                        ):
                            idx_lst[x] = 1.0
            else:
                for x in self.y:
                    idx_lst[x] = 1.0
            self.y = idx_lst

        if self.xtb:
            if self.aimnet:
                data.energy = self.get_aimnet_energy(self.mol)
            else:
                data.energy = self.get_energy(mol_attributes)
            data.energy = data.energy.reshape(-1, 1)

        return data

    def set_graph(self) -> Data:
        """Create a PyTorch Geometric Data object."""
        mol_attributes = self.get_xtb_attributes()

        n_nodes = self.mol.GetNumAtoms()
        n_edges = 2 * self.mol.GetNumBonds()
        X = self.construct_node_features(n_nodes, mol_attributes)
        E = self.construct_edge_index()
        EF = self.construct_edge_features(n_edges, mol_attributes)
        MF = self.get_mol_features(mol_attributes)

        data = Data(
            x=X,
            edge_index=E,
            edge_attr=EF,
            global_attr=MF,
            name=self.name,
            smiles=self.smiles,
            mol=self.mol,
        )

        data = self.process_target_data(data)
        data = self.process_additional_kwargs(data, mol_attributes)

        return data

    def _handle_processing_error(self, error) -> None:
        """Handle errors during processing"""
        logger.error(f"Error processing {self.name}")
        logger.error(f"Error: {error}")
        logger.error(traceback.format_exc())

    # def set_graph(self) -> Data:
    #     """Create a Pytorch Geometric Data object from smiles"""
    #     # get xTB-GFN2 attributes
    #     if self.xtb:
    #         xtb_start_time = timeit.default_timer()
    #         xtb_out = RunXTB(
    #             self.mol, f"--opt --alpb water --lmo -P {self.num_processes}"
    #         )
    #         xtbp = XTBP(xtb_out())
    #         mol_attributes = xtbp()
    #         opt_mol = xtb_out.get_opt_mol()

    #         fukui_out = RunXTB(self.mol, "--vfukui")
    #         fukui = XTBP(fukui_out())
    #         mol_attributes["atomprop"]["fukui"] = fukui["atomprop"]["fukui"]
    #         xtb_stop_time = timeit.default_timer()
    #         with open(
    #             f"timing/{self.num_processes}/xtb_feature_time.txt",
    #             "at",
    #             encoding="utf-8",
    #         ) as f:
    #             f.write(f"{self.name},{xtb_stop_time - xtb_start_time}\n")
    #     else:
    #         mol_attributes = {}

    #     n_nodes = self.mol.GetNumAtoms()
    #     n_edges = 2 * self.mol.GetNumBonds()

    #     # construct node feature matrix X of shape (n_nodes, n_node_features)
    #     X = torch.zeros((n_nodes, self.n_node_features), dtype=torch.float)
    #     for atom in list(self.mol.GetAtoms()):
    #         X[atom.GetIdx()] = self.get_atom_features(atom, mol_attributes)

    #     # construct edge index array E of shape (2, n_edges)
    #     (rows, cols) = np.nonzero(GetAdjacencyMatrix(self.mol))
    #     torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    #     torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    #     E = torch.stack([torch_rows, torch_cols], dim=0)

    #     # construct edge feature array EF of shape (n_edges, n_edge_features)
    #     EF = torch.zeros((n_edges, 1), dtype=torch.float)
    #     bond00_features = self.get_bond_features(
    #         self.mol.GetBondBetweenAtoms(int(rows[0]), int(cols[0])), mol_attributes
    #     )
    #     self.n_edge_features = len(bond00_features)
    #     EF = EF.expand((-1, self.n_edge_features)).clone().detach()
    #     EF[0] = bond00_features
    #     for k, (i, j) in enumerate(zip(rows[1:], cols[1:]), start=1):
    #         EF[k] = self.get_bond_features(
    #             self.mol.GetBondBetweenAtoms(int(i), int(j)), mol_attributes
    #         )

    #     # construct global molecular features array
    #     MF = self.get_mol_features(mol_attributes)

    #     # create data structure
    #     data = Data(
    #         x=X,
    #         edge_index=E,
    #         edge_attr=EF,
    #         global_attr=MF,
    #         name=self.name,
    #         smiles=self.smiles,
    #         mol=self.mol,
    #     )

    #     if not isinstance(self.y, (int, float, list, tuple, type(None))):
    #         self.y = self.convert_strings_func(str(self.y))

    #     if self.convert_strings:
    #         for key, value in self.kwargs.items():
    #             self.kwargs[key] = self.convert_strings_func(value)

    #     if self.idx_to_list and self.y:
    #         idx_lst = np.zeros(n_nodes)
    #         if self.exclude_atom:
    #             if isinstance(self.exclude_atom, int):
    #                 for x in self.y:
    #                     if (
    #                         self.mol.GetAtomWithIdx(x).GetAtomicNum()
    #                         != self.exclude_atom
    #                     ):
    #                         idx_lst[x] = 1.0

    #             elif isinstance(self.exclude_atom, str):
    #                 for x in self.y:
    #                     if (
    #                         self.mol.GetAtomWithIdx(x).GetSymbol()
    #                         != self.exclude_atom.title()
    #                     ):
    #                         idx_lst[x] = 1.0
    #         else:
    #             for x in self.y:
    #                 idx_lst[x] = 1.0

    #         self.y = idx_lst
    #     if self.xtb:
    #         if self.aimnet:
    #             data.energy = self.get_aimnet_energy(opt_mol)
    #         else:
    #             data.energy = self.get_energy(mol_attributes)

    #         data.energy = data.energy.reshape(-1, 1)

    #     if self.y is not None:
    #         # self.y = self.convert_strings_func(self.y)
    #         data.y = torch.tensor(self.y).reshape(-1, 1)

    #     for key, value in self.kwargs.items():
    #         data[key] = value

    #     return data
