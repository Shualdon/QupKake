import glob
import io
import os
import pathlib
import re
import shutil
import tempfile
import traceback
from copy import deepcopy
from itertools import groupby
from subprocess import DEVNULL, PIPE, run
from typing import Any, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .xtbp import XTBP, RunXTB


class Tautomerize:
    """
    Class for finding the lowest energy tautomer of a molecule using xtb calculations.

    Args:
        smiles (str): The SMILES string of the molecule. Cannot be used together with 'mol'.
        mol (rdkit.Chem.Mol): The mol object. Cannot be used together with 'smiles'.
        name (str): Molecule name.
        mol_dir (str): Name of directory to save the mol files.
        run (bool): Whether to run tautomerization if 'check_exists' is False or if the tautomer file is not found.
        keep_mol (bool): Whether to keep the generated mol files.
        check_exists (bool): Whether to check if the tautomer file already exists in 'mol_dir'.
        num_processes (int): Number of processes to use in xtb calculations.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        # self,
        # smiles: str = None,
        # mol: Chem.Mol = None,
        # name: str = "mol",
        # mol_dir: str = "molecules",
        # run: bool = True,
        # keep_mol: bool = True,
        # check_exists: bool = True,
        # num_processes: int = 1,
        # **kwargs,
        self,
        smiles: Optional[str] = None,
        mol: Optional[Chem.Mol] = None,
        name: str = "mol",
        mol_dir: str = "molecules",
        run: bool = True,
        keep_mol: bool = True,
        check_exists: bool = True,
        num_processes: int = 1,
        **kwargs,
    ):
        if smiles and mol:
            raise ValueError('Either "smiles" or "mol" can be used, not both.')

        self.smiles = smiles
        self.mol = mol or self.set_mol()
        self.name = name
        self.mol_dir = mol_dir
        self.keep_mol = keep_mol
        self.lowest_tautomer_path: Optional[str] = None
        self.kwargs = kwargs
        self.run = run
        self.check_exists = check_exists
        self.num_processes = num_processes
        self.tautomers = self.set_tautomers()

        if self.check_exists:
            exists = self.check_exist_mol()
            if not exists and self.run:
                self.make_tautomer_files()
        elif self.run:
            self.make_tautomer_files()

    def check_exist_mol(self) -> bool:
        """Returns true if tautomer file exists.
        Sets the tautomer and tautomer path variables."""
        mol_files = glob.glob(f"{self.mol_dir}/{self.name}_t*.mol")
        if mol_files:
            taut_num = mol_files[0].split("_t")[-1].split(".mol")[0]
            self.lowest_tautomer_num = int(taut_num)
            taut = Chem.MolFromMolFile(mol_files[0], sanitize=False)
            taut.UpdatePropertyCache(strict=False)
            self.lowest_tautomer = taut
            self.lowest_tautomer_path = mol_files[0]
            self.lowest_tautomer_name = pathlib.Path(self.lowest_tautomer_path).stem
            return True
        return False

    def get_smiles(self) -> Optional[str]:
        """Get the smiles"""
        return self.smiles

    def set_mol(self) -> Chem.Mol:
        """Create a RDKit mol object from smiles and perform initial MMFF94 optimization"""
        mol = Chem.MolFromSmiles(self.smiles)
        return self.opt_mol(mol)

    def opt_mol(self, mol: Chem.Mol) -> Chem.Mol:
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
        mol = Chem.RemoveHs(mol)
        return mol

    def get_mol(self) -> Chem.Mol:
        """Get RDKit mol object of the original smiles"""
        return self.mol

    def set_tautomers(self) -> List[Chem.Mol]:
        """Create list of possible tautomers"""
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.Canonicalize(self.mol)
        tauts = enumerator.Enumerate(self.mol)
        return tauts

    def get_tautomers(self, smiles: bool = False) -> List[Union[str, Chem.Mol]]:
        """return a list of tautomers
        if smiles is True returns a list of smiles, otherwise returns list of RDKit mol objects
        """
        if smiles:
            return [Chem.MolToSmiles(x) for x in self.tautomers]
        else:
            return list(self.tautomers)

    def get_gfn2_energy(self, file) -> float:
        """Returns the total energy from a GFN2-xTB output"""
        energy = 0.0
        for line in file:
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[3])
        return energy

    def make_tautomer_files(self) -> None:
        """Create tautomer files, run GNF2-xTB, and parse the total energy"""
        #os.makedirs(self.mol_dir, exist_ok=True)
        tautomer_energy = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i, taut in enumerate(self.tautomers):
                taut_file = f"{tmpdirname}/{self.name}_t{i}.mol"
                Chem.MolToMolFile(taut, taut_file)
                xtb_out = RunXTB(
                    taut_file, f"--opt --alpb water --lmo -P {self.num_processes}"
                )
                xtbp = XTBP(xtb_out())
                mol_attributes = xtbp()
                energy = mol_attributes["totalenergy"]
                tautomer_energy.append((i, energy))

            self.sorted_energies = sorted(tautomer_energy, key=lambda x: x[1])
            self.lowest_tautomer_num = self.sorted_energies[0][0]
            self.lowest_tautomer_energy = self.sorted_energies[0][1]
            self.lowest_tautomer = self.tautomers[self.lowest_tautomer_num]
            self.lowest_tautomer_name = f"{self.name}_t{self.lowest_tautomer_num}"

            if self.keep_mol:
                temp_tautomer_path = (
                    f"{tmpdirname}/{self.name}_t{self.lowest_tautomer_num}.mol"
                )
                shutil.copy(temp_tautomer_path, self.mol_dir)
                self.lowest_tautomer_path = (
                    f"{self.mol_dir}/{self.name}_t{self.lowest_tautomer_num}.mol"
                )

    def get_tautomer_file(self) -> str:
        """Returns the path of the most stable tautomer file."""
        if self.keep_mol:
            return os.path.join(
                self.mol_dir, f"{self.name}_t{self.lowest_tautomer_num}.mol"
            )
        else:
            raise Exception(
                "You chose not to keep the mol file. Set 'keep_mol' to True to save the file."
            )
            # print("You chose not to keep the mol file.\nSet 'keep_mol' to True to save the files.")

    def get_lowest_tautomer(self) -> Optional[Chem.Mol]:
        """Returns the mol object of the most stable tautomer."""
        if self.lowest_tautomer:
            return self.lowest_tautomer
        return None

    def get_lowest_tautomer_num(self) -> Optional[int]:
        """Returns the index of the most stable tautomer."""
        if self.lowest_tautomer_num:
            return self.lowest_tautomer_num
        return None

    def get_lowest_tautomer_energy(self) -> Optional[float]:
        """Returns the energyt of the most stable tautomer."""
        if self.lowest_tautomer_energy:
            return self.lowest_tautomer_energy
        return None

    def get_lowest_tautomer_smiles(self) -> Optional[str]:
        """Returns the smiles of the most stable tautomer."""
        if self.lowest_tautomer:
            self.lowest_tautomer_smiles = Chem.MolToSmiles(self.lowest_tautomer)
            return self.lowest_tautomer_smiles
        return None

    def __call__(self) -> "Tautomerize":
        return self


class Conjugate:
    def __init__(self, mol: Chem, idx: int, h_num: int):
        self.mol = mol
        self.idx = idx
        self.h_num = h_num
        self.conjugate = None
        try:
            self.check_input()
            self.conjugate = self.set_conjugate()
        except Exception as e:
            with open("traceback.txt", "a") as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            # print(e)

    def __call__(self) -> Chem:
        """Returns the conjugated mol"""
        if self.conjugate:
            return self.conjugate
        else:
            raise Exception("Conjugate wasn't created")

    def check_input(self) -> bool:
        """Checkes that the mol can be protonated \ deprotonated at the wanted index"""
        # self.mol = Chem.AddHs(self.mol, addCoords=True)
        atom_nums = len(self.mol.GetAtoms())
        if self.idx > atom_nums:
            raise ValueError("ERROR: Atom index not in mol. Conjugate not created.")

        atom = self.mol.GetAtomWithIdx(self.idx)
        tot_h = atom.GetTotalNumHs(includeNeighbors=True)
        change_h = tot_h + self.h_num
        if change_h < 0:
            # with open("bad_mols.txt", "a") as f:
            #     f.write(Chem.MolToSmiles(self.mol), self.idx, self.h_num)
            #     f.write("\n")
            # pass
            raise ValueError(
                "ERROR: Cannot remove non-existing protons. Conjugate not created."
            )

        return True

    def set_conjugate(self) -> Chem:
        """Creates the conjugate"""
        mol_copy = deepcopy(self.mol)
        rw_conjugate = Chem.RWMol(mol_copy)

        atom = mol_copy.GetAtomWithIdx(self.idx)
        rw_atom = rw_conjugate.GetAtomWithIdx(self.idx)
        charge = atom.GetFormalCharge()
        rw_atom.SetFormalCharge(charge + self.h_num)

        exp_hs = atom.GetNumExplicitHs()

        if exp_hs > 0 and self.h_num < 0:
            rw_atom.SetNumExplicitHs(exp_hs + self.h_num)

        if self.h_num > 0:
            rw_atom.SetNumExplicitHs(exp_hs + self.h_num)

        rw_atom.UpdatePropertyCache()
        rw_conjugate.UpdatePropertyCache()

        return rw_conjugate.GetMol()

    def set_conjugate2(self) -> Chem:
        """Creates the conjugate"""
        mol_copy = deepcopy(self.mol)
        # mol_copy = Chem.AddHs(mol_copy, addCoords=True)
        rw_conjugate = mol_copy  # Chem.RWMol(mol_copy)

        Chem.SanitizeMol(rw_conjugate)
        rw_conjugate = Chem.RemoveHs(rw_conjugate, updateExplicitCount=True)
        # atom = rw_conjugate.GetAtomWithIdx(self.idx)
        rw_atom = rw_conjugate.GetAtomWithIdx(self.idx)
        charge = rw_atom.GetFormalCharge()

        exp_hs = rw_atom.GetNumExplicitHs()
        tot_hs = rw_atom.GetTotalNumHs(includeNeighbors=True)
        tot_valance = rw_atom.GetTotalValence()
        exp_valance = rw_atom.GetExplicitValence()
        imp_valance = rw_atom.GetImplicitValence()
        ps = Chem.RemoveHsParameters()
        # if exp_hs > 0 and self.h_num < 0:
        #     rw_atom.SetNumExplicitHs(exp_hs + self.h_num)

        # if self.h_num > 0:
        #     rw_atom.SetNumExplicitHs(exp_hs + self.h_num)
        if self.h_num < 0:
            rw_atom.SetFormalCharge(charge - 1)
            if exp_hs > 0:
                rw_atom.SetNumExplicitHs(exp_hs - 1)

        elif self.h_num > 0:
            rw_atom.SetFormalCharge(charge + 1)
            # if tot_hs == 0 or exp_hs > 0:
            rw_atom.SetNumExplicitHs(tot_hs + 1)
            # rw_atom.SetNumExplicitHs(exp_hs + self.h_num)

            # raise RuntimeError(
            #     f"Atom: {self.idx}, Symbol {rw_atom.GetSymbol()}, Charge Before {charge} & After{rw_atom.GetFormalCharge()}, Valance {rw_atom.GetTotalValence()}"
            # )
            # rw_conjugate = Chem.RemoveHs(rw_conjugate, updateExplicitCount=True)
            # try:

            #     # rw_conjugate = Chem.RemoveHs(rw_conjugate, updateExplicitCount=True)
        rw_atom.UpdatePropertyCache(strict=True)
        rw_conjugate.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(rw_conjugate)
        #     # except:
        #     # rw_conjugate = rdMolStandardize.Cleanup(rw_conjugate)
        #     # return rw_conjugate.GetMol()
        # except Exception as e:
        #     with open("traceback.txt", "a") as f:
        #         f.write(str(e))
        #         f.write(f"\nAtom: {self.idx}, Symbol {rw_atom.GetSymbol()}\n")
        #         f.write(f"Charge Before {charge} & After {rw_atom.GetFormalCharge()}\n")
        #         f.write(
        #             f"Explicit Hs Before {exp_hs} & After {rw_atom.GetNumExplicitHs()}\n"
        #         )
        #         f.write(
        #             f"Total Hs Before {tot_hs} & After {rw_atom.GetTotalNumHs(includeNeighbors=True)}\n"
        #         )
        #         f.write(
        #             f"Explicit Valance Before {exp_valance} & After {rw_atom.GetExplicitValence()}\n"
        # )
        # f.write(
        #     f"Implicit Valance Before {imp_valance} & After {rw_atom.GetImplicitValence()}\n"
        # )
        # f.write(
        #     f"Total Valance Before {tot_valance} & After {rw_atom.GetTotalValence()}\n"
        # )
        # f.write(traceback.format_exc())
        return rw_conjugate  # .GetMol()

    def get_conjugate(self) -> Chem:
        """Returns the conjugated mol"""
        if self.conjugate:
            return self.conjugate
        else:
            raise Exception("Conjugate wasn't created")
