import os

import pytest
from qupkake.mol_utils import Tautomerize
from rdkit import Chem


@pytest.fixture
def example_tautomer(request):
    smiles = "Oc1c(cccc3)c3nc2ccncc12"
    tautomer = Tautomerize(smiles=smiles, name="pytest_mol")

    # Delete the mol_t0.mol file before the test
    tautomer.mol_dir = "molecules"
    mol_file_path = os.path.join(tautomer.mol_dir, "pytest_mol_t0.mol")
    if os.path.exists(mol_file_path):
        os.remove(mol_file_path)

    # Add a finalizer to ensure the mol_t0.mol file is deleted after the test
    def delete_mol_file():
        if os.path.exists(mol_file_path):
            os.remove(mol_file_path)

    request.addfinalizer(delete_mol_file)

    return tautomer


def test_init(example_tautomer):
    assert example_tautomer.smiles is not None
    assert example_tautomer.mol is not None
    assert example_tautomer.name == "pytest_mol"
    assert example_tautomer.mol_dir == "molecules"
    assert example_tautomer.run is True
    assert example_tautomer.keep_mol is True
    assert example_tautomer.check_exists is True
    assert example_tautomer.num_processes == 1
    assert example_tautomer.lowest_tautomer_path == "molecules/pytest_mol_t2.mol"
    assert example_tautomer.lowest_tautomer_num == 0
    assert example_tautomer.lowest_tautomer_energy is not None
    assert example_tautomer.lowest_tautomer is not None
    assert example_tautomer.lowest_tautomer_name == "pytest_mol_t0"


def test_check_exist_mol(example_tautomer, tmp_path):
    example_tautomer.mol_dir = tmp_path
    assert not example_tautomer.check_exist_mol()


def test_get_smiles(example_tautomer):
    assert example_tautomer.get_smiles() == "Oc1c(cccc3)c3nc2ccncc12"


def test_set_mol(example_tautomer):
    mol = example_tautomer.set_mol()
    assert isinstance(mol, Chem.Mol)


if __name__ == "__main__":
    pytest.main()
