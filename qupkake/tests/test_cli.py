import os
import tempfile

import pytest
from qupkake.cli import (
    check_output_file,
    embed_molecule,
    main_file,
    main_smiles,
)
from rdkit import Chem


@pytest.fixture
def mock_args():
    class MockArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return MockArgs


def create_temp_file(suffix, content):
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def test_embed_molecule():
    # Test benzene
    benzene_smiles = "c1ccccc1"
    benzene_molecule = Chem.MolFromSmiles(benzene_smiles)
    embedded_benzene = embed_molecule(benzene_molecule)
    assert embedded_benzene.GetNumConformers() > 0

    # Test acetic acid
    acetic_acid_smiles = "CC(=O)O"
    acetic_acid_molecule = Chem.MolFromSmiles(acetic_acid_smiles)
    embedded_acetic_acid = embed_molecule(acetic_acid_molecule)
    assert embedded_acetic_acid.GetNumConformers() > 0

    # Test pyridine
    pyridine_smiles = "c1ccncc1"
    pyridine_molecule = Chem.MolFromSmiles(pyridine_smiles)
    embedded_pyridine = embed_molecule(pyridine_molecule)
    assert embedded_pyridine.GetNumConformers() > 0


def test_main_file_csv(mock_args):
    # Create a temporary CSV file with benzene, acetic acid, and pyridine
    temp_output_dir = tempfile.mkdtemp()
    csv_data = (
        "smiles,name\n" "c1ccccc1,benzene\n" "CC(=O)O,acetic_acid\n" "c1ccncc1,pyridine"
    )
    temp_csv_path = create_temp_file(".csv", csv_data)

    # Mock the embed_molecule function
    args = mock_args(
        filename=temp_csv_path,
        smiles_col="smiles",
        name_col="name",
        root=temp_output_dir,
        output="output.sdf",
        multiprocessing=False,
        tautomerize=True,
    )
    main_file(args)

    # Assert that the CSV file was read and processed correctly
    assert os.path.exists(os.path.join(args.root, "output", "output.sdf"))


def test_check_output_file(mock_args):
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_output_dir:
        args = mock_args(filename="output.sdf", root=temp_output_dir)
        result = check_output_file(args.filename, args.root)

    # Assert that the output file path is correct and does not exist
    assert result == "output.sdf"
    assert not os.path.exists(os.path.join(args.root, "output", "output.sdf"))


def test_main_smiles(mock_args):
    # Test benzene
    benzene_smiles = "c1ccccc1"
    temp_output_dir_benzene = tempfile.mkdtemp()
    args_benzene = mock_args(
        smiles=benzene_smiles,
        name="benzene",
        root=temp_output_dir_benzene,
        multiprocessing=False,
        tautomerize=True,
        output="benzene.sdf",
    )
    main_smiles(args_benzene)
    assert not os.path.exists(
        os.path.join(temp_output_dir_benzene, "output", "benzene.sdf")
    )

    # Test acetic acid
    acetic_acid_smiles = "CC(=O)O"
    temp_output_dir_acetic_acid = tempfile.mkdtemp()
    args_acetic_acid = mock_args(
        smiles=acetic_acid_smiles,
        name="acetic_acid",
        root=temp_output_dir_acetic_acid,
        multiprocessing=False,
        tautomerize=True,
        output="acetic_acid.sdf",
    )
    main_smiles(args_acetic_acid)
    assert os.path.exists(
        os.path.join(temp_output_dir_acetic_acid, "output", "acetic_acid.sdf")
    )
    # Check if the output file has 2 molecules
    with Chem.SDMolSupplier(
        os.path.join(temp_output_dir_acetic_acid, "output", "acetic_acid.sdf")
    ) as supplier:
        assert len(list(supplier)) == 2

    # Test pyridine
    pyridine_smiles = "c1ccncc1"
    temp_output_dir_pyridine = tempfile.mkdtemp()
    args_pyridine = mock_args(
        smiles=pyridine_smiles,
        name="pyridine",
        root=temp_output_dir_pyridine,
        multiprocessing=False,
        tautomerize=True,
        output="pyridine.sdf",
    )
    main_smiles(args_pyridine)
    assert os.path.exists(
        os.path.join(temp_output_dir_pyridine, "output", "pyridine.sdf")
    )
    # Check if the output file has 1 molecule
    with Chem.SDMolSupplier(
        os.path.join(temp_output_dir_pyridine, "output", "pyridine.sdf")
    ) as supplier:
        assert len(list(supplier)) == 1
