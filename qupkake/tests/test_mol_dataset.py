import os
import warnings

import pandas as pd
import pytest

from qupkake.mol_dataset import MolDataset, MolPairDataset


@pytest.fixture
def test_data_folder(tmpdir_factory):
    # Create a temporary directory for test data
    tmpdir = tmpdir_factory.mktemp("test_data")
    return tmpdir


@pytest.fixture
def mol_dataset(test_data_folder):
    # Setup MolDataset instance for testing
    # Create a "raw" folder under the temporary directory
    raw_folder = test_data_folder.mkdir("raw")
    # Create a test CSV file
    csv_path = os.path.join(raw_folder, "test_mol_dataset.csv")
    create_test_csv(
        csv_path,
        columns=["name", "smiles"],
        data=[
            {"name": "AceticAcid", "smiles": "CC(=O)O"},
            {"name": "Pyridine", "smiles": "C1=CC=NC=C1"},
        ],
    )
    return MolDataset(
        root=str(test_data_folder), filename="test_mol_dataset.csv", tautomerize=True
    )


@pytest.fixture
def mol_pair_dataset(test_data_folder):
    # Setup MolPairDataset instance for testing
    # Create a "raw" folder under the temporary directory
    raw_folder = test_data_folder.mkdir("raw")
    # Create a test CSV file with three columns: name, smiles, idx, pka_type
    csv_path = os.path.join(raw_folder, "test_mol_pair_dataset.csv")
    create_test_csv(
        csv_path,
        columns=["name", "smiles", "idx", "pka_type"],
        data=[
            {"name": "AceticAcid", "smiles": "CC(=O)O", "idx": 3, "pka_type": "basic"},
            {
                "name": "Pyridine",
                "smiles": "C1=CC=NC=C1",
                "idx": 3,
                "pka_type": "acidic",
            },
        ],
    )
    return MolPairDataset(
        root=str(test_data_folder), filename="test_mol_pair_dataset.csv"
    )


def create_test_csv(file_path, columns, data):
    # Create a test CSV file
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

def test_mol_dataset_creation(mol_dataset):
    assert mol_dataset is not None
    # Add more assertions as needed


def test_mol_pair_dataset_creation(mol_pair_dataset):
    assert mol_pair_dataset is not None
    # Add more assertions as needed
