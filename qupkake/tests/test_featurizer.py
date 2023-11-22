import pytest

from qupkake.featurizer import Featurizer


@pytest.fixture
def example_featurizer():
    smiles = "CCO"
    return Featurizer(smiles=smiles, name="example_molecule", xtb=True)


def test_featurizer(example_featurizer):
    # Call the featurizer to get the PyTorch Geometric Data object
    data = example_featurizer()

    # Basic assertions to check the correctness of the featurization
    assert data.num_nodes == 9
    assert data.num_edges == 16
    assert data.x.shape[1] == example_featurizer.n_node_features
    assert data.edge_attr.shape[1] == example_featurizer.n_edge_features
    assert data.global_attr.shape[1] == example_featurizer.n_global_features


if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main([__file__])
