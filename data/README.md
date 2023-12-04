# Datasets

This folder contains the datasets used to train and test QupKake.


1. `chembl_crest_combined_set.csv.gz` - ChEMBL datases with CREST protonations\deprotonation site indices. The file contains the ChEMBL identifier, the original SMILES string, the SMILES string of the most stable tautomer, the ChemAxon-predicted most acidic and basic pKas, the protonation and deprotonation site index found by CREST. This dataset was used to train the initial pKa prediction model.
2. `exp_training_data.sdf` - The experimental training data used in the transfer learning. [^1]
3. `novartis_qupkake_pka.sdf` - The Novartis dataset used to test QupKake. The file contains the predicted pKa and the reaction site index.[^1]
4. `literature_qupkake_pka.sdf` - The literature dataset used to test QupKake. The file contains the predicted pKa and the reaction site index.[^1]

[^1]: The data file were extracted from [Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa) and were modified to be compatible with QupKake.