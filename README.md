QupKake - Predict micro-pKa of organic molecules
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/shualdon/qupkake/workflows/CI/badge.svg)](https://github.com/shualdon/qupkake/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/QupKake/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/QupKake/branch/main)


<p align="center">
    <img src="QupKake_white.png" alt="QupKake" width="400"/>
</p>
<br>

QupKake combines [GFN2-xTB](https://github.com/grimme-lab/xtb) calculations with graph-neural-networks to accuretly predict micro-pKa values of organic molecules. It is part of the paper: [QupKake: Integrating Machine Learning and Quantum
Chemistry for micro-pKa Predictions]().

# Requirements
* Python >= 3.9
* pytorch >= 2.0
* pytorch_geometric >= 2.3.0
* pytorch_lightning >= 2.0.2
* rdkit >= 2022.03.03
* xTB == 2.4.1

# Installation

We recommend using [conda](https://docs.conda.io/en/latest/) to install QupKake.

## Option 1

Clone the repository:
```bash
git clone https://github.com/Shualdon/QupKake.git
cd qupkake
```

Create a conda environment from the environment.yml file:
```bash
conda env create -f environment.yml
conda activate qupkake
```
This will create a conda environment with all the dependencies installed.

Install the package:
```bash
pip install .
```

## Option 2

Create a conda environment:
```bash
conda create -n qupkake python=3.9
conda activate qupkake
```

`xtb` needs to be installed using conda:
```bash
conda install -c conda-forge xtb==6.4.1
```
See [xtb Installation](#xtb-installation) below for other installation options.

Clone the repository and install using pip:
```bash
git clone https://github.com/Shualdon/QupKake.git
cd qupkake
pip install .
```
This will install the package and all the rest of the dependencies.

## `xtb` Installation

Optionally, `xtb` can be installed from [source](https://github.com/grimme-lab/xtb/releases/tag/v6.4.1), and the path to the executable can be set up:
```bash
export XTBPATH=/path/to/xtb/executable
```
This be used even if `xtb` is installed using conda.


The linux binaries of `xtb` come with the package and will be used by default if the neither the conda package or the `$XTB_PATH` enviroment variable are set up.

# Usage
Qupkake can be used as a python package or as a command line tool. This gives the user the flexibility to use the package in their own code or to use it as a stand-alone tool. 

## Command line

Once installed, QupKake can be used as a command line tool. The general syntext for running the program is:
```bash
$ qupkake <input_type> <input> <flags>
```
The general flags that can be used are:
```bash
-r, --root: Root directory for processing data. Default: data

-t, --tautomerize: Find the most stable tautomer for the molecule(s) Default: False

-mp [N], --multiprocessing [N]: Use Multiprocessing. True if used alone. If followed by a number, its will use that number of suprocesses. (default: False)
```

Qupkake has 2 types of inputs that can be used to run the program:
### 1. A single molecule as a SMILES string:
```bash
$ qupkake smiles "SMILES"
```
Specific flags for this input type are:
```bash
-n, --name: molecule name. Default: molecule

-o, --output: output file name (SDF with pKa predictions). Default: qupkake_output.sdf
```

### 2. A CSV or SDF file containing multiple molecules
```bash
$ qupkake file <filename>
```

Specific flags for this input type are:
```bash
-s, --smiles_col: column name for SMILES strings. Default: smiles

-n, --name_col: column name for molecule names. Default: name

-o, --output: output file name (SDF with pKa predictions). Default: qupkake_output.sdf
```

## Python package
 TBA

# Citation
If to use this package in your research, please cite the following paper:

### Bibtex
```bibtex
@article{
    ...
}
```

### Copyright

Copyright (c) 2023, Omri D Abarbanel, Hutchison Group, University of Pittsburgh, PA, USA.


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
