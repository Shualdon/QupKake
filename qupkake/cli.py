import argparse
import sys

from ._version import get_versions

__version__ = get_versions()["version"]


class MyParser(argparse.ArgumentParser):
    """
    Overriden to show help on default.
    """

    def error(self, message):
        print(f"error: {message}")
        self.print_help()
        sys.exit(2)


def embed_molecule(mol):
    """
    Embeds a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Embedded molecule.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol.GetNumConformers() > 0:
        if mol.GetConformer().Is3D():
            return mol

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    return mol


def process_file(filename: str, smiles_col: str, name_col: str, root: str) -> str:
    """
    Processes a CSV or SDF file.

    Parameters
    ----------
    filename : str
        Input file (CSV or SDF).
    smiles_col : str
        Column name with SMILES strings.
    name_col : str
        Column name with molecule names.
    root : str
        Root directory for processing data.

    Returns
    -------
    str
        Path to the SDF file with the embedded molecules.
    """
    import os
    import warnings
    from pathlib import Path

    import pandas as pd
    from rdkit.Chem import PandasTools

    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        if smiles_col not in df.columns:
            raise ValueError("Invalid CSV file. No SMILES column found.")

        PandasTools.AddMoleculeColumnToFrame(df, smiles_col)
        df["ROMol"] = df["ROMol"].apply(lambda x: embed_molecule(x))

    elif filename.endswith(".sdf"):
        df = PandasTools.LoadSDF(
            filename,
            idName=name_col,
            includeFingerprints=False,
            removeHs=False,
            embedProps=True,
        )
    else:
        raise ValueError("Invalid file type.")

    if name_col not in df.columns:
        df[name_col] = f"molecule_{df.index + 1}"
    filename = Path(filename).stem
    file_path = f"{root}/raw/{filename}.sdf"
    if os.path.exists(file_path):
        i = 1
        file_path = f"{root}/raw/{filename}_{i}.sdf"
        while os.path.exists(file_path):
            i += 1
            file_path = f"{root}/raw/{filename}_{i}.sdf"
        warnings.warn(
            f"File {root}/raw/{filename}.sdf already exists.\n"
            "A new file will be created with a different name.\n"
            "Please delete the existing file if you want to overwrite it.",
        )
        filename = filename + f"_{i}"

    PandasTools.WriteSDF(
        df,
        file_path,
        idName=name_col,
        properties=list(df.columns),
    )

    return filename + ".sdf"


def check_output_file(filename: str, root: str) -> str:
    """
    Checks if the output file exists.

    Parameters
    ----------
    filename : str
        Output file.
    root : str
        Root directory for processing data.

    Returns
    -------
    str
        Path to the output file.
    """
    import os
    import warnings
    from pathlib import Path

    filename = Path(filename).stem
    file_path = f"{root}/output/{filename}.sdf"
    if os.path.exists(file_path):
        i = 1
        file_path = f"{root}/output/{filename}_{i}.sdf"
        while os.path.exists(file_path):
            i += 1
            file_path = f"{root}/output/{filename}_{i}.sdf"
        warnings.warn(
            f"File {root}/output/{filename}.sdf already exists.\n"
            "A new file will be created with a different name.\n"
            "Please delete the existing file if you want to overwrite it.",
        )
        filename = filename + f"_{i}"

    return filename + ".sdf"


def main_file(args):
    """
    Predicts the pKa values for a list of molecules in a CSV or SDF file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """

    create_dirs(args)
    args.filename = process_file(
        args.filename, args.smiles_col, args.name_col, args.root
    )
    args.mol_col = "ROMol"
    run_pipeline(args)


def main_smiles(args):
    """
    Predicts the pKa values for a SMILES string.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """

    create_dirs(args)
    smiles_to_sdf(args.smiles, args.name, args.root)
    args.filename = args.name + ".sdf"
    args.mol_col = "ROMol"
    args.name_col = "name"
    run_pipeline(args)


def smiles_to_sdf(smiles: str, name: str, root: str) -> None:
    """
    Converts a SMILES string to a SDF file.

    Parameters
    ----------
    smiles : str
        SMILES string.
    name : str
        Name of the molecule.
    root : str
        Root directory for processing data.

    Returns
    -------
    None
    """
    from rdkit import Chem
    from rdkit.Chem import SDWriter

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    mol = embed_molecule(mol)

    mol.SetProp("_Name", name)
    writer = SDWriter(f"{root}/raw/{name}.sdf")
    writer.write(mol)
    writer.close()


def create_dirs(args):
    """
    Creates the directory structure for the project.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    import os

    root = args.root
    dirs = [
        os.path.join(root, "raw"),
        os.path.join(root, "processed"),
        #os.path.join(root, "molecules"),
        os.path.join(root, "output"),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # if process is not empty, delete all files in processed
    if os.listdir(dirs[1]):
        for f in os.listdir(dirs[1]):
            os.remove(os.path.join(dirs[1], f))

def run_pipeline(args):
    """
    Runs the pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    from .predict import run_prediction_pipeline

    args.output = check_output_file(args.output, args.root)
    args.mp = args.multiprocessing
    run_prediction_pipeline(**vars(args))


def parse_arguments(args):
    """
    Parses the command line arguments.
    """
    gen_parser = MyParser(
        add_help=False,
    )

    gen_parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="data",
        help="Root directory for processing data.",
    )

    gen_parser.add_argument(
        "-t",
        "--tautomerize",
        action="store_true",
        help="Find the most stable tautomer for the molecule(s).",
    )

    gen_parser.add_argument(
        "-mp",
        "--multiprocessing",
        nargs="?",
        const=True,
        type=int,
        default=False,
        metavar="N",
        help="Use Multiprocessing. True if used alone. If followed by a number, its will use that number of suprocesses.",
    )
    gen_parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )

    parser = MyParser(
        prog="qupkake",
        description="QupKake: Quantum pKa graph-neural-network Estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[gen_parser],
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="QupKake input options.",
        help="Use subcommand --help for more information",
        dest="subcommand",
        required=True,
    )

    parser_file = subparsers.add_parser(
        "file",
        help="Predict pKa values for molecules in a CSV or SDF file",
        parents=[gen_parser],
    )
    parser_file.add_argument("filename", help="Input file (CSV or SDF)")
    parser_file.add_argument(
        "-o",
        "--output",
        help="Output file (SDF with pKa predictions)",
        default="qupkake_output.sdf",
    )
    parser_file.add_argument(
        "-s", "--smiles-col", help="Column name with SMILES strings", default="smiles"
    )
    parser_file.add_argument(
        "-n",
        "--name-col",
        help="Column name with molecule names. If not provided, the name of the molecule will be 'molecule_1', 'molecule_2', etc.",
        default="name",
    )

    parser_file.set_defaults(func=main_file)

    parser_smiles = subparsers.add_parser(
        "smiles",
        help="Predict pKa values for a SMILES string",
        aliases=["smi"],
        parents=[gen_parser],
    )
    parser_smiles.add_argument("smiles", help="SMILES string")
    parser_smiles.add_argument(
        "-n", "--name", help="Name of the molecule", default="molecule"
    )
    parser_smiles.add_argument(
        "-o",
        "--output",
        help="Output file (SDF with pKa predictions)",
        default="qupkake_output.sdf",
    )
    parser_smiles.set_defaults(func=main_smiles)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    parse_arguments(sys.argv[1:])


if __name__ == "__main__":
    main()
