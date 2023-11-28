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


def main_file(args):
    print(args)


def main_smiles(args):
    from importlib import resources as impresources

    from . import models

    models = impresources.files(models)
    print(models)

    print(args)


def create_dirs(args):
    """
    Creates the directory structure for the project.
    """
    import os

    root = args.root
    dirs = [
        os.path.join(root, "raw"),
        os.path.join(root, "processed"),
        os.path.join(root, "molecules"),
        os.path.join(root, "output"),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def parse_arguments(args):
    gen_parser = MyParser(
        add_help=False,
    )

    gen_parser.add_argument(
        "--root", type=str, default="data", help="Root directory for processing data."
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
        help="Output file (CSV with pKa predictions)",
        default="qupkake_output.csv",
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
        help="Output file (CSV with pKa predictions)",
        default="qupkake_output.csv",
    )
    parser_smiles.set_defaults(func=main_smiles)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    parse_arguments(sys.argv[1:])


if __name__ == "__main__":
    main()
