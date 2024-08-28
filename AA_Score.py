import argparse
from pathlib import Path

from rdkit import Chem

from calculator import AAScoreCalculator


def main(
    receptor_path: Path,
    ligand_path: Path,
    output_path: Path | None,
    verbose: bool = False,
) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write("name,score\n")

    protein_rdmol = Chem.MolFromPDBFile(str(receptor_path), removeHs=False)

    aa_score_calculator = AAScoreCalculator()
    for i, ligand_rdmol in enumerate(
        Chem.SDMolSupplier(str(ligand_path), removeHs=False)
    ):
        ligand_name = ligand_rdmol.GetProp("_Name") or f"ligand_{i + 1}"
        score = aa_score_calculator.run(protein_rdmol, ligand_rdmol)

        if verbose:
            print(f"{ligand_name}: {score}")

        if output_path is not None:
            with output_path.open("a") as f:
                f.write(f"{ligand_name},{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse AA Score prediction parameters")

    parser.add_argument(
        "--receptor",
        type=Path,
        help="the file of binding pocket, only support PDB format",
    )
    parser.add_argument(
        "--ligand", type=Path, help="the file of ligands, support mol2, mol, sdf, PDB"
    )
    parser.add_argument(
        "--output", type=Path, help="the output file for recording scores"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to print the score to stdout"
    )

    args = parser.parse_args()

    main(args.receptor, args.ligand, args.output, args.verbose)
