from pathlib import Path
import tempfile
import argparse
import copy
import os
import sys
import warnings
from random import randint

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem

pd.options.mode.chained_assignment = None

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def conc(a, b, c):
    return [a, b, c]


def add_xyz(ligu):
    ligu["xyz"] = ligu.apply(
        lambda row: conc(row["x_coord"], row["y_coord"], row["z_coord"]), axis=1
    )
    return ligu


def cal_dist(a, b):
    return round(np.linalg.norm(np.array(a) - np.array(b)), 2)


def get_min_dist(am, ligu):
    ligu["pro_xyz"] = [am] * ligu.shape[0]
    ligu["dist"] = ligu.apply(lambda row: cal_dist(row["xyz"], row["pro_xyz"]), axis=1)
    return min(ligu["dist"])


class PocketExtractor:
    def __init__(self, protein_file: Path, ligand_file: Path) -> None:
        """
        protein_file: format pdb
        ligand_file: format mol
        """
        if not protein_file.exists():
            raise ValueError(f"{protein_file} does not exist.")
        if not ligand_file.exists():
            raise ValueError(f"{ligand_file} does not exist.")

        self.protein_file = protein_file
        self.ligand_file = ligand_file

        self.ligand_mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
        self.pocket_mol = self._process_pro_and_lig()

    def save_pocket(self, output_path: Path) -> None:
        Chem.MolToPDBFile(self.pocket_mol, self.output_path)

    def _process_pro_and_lig(self):
        ppdb = PandasPdb()
        ppdb.read_pdb(self.protein_file)
        protein_biop = ppdb.df["ATOM"]

        pro_cut, _ = self._select_cut_residue(protein_biop, self.ligand_mol, cut=5.5)

        ppdb.df["ATOM"] = pro_cut

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_pdb = Path(tmp_dir) / "tmp.pdb"
            ppdb.to_pdb(path=str(tmp_pdb), records=["ATOM"])

            return Chem.MolFromPDBFile(tmp_pdb, removeHs=False)

    def _select_cut_residue(self, protein_biop, ligand_mol, cut=5.0):
        """
        pro: biopandas DataFrame
        lig: rdkit mol
        """
        pro = self._cal_pro_min_dist(protein_biop, ligand_mol)

        pro["chain_rid"] = pro.apply(
            lambda row: str(row["chain_id"]) + str(row["residue_number"]), axis=1
        )
        pros = pro[pro["min_dist"] < cut]
        pros_near_lig = copy.deepcopy(pros)
        use_res = list(set(list(pros["chain_rid"])))
        pro = pro[pro["chain_rid"].isin(use_res)]
        pro = pro.drop(["chain_rid"], axis=1)
        return pro, pros_near_lig

    def _get_ligu(self, ligand_mol):
        mol_ligand_conf = ligand_mol.GetConformers()[0]
        pos = mol_ligand_conf.GetPositions()
        df = pd.DataFrame(pos)
        df.columns = ["x_coord", "y_coord", "z_coord"]
        return df

    def _cal_pro_min_dist(self, protein_biop, ligand_mol):
        protein_biop = add_xyz(protein_biop)

        ligu = self._get_ligu(ligand_mol)
        ligu = add_xyz(ligu)
        protein_biop["min_dist"] = protein_biop.apply(
            lambda row: get_min_dist(row["xyz"], ligu), axis=1
        )
        return protein_biop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--output_pocket", type=Path, required=True)

    args = parser.parse_args()

    pocket_extractor = PocketExtractor(args.protein, args.ligand)
    pocket_extractor.save_pocket(args.output_pocket)
