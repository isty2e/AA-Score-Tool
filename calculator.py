from pathlib import Path
import tempfile
import numpy as np
from rdkit import Chem

from descriptors import AAScoreDescriptorCalculator
from pocket_extractor import PocketExtractor


class Model:
    def __init__(self, arr):
        self.arr = arr

    def predict(self, data):
        data = np.array(data)
        return np.sum(self.arr * data) - 0.999


class AAScoreCalculator:
    def __init__(
        self, model_file_path: Path = Path(__file__).parent / "models/model-final.npy"
    ) -> None:
        self.model = Model(np.load(model_file_path))
        self.desc_calculator = AAScoreDescriptorCalculator()

    def run_from_pocket(self, protein_rdmol, ligand_rdmol) -> float:
        descriptors = self.desc_calculator.run(protein_rdmol, ligand_rdmol)
        return self.model.predict(descriptors)

    def run(self, protein_rdmol, ligand_rdmol) -> float:
        with tempfile.TemporaryDirectory() as temp_dir:
            protein_path = Path(temp_dir) / "protein.pdb"
            ligand_path = Path(temp_dir) / "ligand.mol"

            Chem.MolToPDBFile(protein_rdmol, str(protein_path))
            Chem.MolToMolFile(ligand_rdmol, str(ligand_path))

            pocket_extractor = PocketExtractor(protein_path, ligand_path)
            pocket_rdmol = pocket_extractor.pocket_mol

            return self.run_from_pocket(pocket_rdmol, ligand_rdmol)
