import argparse

from schrodinger.structure import PDBWriter, StructureReader
from schrodinger.structutils import assignbondorders, build


def repair_mol(in_file, out_file):
    prot = next(StructureReader(in_file))
    assignbondorders.assign_st(prot, use_ccd="use_ccd")
    build.add_hydrogens(prot)
    with PDBWriter(out_file) as writer:
        writer.append(prot)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_file", help="Input structure file (Maestro, PDB, or SD, mol2, mol)"
    )
    parser.add_argument("out_file", help="Input structure file (PDB)")
    opts = parser.parse_args()

    in_file = opts.in_file
    out_file = opts.out_file
    repair_mol(in_file, out_file)
