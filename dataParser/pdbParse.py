import math
import os
import rdkit

from rdkit import Chem
from rdkit.Chem import rdchem

from prody import parsePDB, writePDB, calcCenter

def read_PDB(path, lig_name):
    '''
    in: pdb file
    out: parsed Prody object and self-parse dict
    '''

    system = {}

    with open(path, 'r') as pdb:
        for lines in pdb:
            line = lines.strip()
            if line[0:4].strip() == "ATOM" or (line[0:6] == "HETATM" and line[17:20].strip() in ["MSE", "KCX"]):
                if not system:
                    system[path.split("/")[-1].split(".")[0]] = line + '\n'
                else:
                    system[path.split("/")[-1].split(".")[0]] += line + '\n'
            elif line[0:6] == "HETATM":
                if not system:
                    system[path.split("/")[-1].split(".")[0]] = line + '\n'
                else:
                    system[path.split("/")[-1].split(".")[0]] += line + '\n'

    prody_system = parsePDB

    return system, prody_system

def ligand_parse_write(path, out, lig_name):
    '''
    '''

    pose = parsePDB(path)

    if os.path.isfile(out + 'lig_' + path.split('/')[-1]):
        pass
    else:
        writePDB(out + 'lig_' + path.split('/')[-1],
                 pose.select('hetero and resname {}'.format(lig_name)))

    ligand_path = out + 'lig_' + path.split('/')[-1]

    return ligand_path

def binding_pocket_selection(system, prody_system, ligand_name, selection_radius, center):
    '''

    '''

    amino = ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN',
             'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET']

    two_let_atom_code = ['Br', 'FE']

    coord = []
    x_coord = []
    binding_pocket = []
    ligand = []

    min_coord = None
    max_coord = None

    for line in system:
        structure = system[line].split('\n')

    for line in structure:
        if line[17:20] == ligand_name:
            coord.append(float(line[30:38]))
            coord.append(float(line[38:46]))
            coord.append(float(line[46:54]))

            ligand_atom = (line[17:20].strip(), line[12:16].strip(), line[30:38].strip(
            ), line[38:46].strip(), line[46:54].strip(), line[-2:].strip())
            ligand.append(ligand_atom)
            ligand_atom = ()

    for i in range(0, len(coord), 3):
        x_coord.append(float(coord[i]))

    if center == 'geometric':

        print("\nGeometric ligand center selected")

        sphere_center_x = (coord[coord.index(max(x_coord))] + coord[coord.index(min(x_coord))]) / 2
        sphere_center_y = (coord[coord.index(max(x_coord))+1] +
                           coord[coord.index(min(x_coord))+1]) / 2
        sphere_center_z = (coord[coord.index(max(x_coord))+2] +
                           coord[coord.index(min(x_coord))+2]) / 2

        print('\n')
        print("Sphere center coordinates: ", sphere_center_x, sphere_center_y, sphere_center_z)
        print("Sphere radius: ", selection_radius)

        for line in structure:
            if (line[0:6].strip() == "ATOM" or line[0:6].strip() == "HETATM") and line[17:20].strip() in amino:

                x1 = math.pow((float(line[30:38]) - sphere_center_x), 2)
                y1 = math.pow((float(line[38:46]) - sphere_center_y), 2)
                z1 = math.pow((float(line[46:54]) - sphere_center_z), 2)

                if (x1 + y1 + z1) <= selection_radius**2:
                    if line[-3:].strip() in two_let_atom_code:
                        atom = (line[17:20].strip(), line[12:16].strip(), line[7:11].strip(), line[30:38].strip(
                        ), line[38:46].strip(), line[46:54].strip(), line[-3:].strip())

                    else:
                        atom = (line[17:20].strip(), line[12:16].strip(), line[7:11].strip(), line[30:38].strip(
                        ), line[38:46].strip(), line[46:54].strip(), line[-3:].strip()[0])
                    binding_pocket.append(atom)
                    atom = ()

        print("Number of protein atoms selected: {}".format(len(binding_pocket)))
        print('Number of ligand atoms selected: {}'.format(len(ligand)))
        print('Total number of atoms selected: {}'.format(len(binding_pocket)+len(ligand)))

    elif center == 'mass':

        print('\nLigand mass center selected')
        ligand_selection = p.select('not water and hetero')

        weights = ligand_selection.getMasses()
        mass_center = calcCenter(ligand_selection, weights)

        sphere_center_x = mass_center[0]
        sphere_center_y = mass_center[1]
        sphere_center_z = mass_center[2]

        print("\nSphere center coordinates: {}, {}, {}".format(
            sphere_center_x, sphere_center_y, sphere_center_z))
        print("Sphere radius: {}".format(selection_radius))

        for line in structure:
            if (line[0:6].strip() == "ATOM" or line[0:6].strip() == "HETATM") and line[17:20].strip() in amino:

                x1 = math.pow((float(line[30:38]) - sphere_center_x), 2)
                y1 = math.pow((float(line[38:46]) - sphere_center_y), 2)
                z1 = math.pow((float(line[46:54]) - sphere_center_z), 2)

                if (x1 + y1 + z1) <= selection_radius**2:
                    if line[-3:].strip() in two_let_atom_code:
                        atom = (line[17:20].strip(), line[12:16].strip(), line[30:38].strip(
                        ), line[38:46].strip(), line[46:54].strip(), line[-3:].strip())

                    else:
                        atom = (line[17:20].strip(), line[12:16].strip(), line[30:38].strip(
                        ), line[38:46].strip(), line[46:54].strip(), line[-3:].strip()[0])
                    binding_pocket.append(atom)
                    atom = ()

        print("Number of protein atoms selected: {}".format(len(binding_pocket)))
        print('Number of ligand atoms selected: {}'.format(len(ligand)))
        print('Total number of atoms selected: {}'.format(len(binding_pocket)+len(ligand)))

    return binding_pocket, ligand

def ligand_atom_type_calc(ligand, ligand_path):

    ligand_atom_types = {'H': 0, 'C': 0, 'N': 0, 'O': 0,
                         'F': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Br': 0, 'I': 0}

    mol = Chem.MolFromPDBFile(ligand_path)

    index = 0

    while index <= mol.GetNumAtoms()-1:

        atomObj = mol.GetAtomWithIdx(index)
        symbol = str(atomObj.GetSymbol())

        ligand_atom_types[symbol] += 1

        Hybridization = str(atomObj.GetHybridization())
        Aromacity = atomObj.GetIsAromatic()

        if Aromacity:
            ligand[index] = ligand[index] + \
                (symbol + '_' + Hybridization + '_AROM_' + str(index),)
        else:
            ligand[index] = ligand[index] + \
                (symbol + '_' + Hybridization + '_' + str(index),)

        index += 1

    return ligand
