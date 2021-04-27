import os
import math

import numpy as np
import pandas as pd


def elementSubgraphsWeights(atom_interactions, types, decay_function, ligand_atom_types, protein_atom_types):
    '''
    Compute subgraph weights using either exponential or Lorentz decay functions.
    '''

    weights = []
    count = 0
    atom_atom = []
    atom_rev = []
    uni_inter = []

    final_weigths = []

    if decay_function == 'expo':
        for inter in atom_interactions:
            type_weight = (np.exp(-abs(inter[-1])/3), "{}-{}".format(inter[0], inter[2]))
            weights.append(type_weight)
            type_weight = ()

    elif decay_function == 'lorentz':
        for inter in atom_interactions:
            type_weight = ((1/1 + (abs(inter[-1])/3)**5), "{}-{}".format(inter[0], inter[2]))
            weights.append(type_weight)
            type_weight = ()

    atom_atom.sort()

    return final_weigths, atom_atom


def atomSubgraphsWeights(atom_interactions, types, decay_function, ligand_atom_types, protein_atom_types):
    '''
    Compute subgraph weights using either exponential or Lorentz decay functions.
    '''

    weights = []
    count = 0
    atom_atom = []
    atom_rev = []
    uni_inter = []

    final_weigths = []

    if decay_function == 'expo':
        for inter in atom_interactions:
            type_weight = (np.exp(-abs(inter[-2])/3), "{}-{}".format(inter[2], inter[1] + '_' + inter[-1]))
            weights.append(type_weight)
            type_weight = ()

    elif decay_function == 'lorentz':
        for inter in atom_interactions:
            type_weight = ((1/1 + (abs(inter[-2])/3)**5), "{}-{}".format(inter[2], inter[1] + '_' + inter[-1]))
            weights.append(type_weight)
            type_weight = ()

    for i in range(len(ligand_atom_types)):
        for j in range(len(protein_atom_types)):
            atom_atom.append(("{}-{}".format(ligand_atom_types[i], protein_atom_types[j])))
            if ligand_atom_types[i] != protein_atom_types[j] and "{}-{}".format(ligand_atom_types[i], protein_atom_types[j]) not in atom_rev:
                atom_rev.append(("{}-{}".format(protein_atom_types[j], ligand_atom_types[i])))

    weights = list(set(weights))

    final_weigths = weights

    atom_atom.sort()

    return final_weigths, atom_atom
