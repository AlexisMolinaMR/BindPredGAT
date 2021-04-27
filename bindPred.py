import os
import sys
import yaml

from dataParser.pdbParse import read_PDB, binding_pocket_selection, ligand_parse_write, ligand_atom_type_calc

from graph.distComp import elementsDistanceCalc, atomTypesDistanceCalc
from graph.weigthsCalc import atomSubgraphsWeights, elementSubgraphsWeights
from graph.build_graph import graph_builder

from utils.utils import visual_graph

def parseyaml():

    with open(sys.argv[1]) as ctrl_file:
        params = yaml.load(ctrl_file, Loader=yaml.FullLoader)

    return params

def main():

    param_args = parseyaml()

    for file in os.listdir(param_args['path']):
        if file.endswith('.pdb') and not file.startswith('lig'):
            system, prody_system = read_PDB(param_args['path'] + file, param_args['ligand_name'])
            selected_protein, selected_ligand = binding_pocket_selection(system, prody_system, param_args['ligand_name'], param_args['selection_radius'], param_args['center'])

            if param_args['nodes'] == 'atoms':

                ligand_path = ligand_parse_write(
                    path=param_args['path'] + file, out=param_args['output'], lig_name=param_args['ligand_name'])
                selected_ligand_at = ligand_atom_type_calc(
                    ligand=selected_ligand, ligand_path=ligand_path)
                interactions, atom_types, ligand_atom_types, protein_atom_types = atomTypesDistanceCalc(
                    binding_pocket=selected_protein, ligand=selected_ligand_at)
                final_weigths, atom_combinations = atomSubgraphsWeights(atom_interactions=interactions, types=atom_types, decay_function=param_args['decay_function'],
                                                                        ligand_atom_types=ligand_atom_types, protein_atom_types=protein_atom_types)

            elif param_args['nodes'] == 'elements':
                interactions, elements, ligand_elements, protein_elements = elementsDistanceCalc(
                    binding_pocket=selected_protein, ligand=selected_ligand)
                final_weigths, atom_combinations = elementSubgraphsWeights(atom_interactions=interactions, types=elements, decay_function=param_args['decay_function'],
                                                                           ligand_atom_types=ligand_elements, protein_atom_types=protein_elements)

            graph_strength, graph_distance = graph_builder(weights=final_weigths)
            visual_graph(graph_strength, out=param_args['output'], run=param_args['run'] + '_graph_strength')
            visual_graph(graph_distance, out=param_args['output'], run=param_args['run'] + '_graph_distance')

    return 0


if __name__ == '__main__':
    main()
