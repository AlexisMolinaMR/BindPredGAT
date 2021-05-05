import networkx as nx
import numpy as np

from spektral.utils import laplacian

def compute_adjacency_matrix(graph, out):
    '''
    '''

    adjacency_matrix = nx.to_numpy_matrix(graph)

    np.save(out.split('.')[0] + '_adj.npy', adjacency_matrix)

    return adjacency_matrix

def compute_laplacian_matrix(graph):
    '''
    '''

    laplacian_matrix = laplacian(graph)

    return laplacian_matrix
