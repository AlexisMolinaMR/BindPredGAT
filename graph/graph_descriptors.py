import networkx as nx
import numpy as np

def compute_adjacency_matrix(graph):
    '''
    '''

    adjacency_matrix = nx.to_numpy_matrix(graph)

    return adjacency_matrix

def compute_laplacian_matrix(graph):
    '''
    '''

    laplacian_matrix = nx.normalized_laplacian_matrix(graph)

    return laplacian_matrix
