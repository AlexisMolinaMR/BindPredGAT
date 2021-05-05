import os
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer

from scipy.sparse import csr_matrix

from spektral.data import Graph, Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import backend as K

def graph_builder(weights):
    '''
    '''

    graph_strength = nx.MultiGraph()
    graph_distance = nx.MultiGraph()

    for weight in weights:

        graph_strength.add_node(weight[1].split('-')[0])
        graph_strength.add_node(weight[1].split('-')[1])
        graph_strength.add_edge(weight[1].split('-')[0], weight[1].split('-')[1], weight=weight[0])

        graph_distance.add_node(weight[1].split('-')[0])
        graph_distance.add_node(weight[1].split('-')[1])
        graph_distance.add_edge(weight[1].split('-')[0], weight[1].split('-')[1], weight=weight[-1])

    return graph_strength, graph_distance

class MyDataset(Dataset):
    """

    """

    def __init__(self, path, out, target, **kwargs):
        self.in_path = path
        self.out_path = out
        self.target = pd.read_csv(target)

        super().__init__(**kwargs)

    def read(self):
        def make_graph(file):

            #Nodes
            graph_pick = nx.readwrite.gpickle.read_gpickle(self.out_path + file.split('.')[0] + '.gpickle')
            nodes = np.array(list(graph_pick.nodes))

            #Encode Nodes
            label = LabelEncoder()
            int_data = label.fit_transform(nodes)
            int_data = int_data.reshape(len(int_data), 1)

            onehot_data = OneHotEncoder(sparse=False)
            onehot_nodes = onehot_data.fit_transform(int_data)

            #Edges
            a = np.load(self.out_path + file.split('.')[0] + '_adj.npy')
            a = csr_matrix(a, dtype=np.float32)

            #Targets
            y = float(self.target.loc[self.target['ligand'] == file.split('.')[0]]['bindingEnergy'])

            return Graph(x=onehot_nodes, a=a, y=y)

        # We must return a list of Graph objects

        return [make_graph(file) for file in os.listdir(self.in_path) if file.endswith('.pdb') if not file.startswith('lig') if os.path.isfile(self.out_path + file.split('.')[0] + '.gpickle')]
