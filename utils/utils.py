import matplotlib.pyplot as plt
import networkx as nx

from spektral.data import DisjointLoader


def to_GPU(GPU):
    '''
    '''

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    return 0

def visual_graph(graph, out):
    '''
    '''

    nx.draw_spring(graph, with_labels = True)
    plt.savefig(out + ".png")

    return 0

def read_graph(graph):

    graph_pick = nx.readwrite.gpickle.read_gpickle(graph)

    return graph_pick


def save_graph(graph, out):

    nx.readwrite.gpickle.write_gpickle(graph, out.split('.')[0] + '.gpickle')

    return 0

def data_loaders(data, batch_size, epochs):
    '''
    '''

    split = int(0.8 * len(data))
    data_tr, data_te = data[:split], data[split:]

    loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_te = DisjointLoader(data_te, batch_size=batch_size)

    return loader_tr, loader_te
