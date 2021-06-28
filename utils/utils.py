import matplotlib.pyplot as plt
import networkx as nx

from spektral.data import DisjointLoader, BatchLoader
from keras import backend as K

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
    loader_tr = BatchLoader(data_tr, batch_size=batch_size)
    loader_te = BatchLoader(data_te, batch_size=batch_size)
    '''

    split = int(0.8 * len(data))
    data_tr, data_te = data[:split], data[split:]

    print(f'Number of training graphs: {data_tr.n_graphs}')
    print(f'Number of test graphs: {data_te.n_graphs}')

    loader_tr = DisjointLoader(data_tr, batch_size=batch_size)
    loader_te = DisjointLoader(data_te, batch_size=batch_size)

    return loader_tr, loader_te

def loss_plot(training_hist, out):

    plt.plot(training_hist.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(out + '_loss.png')

    return 0


def r_squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
