import matplotlib.pyplot as plt
import networkx as nx

def to_GPU(GPU):
    '''
    '''

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    return 0

def visual_graph(graph, out, run):
    '''
    '''

    nx.draw_spring(graph, with_labels = True)
    plt.savefig(out + "{}.png".format(run))

    return 0
