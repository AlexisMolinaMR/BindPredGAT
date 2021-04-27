import matplotlib.pyplot as plt
import networkx as nx


def visual_graph(graph, out, run):
    '''
    '''

    nx.draw_spring(graph, with_labels = True)
    plt.show()
    plt.savefig(out + "{}.png".format(run))

    return 0
