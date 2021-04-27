import networkx as nx


def graph_builder(weights):
    '''
    '''

    graph = nx.MultiGraph()

    for weight in weights:
        graph.add_node(weight[1].split('-')[0])
        graph.add_node(weight[1].split('-')[1])
        graph.add_edge(weight[1].split('-')[0], weight[1].split('-')[1], weight=weight[0])

    return graph
