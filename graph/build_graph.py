import networkx as nx


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
