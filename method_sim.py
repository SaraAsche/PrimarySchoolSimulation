import sys

from network import Network
from analysis import Analysis
from disease_transmission import Disease_transmission
from enums import Traffic_light

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community


def create_sub_graphs(analysis):
    off_diagE = analysis.create_sub_graph_off_diagonal(analysis.network.get_graph(), True, False)
    gradeE = analysis.create_sub_graph_grade_class(analysis.network.get_graph(), False, True)
    classE = analysis.create_sub_graph_grade_class(analysis.network.get_graph(), True, False)

    return off_diagE, gradeE, classE


# Spring drawing saved
def spring_draw(G):
    weights = [None for _ in range(len(G.edges))]
    maximum_count = max(list(map(lambda x: x[-1]["count"], G.edges(data=True))))

    for i, e in enumerate(G.edges(data=True)):
        weights[i] = (0, 0, 0, e[-1]["count"] / (maximum_count - 300))

    color_map = []
    color_map2 = {1: "rosybrown", 2: "sienna", 3: "tan", 4: "darkgoldenrod", 5: "olivedrab"}

    for node in G.nodes():
        if node.get_grade() == 1:
            color_map.append("rosybrown")
        elif node.get_grade() == 2:
            color_map.append("sienna")
        elif node.get_grade() == 3:
            color_map.append("tan")
        elif node.get_grade() == 4:
            color_map.append("darkgoldenrod")
        elif node.get_grade() == 5:
            color_map.append("olivedrab")
        else:
            color_map.append("slategrey")

    pos = nx.spring_layout(G, seed=10396953)

    for i in range(1, 6):
        nodes = [x for x in list(filter(lambda x: x.get_grade() == i, G.nodes()))]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=20, node_color=color_map2[i], label=str(i))
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=weights,
    )

    plt.legend(scatterpoints=1, frameon=False)  # ["1", "2", "3", "4", "5"],

    plt.savefig("./fig_master/sim_graph.png", transparent=True, dpi=500)
    plt.show()
    # alpha=0.8
    # Gcc = G
    # pos = nx.spring_layout(Gcc, seed=10396953)
    # nx.draw_networkx_nodes(Gcc, pos, node_size=20, node_color=color_map)
    # nx.draw_networkx_edges(Gcc, pos, alpha=0.4, edge_color=weights)


# Def printe ut network characteristics
def general_analysis(graph):
    print("-----------------------------------")
    print(f"# of Nodes: {len(graph.nodes())}")
    print("-----------------------------------")
    print(f"# of edges: {len(graph.edges())}")
    print("-----------------------------------")

    degrees = []
    for deg in graph.degree:
        degrees.append(deg[1])

    print(f"Average degree: {np.mean(degrees)}")
    print("-----------------------------------")
    print(f"Network diameter: {nx.diameter(graph)}")
    print("-----------------------------------")
    aver_short = nx.average_shortest_path_length(graph, weight="weight")
    print(f"Average shortest path : {aver_short}")
    print("-----------------------------------")
    print(f"Average clustering: {nx.average_clustering(graph)}")
    print("-----------------------------------")
    weight_clust = nx.average_clustering(graph, weight="weight")
    print(f"Weighted average clustering: {weight_clust}")
    print("-----------------------------------")
    print(f"Network density: {nx.density(graph)}")
    print("-----------------------------------")
    print(f"Heterogeneity in cytoscape")
    print("-----------------------------------")
    cent = []
    for id, ce in nx.degree_centrality(graph).items():
        cent.append(ce)
    print(f"Average closeness centrality: {np.mean(cent)}")
    print("-----------------------------------")
    betw = []
    for id, bet in nx.betweenness_centrality(graph).items():
        betw.append(bet)
    print(f"Average betweenness centrality: {np.mean(betw)}")
    print("-----------------------------------")
    mod = modularity(graph)
    print(f"Modularity = {mod}")


def getClasses(G):

    # grades = nx.get_node_attributes(G, "klasse")
    grades = {}
    for node in G.nodes():
        grades[node] = node.get_class_and_grade()

    d = sorted(grades)

    class_list = {}

    for key, item in grades.items():
        if item not in class_list.keys():
            class_list[item] = [key]
        else:
            class_list[item].append(key)

    lst = []
    for key, val in class_list.items():
        lst.append(val)

    I = list(map(set, lst))

    return list(I)


def modularity(G):
    communities = getClasses(G)

    M = community.modularity(G, communities, "weight")

    return M


def main():
    network = Network(236, 5, 2, class_treshold=23)
    net = network.generate_a_day()

    analysis = Analysis(network)

    general_analysis(network.get_graph())

    spring_draw(network.get_graph())

    # Heatmap of sim
    analysis.heatmap(network.get_graph())

    # Degree dist layers.
    # analysis.degree_distribution_layers(both=True, experimental=True, sim=network.get_graph())

    # Pixel dist layers:
    # analysis.pixel_dist_school(network.get_graph(), old=True, both=True)


main()
