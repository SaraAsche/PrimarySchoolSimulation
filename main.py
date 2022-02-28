import sys

from network import Network
from analysis import Analysis


def main():
    print(sys.argv)
    if len(sys.argv) > 1:
        _, num_students, num_grades, num_classes, num_days = sys.argv

        network = Network(int(num_students), int(num_grades), int(num_classes))
        network.generate_iterations(int(num_days))
    else:
        network = Network(225, 5, 2)

    analysis = Analysis(network)

    G = network.generate_iterations(10)

    analysis.outlierDist(G)

    analysis.pixel_dist_school(G, old=True, both=True)

    analysis.heatmap(G)

    analysis.degree_distribution_layers(both=True, experimental=True, sim=G)


main()
