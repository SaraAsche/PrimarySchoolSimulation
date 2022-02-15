import sys

from network import Network
from analysis import Analysis


def main():
    print(sys.argv)
    if len(sys.argv) > 1:
        _, num_students, num_grades, num_classes, num_days = sys.argv

        network = Network(int(num_students), int(num_grades), int(num_classes))
        network.generateXdays(int(num_days))
    else:
        network = Network(
            225,
            5,
            2,
        )

    analysis = Analysis(network)

    G = network.generateXdays(10)

    analysis.pixel_dist_school(G, old=True, both=True)
    analysis.heatmap(G)

    analysis.plotDegreeDistSubGraphs(both=True, experimental=True, sim=G)

    # analysis.displayNetwork(G)
    # analysis.runAnalysis(network.generateXdays(10))


main()
