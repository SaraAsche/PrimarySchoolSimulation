import sys

import networkx

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
            parameterList=[
                1.00000003e00,
                5.00000065e-01,
                1.10000000e02,
                8.00000007e-01,
                4.00000004e00,
                4.99999944e-01,
                2.49999998e04,
                9.99999633e-02,
            ],
        )
        # network.generateXdays(8)

    analysis = Analysis(network)
    # analysis.heatmap(network.generate_a_day())
    # analysis.histDistributionLog(network.generateXdays(8))
    # G = analysis.createSubGraphWithoutGraph(network.generateXdays(8), False, True)
    G = network.generateXdays(20)
    # G = network.generate_a_day()
    analysis.pixel_dist_school(G, old=True)
    analysis.heatmap(G)
    analysis.histDistributionLog(G)
    # analysis.displayNetwork(G)
    # analysis.runAnalysis(network.generateXdays(10))


main()
