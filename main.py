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
            # parameterList=[1.00000026e00, 4.99999984e-01,1.10000005e02,7.99999986e-01,3.99999977e00,5.00000014e-01,2.50000008e04,9.99996649e-02]
        )
        # network.generateXdays(8)

    analysis = Analysis(network)
    # analysis.heatmap(network.generate_a_day())
    # analysis.histDistributionLog(network.generateXdays(8))
    # G = analysis.createSubGraphWithoutGraph(network.generateXdays(8), False, True)
    G = network.generateXdays(10)

    # G = network.generate_a_day()
    analysis.pixel_dist_school(G, old=True)
    analysis.heatmap(G)
    analysis.histDistributionLog(G)
    # analysis.displayNetwork(G)
    # analysis.runAnalysis(network.generateXdays(10))


main()
