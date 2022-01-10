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
        network = Network(225, 5, 2)
        # network.generateXdays(8)
    
    analysis = Analysis(network)
    #analysis.heatmap(network.generate_a_day())
    #analysis.histDistributionLog(network.generateXdays(8))
    analysis.runAnalysis(network.generateXdays(10))

main()