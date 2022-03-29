import sys

from network import Network
from analysis import Analysis
from disease_transmission import Disease_transmission


def main():
    print(sys.argv)
    if len(sys.argv) > 1:
        _, num_students, num_grades, num_classes, num_days = sys.argv

        network = Network(int(num_students), int(num_grades), int(num_classes))
        network.generate_iterations(int(num_days))
    else:
        network = Network(225, 5, 2)

    analysis = Analysis(network)

    # G = network.generate_iterations(10)
    G = network.generate_a_day()

    print(network.students[0].p_vector)

    # analysis.replica_degree(G, network)
    # analysis.replica_pixel(G, network)
    # analysis.pixel_dist_school(G, old=True, both=True)
    # analysis.degree_distribution_layers(sim=G, experimental=True, both=True)
    # disease_transmission = Disease_transmission(network)
    # disease_transmission.run_transmission(14)
    # analysis.heatmap(G)
    # analysis.runAnalysis2(G)

    # analysis.outlierDist(G)

    # analysis.modularity(G)


main()
