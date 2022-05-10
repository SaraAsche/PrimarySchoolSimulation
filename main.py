import sys

from network import Network
from analysis import Analysis
from disease_transmission import Disease_transmission
from enums import Traffic_light


def main():
    print(sys.argv)
    if len(sys.argv) > 1:
        _, num_students, num_grades, num_classes, num_days = sys.argv

        network = Network(int(num_students), int(num_grades), int(num_classes))
        network.generate_iterations(int(num_days))
    else:
        network = Network(236, 5, 2)
    network.generate_a_day()
    analysis = Analysis(network)

    # Analysis of degree and interaction distribution as well as heatmap.
    analysis = Analysis(network)
    analysis.degree_distribution_layers(both=True, experimental=True, sim=network.get_graph())
    analysis.pixel_dist_school(network.get_graph(), old=True, both=True)
    analysis.heatmap()

    # Disease transmission run on 16 days
    disease_transmission = Disease_transmission(network, stoplight=Traffic_light.G)
    disease_transmission.run_transmission(days=16, testing=True)


main()
