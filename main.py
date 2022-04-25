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
        network = Network(225, 5, 2)
    network.generate_a_day()
    analysis = Analysis(network)
    # analysis.pie_chart("./data/traffic_light/Traffic_light.G_infection_by_p0.csv", save_as="traffic_light.G")
    # analysis.pie_chart("./data/traffic_light/Traffic_light.Y_infection_by_p0.csv", save_as="traffic_light.Y")
    # analysis.pie_chart("./data/traffic_light/Traffic_light.R_infection_by_p0.csv", save_as="traffic_light.R")

    # network.generate_a_day()
    # analysis = Analysis(network)
    # analysis.degree_distribution_layers(both=True, experimental=False)
    # analysis.heatmap_asymptomatic_calibration()

    # disease_transmission = Disease_transmission(network, stoplight=Traffic_light.G)
    # disease_transmission.run_transmission(days=16, testing=True)


main()
