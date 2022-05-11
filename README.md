# Modelling the transmission of COVID-19 in Primary schools

## A framework for modeling interactions in primary schools

The current GitHub presents the framework used to model primary school interactions in Sara Johanne Asche's Master thesis. A thourogh introduction into the folders will be given in this README.md. In addition, each file is documented with a description of the overall file and its fuctions.

First, all data the generated and used throughout the thisis is shown in the data folder, which is separated into Empiric_network, empiric_vs_model, fig_master, pickles, traffic_light and weekly_testing. Empiric_network contains the accumulated network by Barrat et al.[^1], whilst fig_master all illustrations generated in this thesis saved. The pickles folder contains pickled dictionaries of the calculated degree and interaction distributions. Furthermore, empiric_vs_model contains the data generated to test transmission on the model versus the empiric network, and traffic_light and weekly_testing also contains transmission data for different traffic light and testing strategies. Finally, the asymptomatic_symptomatic folder contains the R0 values generated with either only asymptomatic or symptomatic cases in the system.

To generate a network, make sure to import Network as well as Analysis if you would like to display or run any network analysis on it. In addition, the Disease_transmission class alongside Traffic_light needs to be imported in order to run transmission on the generated network. A typical user case will be shown below, where a network is generated and first, network analysis is run on it and then disease transmission. The main.py file is used to create and network objects and run transmission on them. Main also allows for running the model in terminal by using sys.argv.

```python
from network import Network
from analysis import Analysis
from disease_transmission import Disease_transmission
from enums import Traffic_light

network = Network(236, 5, 2)
network.generate_a_day()

# Analysis of degree and interaction distribution as well as heatmap.
analysis = Analysis(network)
analysis.degree_distribution_layers(both=True, experimental=True, sim=network.get_graph())
analysis.pixel_dist_school(network.get_graph(), old=True, both=True)
analysis.heatmap()

# Disease transmission run on 16 days
disease_transmission = Disease_transmission(network, stoplight=Traffic_light.G)
disease_transmission.run_transmission(days=16, testing=True)

```

This should produce a plot of transmission on the network and be a different variation of the image shown below

![This is an image](https://github.com/SaraAsche/PrimarySchoolSimulation/blob/master/data/fig_master/transmission.png)

If there are any questions regarding the use of the code, do not hesitate to contact me on saraj.asche@gmail.com.

[^1]:
    Alain Barrat, Ciro Cattuto, Alberto. E. Tozzi, Philippe Vanhems, and
    Nicholar Voirin. Measuring contact patterns with wearable sensors: methods, data characteristics and applications to data-driven simulations of infectious diseases. Clinical Microbiology and Infection, 20(1):10–16, 2014. doi: 10.1111/1469-0691.12472
