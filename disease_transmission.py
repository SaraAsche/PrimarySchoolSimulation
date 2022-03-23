"""disease_transmission class

Describes the how disease may spread on a Network object. The Disease_transmission
class is initialised with a Network object, which is used to simulate viral transmission.
The class uses the disease states presented in the enum Disease_states, and the course
of the disease is usually from:
Susceptible -> Exposed -> Infected (asymptomatic or pre-symtomatic -> symptomatic) -> Recovered

  Typical usage example:
    network = Network(num_students=236, num_grades=5, num_classes=2, class_treshold=23)
    disease_transmission = Disease_transmission(network) 
    disease_transmission.run_transmission(14)

  Which will give the output:
    A matplotlib image of how disease spreads over the 14 days the transmission is modelled
    An overview of how many individuals are in a given disease state at a given time

Author: Sara Johanne Asche
Date: 14.02.2022
File: disease_transmission.py
"""

from enums import Disease_states
from network import Network
from person import Person
import random
from helpers import Helpers
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle as pickle


class Disease_transmission:
    """A class to simulate disease spread on a Network object

    Simulates how individuals move through different disease states and infect
    and expose other students.

    Attributes
    ----------
    ....

    Methods
    -------
    .....
    """

    def __init__(self, network, Ias=0.4):
        """Inits Disease_transmission object with a Network object

        Parameters
        ----------
        network: Network
            Network object containing interactions between Person objects
        students: list
            List of all person objects that are a part of Network
        patient_zero: Person
            The first person to introduce disease into the Network
        day_number: int
            The number of days the model has been run
        infectious_rates: dict
            Keeps track of the likelyhood of a Person having an asymptomatic
            or presymptomatic course of the disease.

        Methods
        --------
        init()
        """

        self.network = network
        self.students = network.get_students()
        self.patient_zero = None
        # self.graph = network.get_graph()
        day_number = 0
        self.infectious_rates = {Disease_states.IAS: 0.2, Disease_states.IP: 0.2, Disease_states.IS: 0.4}
        self.Ias = Ias
        self.Ip = 1 - self.Ias
        # self.init()

    def init(self):
        """Initialises the Disease_transmission object with generating a day, patient zero and a layout

        day_one is generated alongside the first individual at the school to get infected, patient_zero.
        In addition, the positions for the disease spread graph are set using a networkX's spring_layout.

        """
        self.day_one = self.network.generate_a_day(disease=True)
        for stud in self.students:
            stud.add_day_in_state(self.Ias, self.Ip)
        self.generate_patient_zero()
        self.positions = nx.spring_layout(self.day_one, seed=10396953)

    def update_ias_ip(self, Ias):
        self.Ias = Ias
        self.Ip = 1 - self.Ias

    def get_day(self) -> int:
        return self.day_number

    def set_day(self, day) -> None:
        self.day_number = day

    def get_all_states(self) -> list:
        state_list = []
        for student in self.students:
            state_list.append(student.get_state())
        return state_list

    def generate_patient_zero(self, num=1) -> None:
        """Generates num students in the network that are infected

        Default is to start with one infected individual with num=1.

        """
        for _ in range(num):
            patient_zero = random.choice(self.students)
            patient_zero.set_state(Helpers.get_infection_root(pA=self.Ias, pP=self.Ip))

    def set_patient_zero(self, person) -> None:
        """Sets a specific Person object to be patient zero"""
        self.patient_zero = person
        self.patient_zero.set_state("I")

    def get_patient_zero(self) -> Person:
        return self.patient_zero

    def get_susceptible(self) -> list:
        """Returns a list of all susceptible Person objects in the Network"""
        susceptible = []
        for student in self.students:

            if student.get_state() == "S":
                susceptible.append(student)
        return susceptible

    def get_exposed(self) -> list:
        """Returns a list of all exposed Person objects in the Network"""
        exposed = []
        for student in self.students:

            if student.get_state() == "E":
                exposed.append(student)
        return exposed

    def infection_spread(self) -> None:
        """Simulate disease transmission for one day

        Using the graph initialised for disease_transmission, the students that are
        either infected presymptomatic, infected asymptomatic or infected symptomatic
        are looped over and their neighbors have a given probability of getting exposed

        """
        graph = self.network.get_graph()
        for stud in self.students:
            if stud.state in [Disease_states.IP, Disease_states.IAS, Disease_states.IS]:
                for n in graph.neighbors(stud):
                    infected = self.infectious_rates[stud.state] * graph.edges[(stud, n)]["count"] > 10
                    if infected:
                        n.set_state(Disease_states.E)

    def run_transmission(self, days, plot=True, Ias=0.4) -> None:
        """Simulate and draw the disease transmission for multiple days

        Generates a plot for transmission of each day in days. A new network
        is generated each day with differing interactions from previous days.

        Parameters
        ----------
        days int:
            The number of days the transmission should be simulated
        """

        self.update_ias_ip(Ias)

        self.init()

        d = dict([(e, 0) for e in Disease_states])
        for stud in self.students:
            d[stud.state] += 1
        days_dic = {}
        days_dic[0] = d
        # print(f"-------------Day {0}-------------")
        # pprint(d)
        if plot:
            self.plot()
        for i in range(days):
            day = self.network.generate_a_day(disease=True)
            self.infection_spread()
            d = dict([(e, 0) for e in Disease_states])
            for stud in self.students:
                self.students[i].add_day_in_state(self.Ias, self.Ip)
                d[stud.state] += 1

            # print(f"-------------Day {i+1}-------------")
            # pprint(d)
            days_dic[i + 1] = d
            if plot:
                if i == days - 1:
                    self.plot(block=True)
                else:
                    self.plot()
        # print(f"-----------IAS: {self.Ias}, IP: {self.Ip}-----------")
        return self.diff(days_dic)

    def diff(self, state_dict):
        for entry in state_dict:
            state_dict.update({entry: 236 - state_dict[entry][Disease_states.S]})
        return state_dict

    def plot(self, interval=1, block=False) -> None:
        """Plots the Disease_states of each node at a given day

        Uses colors to represent the different Disease_states.
        If the plot is of the last day, the plot stays open and
        doeas not close after interval amount of seconds.
        Default values are interval=1 and block=False.

        Parameters
        ----------
        interval int:
            How long of a time, in seconds, a plot should be shown
        block bool:
            Keeps track of whether or not the plot is the last day defined
        """
        plt.clf()
        G = self.network.get_graph()
        sizes = [100 for _ in range(len(G.nodes))]
        weights = [None for _ in range(len(G.edges))]
        edge_color = ["grey" for _ in range(len(G.nodes))]

        for i, stud in enumerate(G.nodes):
            if stud.state == Disease_states.E:
                edge_color[i] = "yellow"
            elif stud.state in [Disease_states.IP, Disease_states.IS]:
                edge_color[i] = "red"
            elif stud.state == Disease_states.IAS:
                edge_color[i] = "blue"
            elif stud.state == Disease_states.R:
                edge_color[i] = "lightgreen"

        maximum_count = max(list(map(lambda x: x[-1]["count"], G.edges(data=True))))
        for i, e in enumerate(G.edges(data=True)):
            weights[i] = (0, 0, 0, e[-1]["count"] / maximum_count)

        nx.draw(
            G,
            with_labels=False,
            node_color=edge_color,
            pos=self.positions,
            node_size=sizes,
            edge_color=weights,
        )

        if block:
            plt.show(block=True)
        else:
            plt.pause(interval)

    def asymptomatic_calibration(self):
        Ias_dict = {}
        for Is in range(0, 101, 10):
            Ias = (100 - Is) / 100
            Is = Is / 100

            day_list = np.zeros(15)  # 15
            for replica in range(0, 20):  # 20
                for key, val in self.run_transmission(14, plot=False, Ias=Ias).items():  # 14
                    day_list[key] += val
            day_list = day_list / 20
            Ias_dict[Ias] = day_list.tolist()
            self.network.reset_student_disease_states()
        print(Ias_dict)
        with open("asymptomatic_calibration.pickle", "wb") as handle:
            pickle.dump(Ias_dict, handle)


if __name__ == "__main__":
    network = Network(num_students=236, num_grades=5, num_classes=2, class_treshold=23)
    disease_transmission = Disease_transmission(network)
    # disease_transmission.run_transmission(14)

    disease_transmission.asymptomatic_calibration()

    # disease_transmission.generate_patient_zero()
    # print(disease_transmission.get_all_states())
