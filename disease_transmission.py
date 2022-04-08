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

from enums import Disease_states, Traffic_light
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

    def __init__(self, network, stoplight=None, Ias=0.4):
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
        ..
        """

        self.network = network
        self.graph = network.get_graph()
        self.students = network.get_students()
        self.day_no = 0
        self.p_0 = 0.01

        self.patient_zero = None
        # self.infectious_rates = {Disease_states.IAS: 0.1, Disease_states.IP: 1.3, Disease_states.IS: 0.1}  # FHI
        self.infectious_rates = {Disease_states.IAS: 0.2, Disease_states.IP: 0.4, Disease_states.IS: 0.1}
        self.Ias = Ias
        self.Ip = 1 - self.Ias
        self.positions = nx.spring_layout(self.graph, seed=10396953)
        self.stoplight = stoplight
        self.days = [self.graph]

    def update_ias_ip(self, Ias) -> None:
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
            patient_zero.set_day_infected(0)

    def set_patient_zero(self, person) -> None:
        """Sets a specific Person object to be patient zero"""
        self.patient_zero = person
        self.patient_zero.set_state(Helpers.get_infection_root(pA=self.Ias, pP=self.Ip))
        self.patient_zero.set_day_infected(0)

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
        j = 0
        for stud in self.students:
            # If the student has a state that is transmissible, its neighbours will have a probability of being exposed
            if stud.state in [Disease_states.IP, Disease_states.IAS, Disease_states.IS]:
                if not stud.get_tested():
                    for n in self.graph.neighbors(stud):
                        # infected = self.infectious_rates[stud.state] * self.graph.edges[(stud, n)]["count"] > 15
                        p_inf = 1 - (1 - self.p_0) ** (
                            self.graph.edges[(stud, n)]["count"] * self.infectious_rates[stud.state]
                        )
                        rand = random.random()
                        infected = rand < p_inf
                        # print(f"p_inf: {p_inf}, random: {rand}, infected: {infected}")
                        if infected and n.get_state() == Disease_states.S:
                            n.set_state(Disease_states.E)
                            n.set_day_infected(self.day_no)
                            n.set_infected_by(stud)

    def run_transmission(self, days, plot=True, Ias=0.4, testing=False) -> dict:
        """Simulate and draw the disease transmission for multiple days

        Generates a plot for transmission of each day in days. A new network
        is generated each day with differing interactions from previous days.

        Parameters
        ----------
        days int:
            The number of days the transmission should be simulated
        """

        self.update_ias_ip(Ias)  # makes sure Ip and Ias are set.

        days_dic = {}  # Keeps track of how many individuals are in the given state at time key

        for i in range(days):  # Meaning: 0-day-1
            if testing:
                self.weekly_testing()
            d = dict([(e, 0) for e in Disease_states])  # Keeps track of how many individuals are in a given state
            if i == 0:  # Day 0, no disease_transmission.
                self.generate_patient_zero()  # Patient zero is introduced

            else:
                if self.stoplight == Traffic_light.G:
                    self.graph = self.generate_green_stoplight(self.graph)
                elif self.stoplight == Traffic_light.O:
                    self.graph = self.generate_orange_stoplight(self.graph)
                elif self.stoplight == Traffic_light.R:
                    self.graph = self.generate_red_stoplight(self.graph)
                else:
                    self.graph = self.network.generate_a_day()  # A new network is generated each day

                self.days.append(self.graph)
                self.infection_spread()

            for stud in self.students:
                # Update state if conditions are fullfilled (x amount of days in state y)
                stud.add_day_in_state(self.Ias, self.Ip)
                # Update the amount of days an individual has been in state
                d[stud.state] += 1

            print(f"-------------Day {i}-------------")
            pprint(d)

            days_dic[i] = d
            self.day_no += 1

            # Sets the key (represents day i) to have the value: dict over states and people in that state
            if plot:
                if i == days:
                    # On the last day, the plot stays untill being dismissed
                    self.plot(block=True)
                else:
                    self.plot()

        return self.diff(days_dic)

    def isolate(self, graph):
        if self.stoplight in [Traffic_light.O, Traffic_light.R]:
            for stud in self.students:
                if stud.state == Disease_states.IS:
                    self.network.remove_all_interactions(graph, stud)

    def generate_green_stoplight(self, graph):
        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_orange_stoplight(self, graph):
        ## TODO: Associate cohorts with normal interaction, less elsewhere

        self.generate_cohorts()
        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_red_stoplight(self, graph):
        ## TODO: almost exclusively cohort interactions
        self.generate_cohorts()
        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_cohorts(self) -> None:

        grades = self.network.get_available_grades()
        classes = self.network.get_available_classes()
        grade_and_classes = [f"{i}{j}" for i in grades for j in classes]
        cohorts = []

        if self.stoplight == Traffic_light.O:
            for i in range(0, len(grade_and_classes) - 1, 2):
                cohorts.append(f"{grade_and_classes[i]}{grade_and_classes[i+1]}")
                students = list(
                    filter(
                        lambda x: x.get_class_and_grade() == grade_and_classes[i]
                        or x.get_class_and_grade() == grade_and_classes[i + 1],
                        self.students,
                    )
                )
                for stud in students:
                    stud.set_cohort(cohorts[-1])

        elif self.stoplight == Traffic_light.R:
            for class_group in grade_and_classes:
                students = list(filter(lambda x: x.get_class_and_grade() == class_group, self.students))
                for i in range(0, (len(students) // 2) + 1):
                    students[i].set_cohort(f"{class_group}1")
                    students[-i - 1].set_cohort(f"{class_group}2")

    def weekly_testing(self, recurr=7) -> None:
        ## Testing vil plukke opp asymtpomatiske og symptomatiske og isolere til R
        ## Må forbli isolerte til de er recovered.
        if self.day_no % 7 == 0 and self.day_no != 0:
            for stud in self.students:
                if stud.get_state() in [Disease_states.IAS, Disease_states.IS, Disease_states.IP]:
                    self.network.remove_all_interactions(self.graph, stud)
                    stud.set_tested(True)

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

    def diff(self, state_dict) -> dict:
        """Returns a dict of how many individuals that are not susceptible in the model for the days it is run"""
        for entry in state_dict:
            state_dict.update({entry: 236 - state_dict[entry][Disease_states.S]})
        return state_dict

    def asymptomatic_calibration(self) -> None:
        Ias_dict = {}
        for Is in range(0, 101, 10):
            Ias = (100 - Is) / 100
            Is = Is / 100

            day_list = np.zeros(35)  # 15
            for replica in range(0, 20):  # 20
                for key, val in self.run_transmission(35, plot=False, Ias=Ias).items():  # 14
                    day_list[key] += val

                self.network.reset_student_disease_states()
            day_list = day_list / 20
            Ias_dict[Ias] = day_list.tolist()

        with open("asymptomatic_calibration.pickle", "wb") as handle:
            pickle.dump(Ias_dict, handle)

    def R_null(self):
        """
        We assume self.run_transmission has aldready been run
        """
        # TODO: kjøre smitte i 30 dager, se hvor mange de som blir smittet i løpet av de 5 første dagene smitter videre. Kjøre gjennomsnitt.
        assert len(self.days) > 5, f"Disease transmission object has only {len(self.days)} days generated"
        infected_dict = dict([(stud.get_ID(), 0) for stud in self.students])

        # for day in range(5):
        #     infected.append(list(lambda x: x.get_infection_day() == day, self.students))

        for stud in filter(lambda x: x.get_state() != Disease_states.S, self.students):
            key = stud.get_infected_by()
            if key is None:
                infected_dict[None] = 1
            else:
                infected_dict[stud.get_infected_by().get_ID()] = (
                    infected_dict.get(stud.get_infected_by().get_ID(), 0) + 1
                )
        infected_day_5_or_less = list(
            filter(lambda x: x.get_day_infected() is not None and x.get_day_infected() <= 5, self.students)
        )

        summ = 0
        for stud in infected_day_5_or_less:
            if stud is not None:
                summ += infected_dict[stud.get_ID()]

        print(f"sum: {summ}")
        print(f"Infected day 5 or less: {len(infected_day_5_or_less)}")
        return summ / (len(infected_day_5_or_less) - 1)


if __name__ == "__main__":
    network = Network(num_students=236, num_grades=5, num_classes=2, class_treshold=23)

    disease_transmission = Disease_transmission(network)
    disease_transmission.run_transmission(15, plot=False)
    print(f"R_null: {disease_transmission.R_null()}")
    # disease_transmission.generate_cohorts()

    # disease_transmission.asymptomatic_calibration()
