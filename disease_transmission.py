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

from cProfile import label
from cgi import test

from matplotlib import testing
from enums import Disease_states, Traffic_light
from network import Network
from person import Person
import random
from helpers import Helpers
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import pickle as pickle
import seaborn as sns
import sys


class Disease_transmission:
    """A class to simulate disease spread on a Network object

    Simulates how individuals move through different disease states and infect
    and expose other students.

    Attributes
    ----------
    network : Network
         Network object containing a nx.Graph and students present in the network
    stoplight : Traffic_light
        Contains a Traffic_light entry from the traffic light model (green ("G"), yellow ("Y"), red("R")) or None
    graph : nx.Graph
        nx.Graph object containing interactions between Person objects
    students : list
        list of all person objects that are a part of Network
    patient_zero : Person
        The first person to introduce disease into the Network
    day_no : int
        Keeps track of the current day Disease_transmission is run on
    days : list
        list of graphs generated for all days Disease_transmission is run on
    p_0 : float
        Estimated Disease_transmission parameter
    infectious_rates : dict
        Dict containing Disease_states as keys and their relative infectiousness (in relation to the symptomaticstate) as values
    Ias : float
        Describes the percentage of individuals having a asymptomatic disease course
    Ip : float
        Describes the percentage of individuals having a presymptomatic/symptomatic disease course
    positions : dict
        A dictionary of positions keyed by node

    Methods
    -------
    set_day(int)
        Sets the disease_transmissions object day_no to a specific day
    get_day()
        Returns a int representing the day Disease_transmission is currently on
    get_all_states(Disease_state)
        Returns a list of all Person objects' current Disease_states
    update_ias_ip(Ias)
        Updates the attribute of Disease_transmission to the given float Ias
    set_patient_zero(person: Person)
        Sets a specific Person object to be patient_zero
    generate_patient_zero(num=1, sympt=True)
        Sets num random Person object in the network to be infected
    get_patient_zero()
        Returns the Person object that is patient_zero
    get_state(state: Disease_states)
        Returns a list of all Person objects in the Network with the given Disease_state, state
    get_all_person_has_infected(p: Person)
        Returns a list of all the Person objects Person p has infected
    infection_spread()
        Simulate disease transmission for one day
    run_transmission(days: int, plot=True, Ias=0.4, testing=False, save_to_file=False, sympt=True, R_null=False)
        Simulate and draw the disease transmission for multiple days
    run_transmission_empiric(days: int, graph1=None, graph2=None, day1=True, switch=False, plot=True, Ias=0.4, testing=False,save_to_file=False)
        Simulate and draw the disease transmission for days on the empiric network
    plot(interval=1, block=False)
        Plots the Disease_states of each node at a given day
    diff(state_dict: dict)
        Returns a dict of how many individuals that are not susceptible in the model for the days it is run
    R_null(days=5)
        Is generated after run_transmission is run. Gives an estimate on R_0 based on the first days (5) infected individuals
    average_infected_on_day(day: int)
        Calculates how many are on average infected on a certain day
    average_recovered_infected()
        How many on average all individuals have infected before they recovered. Is accumulating
    plot_exposed(filename: str)
        Plots the number of exposed individuals as a function of the day
    plot_recovered(filename: str, show=True, lab=None, colour="grey")
        Plots the number of recovered individuals as a function of the day
    plot_R0(filename)
        Plots the number of R0 as a function of the day
    plot_r0_and_day(data: pd.DataFrame, plot=True)
        Scatterplot of R0 for the days. Not currently in use
    extract_r0_day()
        Extracts the number of infected, R0, symptomatic and count. Not currently in use
    run_transmission_with_R0_plots(iterations: int, num_days: int, save=True)
        Plots the average of numberous iterations for num_days days. Not currently in use
    isolate(graph: nx.Graph)
        Isolates all individuals in a graph that has the disease_state Infected_symptomatic
    generate_green_stoplight(graph: nx.Graph)
        The level in the primary school is set to green
    generate_yellow_stoplight(graph: nx.Graph)
        The level in the primary school is set to yellow
    generate_red_stoplight(graph: nx.Graph)
        The level in the primary school is set to red
    generate_cohorts()
        Generates cohorts for the Person objects present in the students list
    weekly_testing(recurr=7)
        Weekly testing sets a tested state to True in all Person objects presents, and registreres every IS, IP and IAS.
    asymptomatic_calibration()
        Investigates the effect of changing asymptomatic vs symptomatic percentage
    sympt_asympt_R_0(iterations=100, sympt=True)
        Stores how many individuals patient_zero infect and takes the average of 100 iterations
    """

    def __init__(self, network, stoplight=None, Ias=0.4):
        """Inits Disease_transmission object with a Network object

        Parameters
        ----------
        network: Network
            Network object containing interactions between Person objects
        stoplight : None or Traffic_light
            Contains a Traffic_light entry from the traffic light model (green ("G"), yellow ("Y"), red("R")). Default is None
        Ias : float
            Describes the percentage of individuals having a asymptomatic disease course. Default is 0.4
        """

        self.network = network
        self.stoplight = stoplight

        self.graph = network.get_graph()
        self.students = network.get_students()
        self.patient_zero = None

        self.day_no = 0
        self.days = [self.graph]

        self.p_0 = 0.001
        self.infectious_rates = {Disease_states.IAS: 0.1, Disease_states.IP: 1.3, Disease_states.IS: 1}  # FHI
        self.Ias = Ias
        self.Ip = 1 - self.Ias

        self.positions = nx.spring_layout(self.graph, seed=10396953)

    def set_day(self, day: int) -> None:
        """Sets the day disease transmission is run on. Is used to reset when running iterations"""
        self.day_no = day

    def get_day(self) -> int:
        """Returns the current day disease transmission is run on"""
        return self.day_n0

    def get_all_states(self) -> list:
        """Returns a list of all Person objects that have a given state"""
        state_list = []
        for student in self.students:
            state_list.append(student.get_state())
        return state_list

    def update_ias_ip(self, Ias: float) -> None:
        """Updates the attribute asymptomatic proportion, Ias, for the Disease_transmission object

        Parameters
        ----------
        Ias : float
            The percentage of individuals having a asymptomatic disease course
        """

        self.Ias = Ias
        self.Ip = 1 - self.Ias

    def set_patient_zero(self, person: Person) -> None:
        """Sets a specific Person object to be patient_zero

        Parameters
        ----------
        person : Person
            The Person object that is going to be set as patient_zero
        """
        self.patient_zero = person
        self.patient_zero.set_state(Helpers.get_infection_root(pA=self.Ias, pP=self.Ip))
        self.patient_zero.set_day_infected(0)
        self.patient_zero.set_infected_by(None)

    def generate_patient_zero(self, num=1, sympt=True) -> None:
        """Sets num random Person object in the network to be infected

        Default is to start with one infected individual with num=1.

        Parameters
        ----------
        num : int
            The number of patient_zeros that should be introduced into the model at a given time. Default is 1
        sympt : bool
            Denotes whether or not patient_zero will have a symptomatic or asymptomatic Disease_state.
            Default is True, meaning that patient_zero is set to a symptomatic disease course
        """
        for _ in range(num):
            self.patient_zero = random.choice(self.students)
            if sympt:
                self.patient_zero.set_state(Disease_states.IP)
            else:
                self.patient_zero.set_state(Disease_states.IAS)
            # else:
            # self.patient_zero.set_state(Helpers.get_infection_root(pA=self.Ias, pP=self.Ip))
            self.patient_zero.set_day_infected(0)
            self.patient_zero.set_infected_by(
                None
            )  # Since it is the first infected individual, it is infected from an outside source

    def get_patient_zero(self) -> Person:
        """Returns the Person object that is patient_zero"""
        return self.patient_zero

    def get_state(self, state: Disease_states) -> list:
        """Returns a list of all Person objects in the Network with the given Disease_state, state

        Parameters
        ----------
        state : Disease_state
            The state in which you would like to extract a list of individuals in
        """
        state = []
        for student in self.students:
            if student.get_state() == state:
                state.append(student)
        return state

    def get_all_person_has_infected(self, p: Person) -> list:
        """Returns a list of all the Person objects Person p has infected

        Parameters
        ----------
        p : Person
            The Person object in which you would like to get a list of who they have infected
        """
        return list(filter(lambda x: x.get_infected_by() == p, self.students))

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
                        p_inf = 1 - (1 - self.p_0) ** (
                            self.graph.edges[(stud, n)]["count"] * self.infectious_rates[stud.state]
                        )
                        rand = random.random()
                        infected = rand < p_inf
                        if infected and n.get_state() == Disease_states.S:

                            n.set_state(Disease_states.E)
                            n.set_day_infected(self.day_no)
                            n.set_infected_by(stud)

    def run_transmission(
        self,
        days: int,
        plot=True,
        Ias=0.4,
        testing=False,
        save_to_file=None,
        sympt=True,
        R_null=False,
        recovered_R0=False,
        test_every=7,
    ) -> dict:
        """Simulate and draw the disease transmission for multiple days

        Generates a plot for transmission of each day in days. A new network
        is generated each day with differing interactions from previous days.

        Parameters
        ----------
        days : int
            The number of days the transmission should be simulated
        plot : bool
            Determines if transmission should be plotted. If True, which is the default, the transmission is plotted day by day.
        Ias : float
            Describes the percentage of asymptomatic individuals for the given day run_transmission is run. Default is 0.4
        testing : bool
            Determined wheter or not weekly testing is implemented. By default it is False, meaning no testing is implemented
        dave_to_file : bool
            Determines if the disease transmission should be saved to a file or not. Default is False, meaning the data is not saved to file
        sympt : bool
            Determines whether patient_zero is asymptomatic or presymptomatic. Default is True, meaning patient_zero is set to a symptomatic disease course
        R_null : bool
            Determines the output of the model. If True, the model only returns the R_0 of patient_zero. If False, it returns a dict of the number of individuals in different Disease_states
        """
        self.network.reset_student_disease_states()  # Reset the states of the Person objects
        self.set_day(0)  # Resets the day

        self.update_ias_ip(Ias)  # makes sure Ip and Ias are set.

        days_dic = {}  # Keeps track of how many individuals are in the given state at time key

        for i in range(days):  # Meaning: 0-day-1
            if testing:
                self.weekly_testing(test_every)
            d = dict([(e, 0) for e in Disease_states])  # Keeps track of how many individuals are in a given state
            if i == 0:  # Day 0, no disease_transmission. Patient zero is introduced
                self.generate_patient_zero(sympt=sympt)  # Set patient to be symptomatic
            else:
                if self.stoplight == Traffic_light.G:
                    self.graph = self.generate_green_stoplight(self.graph)
                elif self.stoplight == Traffic_light.Y:
                    self.graph = self.generate_yellow_stoplight(self.graph)
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

                # Sets the key (represents day i) to have the value: dict over states and people in that state
            if plot:
                if i == days - 1:
                    # On the last day, the plot stays untill being dismissed
                    self.plot(block=True)
                else:
                    self.plot()
            ######## Prints out how many individuals are in a given disease state for each day########
            # print(f"-------------Day {i}-------------")
            # pprint(d)
            # print(f"New: {self.average_infected_on_day(i)}\nNew_new: {self.average_recovered_infected()}")
            d["R_null"] = self.average_recovered_infected()
            days_dic[i] = d
            self.day_no += 1

        if save_to_file:  # The amount of individuals on certain days and R0 is saved to file

            with open(f"./data/weekly_testing//{save_to_file}transmission.csv", "w") as f:
                f.write(
                    "Day,Suceptible,Exposed,Infected_asymptomatic,Infected_presymptomatic,Infected_symptomatic,Recovered,Hospitalized,Death,R_null\n"
                )
                for key, val in days_dic.items():
                    s, e, ia, ip, Is, r, h, death, r_null = (
                        val[Disease_states.S],
                        val[Disease_states.E],
                        val[Disease_states.IAS],
                        val[Disease_states.IP],
                        val[Disease_states.IS],
                        val[Disease_states.R],
                        val[Disease_states.H],
                        val[Disease_states.D],
                        val["R_null"],
                    )
                    f.write(f"{key},{s},{e},{ia},{ip},{Is},{r},{h},{death},{r_null}\n")
        if R_null:
            i = 0
            for stud in self.students:
                if stud.get_infected_by() == self.patient_zero:
                    i += 1

            return days_dic, i
        if recovered_R0:
            return days_dic, self.average_recovered_infected(return_list=True)
        return days_dic  # return diff(days_dict)

    def run_transmission_empiric(
        self,
        days: int,
        graph1=None,
        graph2=None,
        day1=True,
        switch=False,
        plot=True,
        Ias=0.4,
        testing=False,
        save_to_file=False,
    ):
        """Simulate and draw the disease transmission for days on the empiric network

        Generates a plot for transmission of each day in days. The empiric network is used as a graph.
        Whether or not one switches between the two empiric days or not is denotes by the switch parameter.
        Traffic_light conditions are not implemented.

        Parameters
        ----------
        days : int
            The number of days the transmission should be simulated
        graph1 : nx.Graph
            The empiric graph for interactions on day 1
        graph2 : nx.Graph
            The empiric graph for interactions on day 1
        day1 : bool
            Determines if transmission should begin on day 1. If True, which is the default, transmission begins on day 1.
        switch : bool
            Determines if transmission is run on only one of the two days or not. If switch is False, transmission is only run on one network for all the days
            If True, transmission is run on alternating graphs for day 1 and day 2.
        plot : bool
            Determines if transmission should be plotted. If True, which is the default, the transmission is plotted day by day.
        Ias : float
            Describes the percentage of asymptomatic individuals for the given day run_transmission is run. Default is 0.4
        testing : bool
            Determined wheter or not weekly testing is implemented. By default it is False, meaning no testing is implemented
        dave_to_file : bool
            Determines if the disease transmission should be saved to a file or not. Default is False, meaning the data is not saved to file
        sympt : bool
            Determines whether patient_zero is asymptomatic or presymptomatic. Default is True, meaning patient_zero is set to a symptomatic disease course
        """
        self.update_ias_ip(Ias)  # makes sure Ip and Ias are set.
        self.network.reset_student_disease_states()

        days_dic = {}  # Keeps track of how many individuals are in the given state at time key

        if day1 == True:
            self.graph = graph1.get_graph()
        else:
            self.graph = graph2.get_graph()

        for i in range(days):  # Meaning: 0-day-1
            if testing:
                self.weekly_testing()
            d = dict([(e, 0) for e in Disease_states])  # Keeps track of how many individuals are in a given state
            if i == 0:  # Day 0, no disease_transmission.
                self.generate_patient_zero()  # Patient zero is introduced

            else:
                if switch:
                    if self.graph == graph1.get_graph():
                        self.graph = graph2.get_graph()
                        self.students = graph2.get_students()
                    else:
                        self.graph = graph1.get_graph()
                        self.students = graph1.get_students()

                self.days.append(self.graph)

                self.infection_spread()

            for stud in self.students:
                # Update state if conditions are fullfilled (x amount of days in state y)
                stud.add_day_in_state(self.Ias, self.Ip)
                # Update the amount of days an individual has been in state
                d[stud.state] += 1

            print(f"-------------Day {i}-------------")
            pprint(d)
            print(
                f"Average infected today: {self.average_infected_on_day(i)}\nAverage infected of all recovered: {self.average_recovered_infected()}"
            )
            d["R_null"] = self.average_recovered_infected()
            days_dic[i] = d
            self.day_no += 1

        if save_to_file:
            with open(f"./data/empiric_vs_model/day1{day1}switch{switch}.csv", "w") as f:
                f.write(
                    "Day,Suceptible,Exposed,Infected_asymptomatic,Infected_presymptomatic,Infected_symptomatic,Recovered,Hospitalized,Death,R_null\n"
                )
                for key, val in days_dic.items():
                    s, e, ia, ip, Is, r, h, death, r_null = (
                        val[Disease_states.S],
                        val[Disease_states.E],
                        val[Disease_states.IAS],
                        val[Disease_states.IP],
                        val[Disease_states.IS],
                        val[Disease_states.R],
                        val[Disease_states.H],
                        val[Disease_states.D],
                        val["R_null"],
                    )
                    f.write(f"{key},{s},{e},{ia},{ip},{Is},{r},{h},{death},{r_null}\n")

            # Sets the key (represents day i) to have the value: dict over states and people in that state
            if plot:
                if i == days:
                    # On the last day, the plot stays untill being dismissed
                    self.plot(block=True)
                else:
                    self.plot()

        return days_dic

    def plot(self, interval=1, block=False) -> None:
        """Plots the Disease_states of each node at a given day

        Uses colors to represent the different Disease_states.
        If the plot is of the last day, the plot stays open and
        doeas not close after interval amount of seconds.
        Default values are interval=1 and block=False.

        Parameters
        ----------
        interval int:
            How long of a time, in seconds, a plot should be shown. Default is 1 seconds
        block bool:
            Keeps track of whether or not the plot is the last day defined. Default is False
        """

        plt.clf()
        G = self.network.get_graph()
        sizes = [100 for _ in range(len(G.nodes))]
        weights = [None for _ in range(len(G.edges))]
        edge_color = ["grey" for _ in range(len(G.nodes))]

        for i, stud in enumerate(G.nodes):
            if stud.state == Disease_states.E:
                edge_color[i] = "khaki"
            elif stud.state in [Disease_states.IP, Disease_states.IS]:
                edge_color[i] = "indianred"
            elif stud.state == Disease_states.IAS:
                edge_color[i] = "cadetblue"
            elif stud.state == Disease_states.R:
                edge_color[i] = "darkseagreen"
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
            plt.savefig("./fig_master/transmission.png", transparent=True, dpi=500)
            plt.show(block=True)

        else:
            plt.pause(interval)

    def diff(self, state_dict: dict) -> dict:
        """Returns a dict of how many individuals that are not susceptible in the model for the days it is run

        Parameters
        ----------
        state_dict : dict
            Dict containing Disease_states as keys and number of individuals in that given state as the value
        """

        for entry in state_dict:
            state_dict.update({entry: 236 - state_dict[entry][Disease_states.S]})
        return state_dict

    def R_null(self, days=5) -> float:
        """Is generated after run_transmission is run. Gives an estimate on R_0 based on the first days (5) infected individuals

        Counts how many Person objects the first days (5) infected infect before they recover. Draws an average of the days (5). Default is 5 days.

        Parameters
        ----------
        days : int
            Determines how many infected during the first days days, and furthermore how many they infect. Default is 5,
            meaning that the R0 is calculated by taking the average of how many the infected individuals on day 5 infect before they recover
        """
        infected_dict = dict([(stud.get_ID(), 0) for stud in self.students])

        for stud in filter(lambda x: x.get_state() != Disease_states.S, self.students):
            key = stud.get_infected_by()
            if key is None:
                infected_dict[None] = 1
            else:
                infected_dict[stud.get_infected_by().get_ID()] = (
                    infected_dict.get(stud.get_infected_by().get_ID(), 0) + 1
                )
        infected_day_5_or_less = list(
            filter(lambda x: x.get_day_infected() is not None and x.get_day_infected() <= days, self.students)
        )

        summ = 0
        for stud in infected_day_5_or_less:
            if stud is not None:
                summ += infected_dict[stud.get_ID()]

        print(f"sum: {summ}")
        print(f"Infected day 5 or less: {len(infected_day_5_or_less)}")
        if len(infected_day_5_or_less) == 1:
            return 0.0
        return summ / (len(infected_day_5_or_less) - 1)

    def average_infected_on_day(self, day: int) -> float:
        """Calculates how many are on average infected on a certain day

        Parameters
        ----------
        day : int
            The day in which one would like to calculate how many on average are infected
        """
        infected_dict = {}  # dict([(stud.get_ID(), 0) for stud in self.students])

        for stud in filter(lambda x: x.get_day_infected() == day, self.students):
            key = stud.get_infected_by()
            if key == -1 or key is None:
                continue
            infected_dict[stud.get_infected_by().get_ID()] = infected_dict.get(stud.get_infected_by().get_ID(), 0) + 1
        if not len(infected_dict):
            return 0.0
        return sum(infected_dict.values()) / len(infected_dict)

    def average_recovered_infected(self, return_list=False) -> float:
        """How many on average all individuals have infected before they recovered. Is accumulating"""
        recovered = list(filter(lambda x: x.get_state() == Disease_states.R, self.students))
        if not len(recovered):
            return 0

        infected_dict = {key: 0 for key in recovered}
        for stud in filter(lambda x: x.get_state() != Disease_states.S, self.students):
            key = stud.get_infected_by()
            if key is None:
                continue
            if key in recovered:
                infected_dict[key] = infected_dict.get(key, 0) + 1
        if return_list:
            print(infected_dict)
            return infected_dict
        if not len(infected_dict):
            return 0.0
        return sum(infected_dict.values()) / len(infected_dict)

    def plot_exposed(self, filename: str) -> None:
        """Plots the number of exposed individuals as a function of the day

        Parameters
        ----------
        filename : str
            String with filename and path
        """
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        df = pd.read_csv(filename)

        x = df["Day"]
        y = df["Exposed"]

        plt.scatter(x, y)

        plt.xlabel("Day")
        plt.ylabel("Exposed")
        plt.show()

    def plot_recovered(self, filename: str, show=True, lab=None, colour="grey", alpha=1, tested=False):
        """Plots the number of recovered individuals as a function of the day

        Parameters
        ----------
        filename : str
            String with filename and path
        show : bool
            Whether or not to plt.show() the plot. Default is True
        lab : str
            The label of the data plotted. Default is None
        colour : str
            Matplotlib color for the data plotted. Default is "grey"
        """

        plt.rcParams["figure.figsize"] = [7.50, 4.5]
        plt.rcParams["figure.autolayout"] = True

        df = pd.read_csv(filename)

        x = df["Day"]
        y = df["Recovered"]

        if lab == "Average":
            plt.plot(x, y, label=lab, color=colour, alpha=alpha, linewidth=2)  # s=30
        else:
            plt.plot(x, y, "-", label=lab, color=colour, alpha=alpha, linewidth=3)  # s=30

        plt.xlabel("Day")
        plt.ylabel("Recovered")

        if show:
            if tested:
                for i in range(7, 101, 7):
                    if i > 97:
                        plt.vlines(
                            i, ymin=0, ymax=65, colors="grey", linestyles="dashed", label="Days tested", alpha=0.7
                        )
                    else:
                        plt.vlines(i, ymin=0, ymax=65, colors="grey", linestyles="dashed", alpha=0.7)

            plt.ylabel("#Recovered", fontsize=14)
            plt.xlabel("Day", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend()
            plt.tight_layout()
            # plt.savefig("./fig_master/red_light_all.png", transparent=True, dpi=500)
            plt.show()

    def plot_R0(self, filename):
        """Plots the number of R0 as a function of the day

        Parameters
        ----------
        filename : str
            String with filename and path
        """
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        df = pd.read_csv(filename)

        x = df["Day"]
        y = df["R_null"]

        plt.scatter(x, y)

        plt.xlabel("Day")
        plt.ylabel("R0")
        # plt.gcf().autofmt_xdate()

        plt.show()

    def plot_r0_and_day(self, data: pd.DataFrame, plot=True) -> None:
        """Scatterplot of R0 for the days. Not currently in use"""
        b = sns.scatterplot(
            data=data,
            x="day_infected",
            y="r0",
            size="count",
            hue="sympt",
            sizes=(100, 200),
            alpha=0.5,
            legend=False,
            palette=["darkgreen", "rebeccapurple"],
        )
        b.set_yticklabels(b.get_yticks(), size=12)
        b.set_xticklabels(b.get_yticks(), size=12)
        plt.tight_layout()
        b.set(xlabel=[])
        b.set(ylabel=[])
        b.set_xlabel("Day infected", fontsize=12)
        b.set_ylabel("$R_{0}$", fontsize=12)

        if plot:
            plt.show()
        else:
            plt.savefig("./fig_master/R0_and_day.png", transparent=True, dpi=500)

    def extract_r0_day(self) -> pd.DataFrame:
        """Extracts the number of infected, R0, symptomatic and count."""
        d = {}
        l = []
        for stud in filter(lambda x: x.state == Disease_states.R, self.students):
            sympt = stud.is_symptomatic()
            day_infected = stud.get_day_infected()

            r0 = len(self.get_all_person_has_infected(stud))
            d[(day_infected, r0, sympt)] = d.get((day_infected, r0, sympt), 0) + 1

        for (day_infected, r0, sympt), count in d.items():
            l.append((day_infected, r0, sympt, count))

        return pd.DataFrame(data=l, columns=["day_infected", "r0", "sympt", "count"])

    def run_transmission_with_R0_plots(self, iterations: int, num_days: int, save=True) -> None:
        """Plots the average of numerous iterations for num_days days.

        Parameters
        ----------
        iterations : int
            the number of iterations transmission should be run for
        num_days : int
            The number of days each iteration shoulc be run for
        save : bool
            Whether or not to save or plot the results

        """
        df1 = pd.DataFrame(data=[], columns=["day_infected", "r0", "sympt", "count"])
        for _ in range(iterations):
            self.run_transmission(num_days, plot=False, save_to_file=True)
            df = self.extract_r0_day()
            df1 = (
                pd.concat([df1, df])
                .groupby(["day_infected", "r0", "sympt"], as_index=False)["count"]
                .sum()
                .reset_index()
            )

        self.plot_r0_and_day(df1, plot=not save)

    def isolate(self, graph: nx.Graph) -> None:
        """Isolates all individuals in a graph that has the disease_state Infected_symptomatic

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph that contains all interactions between the Person objects
        """

        if self.stoplight in [Traffic_light.G, Traffic_light.Y, Traffic_light.R]:
            for stud in self.students:
                if stud.state == Disease_states.IS:
                    self.network.remove_all_interactions(graph, stud)

    def generate_green_stoplight(self, graph: nx.Graph) -> nx.Graph:
        """The level in the primary school is set to green.

        In effect this means decreasing the interactions slightly, and isolating sick individuals

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph that contains all interactions between the Person objects
        """

        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_yellow_stoplight(self, graph: nx.Graph) -> nx.Graph:
        """The level in the primary school is set to yellow.

        In effect this means decreasing the interactions more, isolating sick individuals and dividing into cohorts

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph that contains all interactions between the Person objects
        """

        self.generate_cohorts()
        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_red_stoplight(self, graph: nx.Graph) -> nx.Graph:
        """The level in the primary school is set to red.

        In effect this means decreasing the interactions more, isolating sick individuals and
        dividing each class into two cohorts

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph that contains all interactions between the Person objects
        """
        self.generate_cohorts()
        graph = self.network.decrease_interaction_day(self.stoplight)
        self.isolate(graph)
        return graph

    def generate_cohorts(self) -> None:
        """Generates cohorts for the Person objects present in the students list

        If Traffic_light is yellow, one class is divided into one cohort.
        If the level is red, one class is divided into two cohorts
        """

        grades = self.network.get_available_grades()
        classes = self.network.get_available_classes()
        grade_and_classes = [f"{i}{j}" for i in grades for j in classes]
        cohorts = []

        if self.stoplight == Traffic_light.Y:
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
        """Weekly testing sets a tested state to True in all Person objects presents, and registreres every IS, IP and IAS.

        Has a 80% chance of picking up disease

        Parameters
        ----------
        recurr : int
            Determines the interval for how often testing should be performed. Default is every week, meaning every 7 days.
        """

        if self.day_no % recurr == 0 and self.day_no != 0:
            for stud in self.students:
                if stud.get_state() in [Disease_states.IAS, Disease_states.IS, Disease_states.IP]:
                    r = random.random()
                    if r < 0.6:
                        self.network.remove_all_interactions(self.graph, stud)
                        stud.set_tested(True)

    def asymptomatic_calibration(self) -> None:
        """Investigates the effect of changing asymptomatic vs symptomatic percentage"""
        Ias_dict = {}
        for Is in range(0, 101, 10):
            Ias = (100 - Is) / 100
            Is = Is / 100

            day_list = np.zeros(35)  # 15
            for _ in range(0, 20):  # 20
                for key, val in self.run_transmission(35, plot=False, Ias=Ias).items():  # 14
                    day_list[key] += val

                self.network.reset_student_disease_states()
            day_list = day_list / 20
            Ias_dict[Ias] = day_list.tolist()

        with open("asymptomatic_calibration.pickle", "wb") as handle:
            pickle.dump(Ias_dict, handle)

    def sympt_asympt_R_0(self, iterations=100, sympt=True):
        """Stores how many individuals patient_zero infect and takes the average of 100 iterations

        Parameters
        ----------
        iterations : int
            Determines how many times disease transmission should be run for the 12 days. Default is 100
        sympt : bool
            Determines if patient_zero is symptomatic or asymptomatic. True meaning, symptomatic patient_zero is default
        """

        R_null_list = []
        for _ in range(1, iterations + 1):

            _, R_null = self.run_transmission(days=12, plot=False, sympt=sympt, R_null=True)

            R_null_list.append(R_null)
        print(R_null_list)
        df = pd.DataFrame(R_null_list)
        df.to_csv(f"./asymptomatic_symptomatic/sympt:{sympt}.csv")

    def traffic_light_transmission(self, iterations=3, days=100):
        d = {}
        for e in Traffic_light:
            d[e] = {}
            for i in range(days):
                d[e][i] = {}
        R_null_dict = dict([(stoplight, []) for stoplight in Traffic_light])
        for i in range(1, iterations + 1):
            for stoplight in [Traffic_light.G, Traffic_light.Y, Traffic_light.R]:
                self.stoplight = stoplight
                dic, people_infected_by_p0 = self.run_transmission(
                    days=days, save_to_file=str(stoplight) + str(i), plot=False, R_null=True
                )
                for day in range(days):
                    for disease_key in [e for e in Disease_states] + ["R_null"]:
                        d[stoplight][day][disease_key] = d[stoplight][day].get(disease_key, 0) + dic[day][disease_key]
                    R_null_dict[stoplight].append(people_infected_by_p0)

        green_averages = self.calculate_averages(d[Traffic_light.G], iterations)
        yellow_averages = self.calculate_averages(d[Traffic_light.Y], iterations)
        red_averages = self.calculate_averages(d[Traffic_light.R], iterations)

        self.save_to_file(green_averages, "Traffic_light.G_average.csv")
        self.save_to_file(yellow_averages, "Traffic_light.Y_average.csv")
        self.save_to_file(red_averages, "Traffic_light.R_average.csv")

        print(R_null_dict)
        for key, val in R_null_dict.items():
            R_null_list = val
            df = pd.DataFrame(R_null_list)
            df.to_csv(f"./data/traffic_light/{key}_infection_by_p0.csv")

    def traffic_light_plots(self):
        self.plot_recovered(
            "./data/traffic_light/Traffic_light.G_average.csv",
            show=False,
            lab="Green",
            colour="darkolivegreen",
            alpha=0.8,
        )
        self.plot_recovered(
            "./data/traffic_light/Traffic_light.Y_average.csv", show=False, lab="Yellow", colour="gold", alpha=0.8
        )
        self.plot_recovered(
            "./data/traffic_light/Traffic_light.R_average.csv", show=True, lab="Red", colour="indianred", alpha=0.8
        )

    def save_to_file(self, d, filename):
        """Saves a disease states for each day into a csv.

        Parameters
        ----------
        d : dict
            Dict containing the day as key and the number of individuals in each disease state as value
        filename : str
            The filename in which to save the dict as
        """
        with open(f"./data/weekly_testing2/{filename}", "w") as f:
            f.write(
                "Day,Suceptible,Exposed,Infected_asymptomatic,Infected_presymptomatic,Infected_symptomatic,Recovered,Hospitalized,Death,R_null\n"
            )
            for key, val in d.items():
                s, e, ia, ip, Is, r, h, death, r_null = (
                    val[Disease_states.S],
                    val[Disease_states.E],
                    val[Disease_states.IAS],
                    val[Disease_states.IP],
                    val[Disease_states.IS],
                    val[Disease_states.R],
                    val[Disease_states.H],
                    val[Disease_states.D],
                    val["R_null"],
                )
                f.write(f"{key},{s},{e},{ia},{ip},{Is},{r},{h},{death},{r_null}\n")

    def calculate_averages(self, d, iterations):
        dic = {}

        for day, states_dic in d.items():
            dic[day] = {
                Disease_states.S: states_dic[Disease_states.S] / iterations,
                Disease_states.E: states_dic[Disease_states.E] / iterations,
                Disease_states.IAS: states_dic[Disease_states.IAS] / iterations,
                Disease_states.IP: states_dic[Disease_states.IP] / iterations,
                Disease_states.IS: states_dic[Disease_states.IS] / iterations,
                Disease_states.R: states_dic[Disease_states.R] / iterations,
                Disease_states.H: states_dic[Disease_states.H] / iterations,
                Disease_states.D: states_dic[Disease_states.D] / iterations,
                "R_null": states_dic["R_null"] / iterations,
            }

        return dic

    def weekly_testing_transmission(self, iterations=1, days=100, ID=0):
        d = {}
        R_null_dict = {}

        for test in ["tested_weekly", "tested_biweekly", "not_tested"]:
            d[test] = {}
            R_null_dict[test] = {}
            for i in range(days):
                d[test][i] = {}
                R_null_dict[test][i] = {}

        for i in range(1, iterations + 1):
            for test in d.keys():
                print(test)
                if test == "tested_weekly":
                    dic, recovered_R0 = self.run_transmission(
                        days=days,
                        save_to_file=str(test) + str(i)#+ str(ID),
                        plot=False,
                        testing=True,
                        recovered_R0=True,
                        test_every=7,
                    )
                if test == "tested_biweekly": #str(ID),
                    dic, recovered_R0 = self.run_transmission(days=days,save_to_file=(str(test) + str(ID)),plot=False, testing=True,recovered_R0=True, test_every=14)
                else:
                    dic, recovered_R0 = self.run_transmission(
                        days=days, save_to_file=str(test) + str(i), plot=False, testing=False, recovered_R0=True
                    )

                R_null_dict[test][i] = recovered_R0

                for day in range(days):
                    for disease_key in [e for e in Disease_states] + ["R_null"]:

                        d[test][day][disease_key] = d[test][day].get(disease_key, 0) + dic[day][disease_key]

        tested7_average = self.calculate_averages(d["tested_weekly"], iterations)
        tested14_average = self.calculate_averages(d["tested_biweekly"], iterations)
        not_tested_average = self.calculate_averages(d["not_tested"], iterations)

        self.save_to_file(tested7_average, "tested_weekly_average.csv")
        self.save_to_file(tested14_average, "tested_biweekly_average.csv")
        self.save_to_file(not_tested_average, "not_tested_average.csv")

        total_R0 = {}
        for tested, dict_of_iterations in R_null_dict.items():
            l = []
            for itera in range(1, iterations + 1):
                new_dict = R_null_dict[tested][itera]
                for ID, R_0 in new_dict.items():
                    l.append(R_0)
            total_R0[tested] = l.copy()

        for key, val in total_R0.items():
            R_null_list = val
            df = pd.DataFrame(R_null_list)
            df.to_csv(f"./data/weekly_testing2/{key}_infection_by_p0.csv")

    def plot_all_recovered(self, filename, testing_type=None):  # red: indi

        if testing_type == "G":
            col = "darkolivegreen"
        elif testing_type == "Y":
            col = "goldenrod"
        elif testing_type == "R":
            col = "indianred"
        else:
            col = "grey"
        for i in range(1, 11):
            self.plot_recovered(
                f"{filename}/{testing_type}{i}transmission.csv",
                show=False,
                lab=f"{i}",
                colour=col,
                alpha=i / 10,
            )

        self.plot_recovered(
            f"{filename}/{testing_type}_average.csv",
            show=True,
            lab=f"Average",
            colour=col,
        )


if __name__ == "__main__":
    network = Network(num_students=222, num_grades=5, num_classes=2, class_treshold=23)

    disease_transmission = Disease_transmission(network)
    # disease_transmission.plot_all_recovered(filename="./data/weekly_testing", testing_type="tested_biweekly")
    
    #ID = sys.argv[1]

    disease_transmission.weekly_testing_transmission(100, 100) #, ID=ID)

    # Traffic light
    # disease_transmission.plot_recovered(
    #     "./data/traffic_light/Traffic_light.G_average.csv", show=False, lab="Green", colour="darkseagreen"
    # )
    # disease_transmission.plot_recovered(
    #     "./data/traffic_light/Traffic_light.Y_average.csv",
    #     show=False,
    #     lab="Yellow",
    #     colour="gold",
    # )
    # disease_transmission.plot_recovered(
    #     "./data/traffic_light/Traffic_light.R_average.csv",
    #     show=True,
    #     lab="Red",
    #     colour="indianred",
    # )
# empiric vs model transmission
# disease_transmission.plot_recovered(
#     "./data/empiric_vs_model/day1FalseSwitchFalse_average.csv", show=False, lab="Day two", colour="khaki"
# )
# disease_transmission.plot_recovered(
#     "./data/empiric_vs_model/day1TrueSwitchFalse_average.csv", show=False, lab="Day one", colour="darkgoldenrod"
# )
# disease_transmission.plot_recovered(
#     "./data/empiric_vs_model/day1TrueSwitchTrue_average.csv", show=False, lab="Switch", colour="cadetblue"
# )

# disease_transmission.plot_recovered(
#     "./data/empiric_vs_model/model_average.csv", show=True, lab="Model", colour="rosybrown"
# )

# testing
# disease_transmission.plot_recovered(
#     "./data/weekly_testing/not_tested_average.csv", show=False, lab="Not tested", colour="rosybrown"
# )
# disease_transmission.plot_recovered(
#     "./data/weekly_testing/tested_weekly_average.csv",
#     show=False,
#     lab="Weekly tested",
#     colour="darkseagreen",
#     tested=True,
# )
# disease_transmission.plot_recovered(
#     "./data/weekly_testing/tested_biweekly_average.csv",
#     show=True,
#     lab="Biweekly tested",
#     colour="darkgoldenrod",
#     tested=True,
# )
