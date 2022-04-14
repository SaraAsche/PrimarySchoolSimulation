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

import re
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
import os
import seaborn as sns


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
        self.p_0 = 0.001

        self.patient_zero = None
        self.infectious_rates = {Disease_states.IAS: 0.1, Disease_states.IP: 1.3, Disease_states.IS: 1}  # FHI
        # self.infectious_rates = {Disease_states.IAS: 0.06, Disease_states.IP: 0.45, Disease_states.IS: 0.4}
        self.Ias = Ias
        self.Ip = 1 - self.Ias
        self.positions = nx.spring_layout(self.graph, seed=10396953)
        self.stoplight = stoplight
        self.days = [self.graph]

    def update_ias_ip(self, Ias) -> None:
        self.Ias = Ias
        self.Ip = 1 - self.Ias

    def get_day(self) -> int:
        return self.day_n0

    def set_day(self, day) -> None:
        self.day_no = day

    def get_all_states(self) -> list:
        state_list = []
        for student in self.students:
            state_list.append(student.get_state())
        return state_list

    def generate_patient_zero(self, num=1, sympt=True) -> None:
        """Generates num students in the network that are infected

        Default is to start with one infected individual with num=1.

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
            self.patient_zero.set_infected_by(None)

    def set_patient_zero(self, person) -> None:
        """Sets a specific Person object to be patient zero"""
        self.patient_zero = person
        self.patient_zero.set_state(Helpers.get_infection_root(pA=self.Ias, pP=self.Ip))
        self.patient_zero.set_day_infected(0)
        self.patient_zero.set_infected_by(None)

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

    def run_transmission(
        self, days, plot=True, Ias=0.4, testing=False, save_to_file=False, sympt=True, R_null=False
    ) -> dict:
        """Simulate and draw the disease transmission for multiple days

        Generates a plot for transmission of each day in days. A new network
        is generated each day with differing interactions from previous days.

        Parameters
        ----------
        days int:
            The number of days the transmission should be simulated
        """
        self.network.reset_student_disease_states()
        self.set_day(0)

        self.update_ias_ip(Ias)  # makes sure Ip and Ias are set.

        days_dic = {}  # Keeps track of how many individuals are in the given state at time key

        for i in range(days):  # Meaning: 0-day-1
            if testing:
                self.weekly_testing()
            d = dict([(e, 0) for e in Disease_states])  # Keeps track of how many individuals are in a given state
            if i == 0:  # Day 0, no disease_transmission. Patient zero is introduced
                self.generate_patient_zero(sympt=sympt)  # Set patient to be symptomatic
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

                # Sets the key (represents day i) to have the value: dict over states and people in that state
            if plot:
                if i == days:
                    # On the last day, the plot stays untill being dismissed
                    self.plot(block=True)
                else:
                    self.plot()

            print(f"-------------Day {i}-------------")
            pprint(d)
            print(f"New: {self.new_r_null(i)}\nNew_new: {self.new_new_r_null()}")

            d["R_null"] = self.new_new_r_null()
            days_dic[i] = d
            self.day_no += 1

        if save_to_file:
            dirs = os.listdir("./data")
            if not len(dirs):
                last_file = -1
            else:
                last_file = int(dirs[-1][2])

            with open(f"./data/{str(last_file + 1).zfill(3)}transmission.csv", "w") as f:
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

            return self.patient_zero, i

        return self.diff(days_dic)

    def run_transmission_empiric(
        self,
        days,
        graph1=None,
        graph2=None,
        day1=True,
        switch=False,
        plot=True,
        Ias=0.4,
        testing=False,
        save_to_file=False,
    ):
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
                # if self.stoplight == Traffic_light.G:
                #     self.graph = self.generate_green_stoplight(self.graph)
                # elif self.stoplight == Traffic_light.O:
                #     self.graph = self.generate_orange_stoplight(self.graph)
                # elif self.stoplight == Traffic_light.R:
                #     self.graph = self.generate_red_stoplight(self.graph)

                if switch:
                    if self.graph == graph1.get_graph():
                        self.graph = graph2.get_graph()
                    else:
                        self.graph = graph1.get_graph()

                self.days.append(self.graph)

                self.infection_spread()

            for stud in self.students:
                # Update state if conditions are fullfilled (x amount of days in state y)
                stud.add_day_in_state(self.Ias, self.Ip)
                # Update the amount of days an individual has been in state
                d[stud.state] += 1

            print(f"-------------Day {i}-------------")
            pprint(d)
            print(f"New: {self.new_r_null(i)}\nNew_new: {self.new_new_r_null()}")

            # r0 = self.new_new_r_null()
            # print(r0)

            days_dic[i] = d
            self.day_no += 1

            if save_to_file:
                continue

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
            for _ in range(0, 20):  # 20
                for key, val in self.run_transmission(35, plot=False, Ias=Ias).items():  # 14
                    day_list[key] += val

                self.network.reset_student_disease_states()
            day_list = day_list / 20
            Ias_dict[Ias] = day_list.tolist()

        with open("asymptomatic_calibration.pickle", "wb") as handle:
            pickle.dump(Ias_dict, handle)

    def R_null(self, days=5):
        """
        We assume self.run_transmission has aldready been run
        """
        # TODO: kjøre smitte i 30 dager, se hvor mange de som blir smittet i løpet av de 5 første dagene smitter videre. Kjøre gjennomsnitt.
        # assert len(self.days) > 0, f"Disease transmission object has only {len(self.days)} days generated"
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

    def new_r_null(self, day):
        infected_dict = {}  # dict([(stud.get_ID(), 0) for stud in self.students])

        for stud in filter(lambda x: x.get_day_infected() == day, self.students):
            key = stud.get_infected_by()
            if key == -1 or key is None:
                continue
            infected_dict[stud.get_infected_by().get_ID()] = infected_dict.get(stud.get_infected_by().get_ID(), 0) + 1
        if not len(infected_dict):
            return 0.0
        return sum(infected_dict.values()) / len(infected_dict)

    def new_new_r_null(self):
        recovered = list(filter(lambda x: x.get_state() == Disease_states.R, self.students))
        if not len(recovered):
            return 0

        infected_dict = {}
        for stud in filter(lambda x: x.get_state() != Disease_states.S, self.students):
            key = stud.get_infected_by()
            if key is None:
                continue
            if key in recovered:
                infected_dict[key] = infected_dict.get(key, 0) + 1
        if not len(infected_dict):
            return 0.0
        return sum(infected_dict.values()) / len(infected_dict)

    def plot_exposed(self, filename):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        df = pd.read_csv(filename)

        x = df["Day"]
        y = df["Exposed"]

        plt.scatter(x, y)

        plt.xlabel("Day")
        plt.ylabel("Exposed")
        # plt.gcf().autofmt_xdate()
        plt.show()

    def plot_recovered(self, filename):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        df = pd.read_csv(filename)

        x = df["Day"]
        y = df["Recovered"]

        plt.scatter(x, y)

        plt.xlabel("Day")
        plt.ylabel("Recovered")

        plt.show()
        return

    def plot_R0(self, filename):
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

    def sympt_asympt_R_0(self, iterations=100, sympt=True):

        # dict = {'First Name': 'Vikram', 'Last Name': 'Aruchamy', 'Country': 'India'}
        # df = df.append(dict, ignore_index = True)
        R_null_list = []
        for i in range(1, iterations + 1):

            pers, R_null = self.run_transmission(days=12, plot=False, sympt=sympt, R_null=True)

            R_null_list.append(R_null)
        print(R_null_list)
        df = pd.DataFrame(R_null_list)
        df.to_csv(f"./asymptomatic_symptomatic/sympt:{sympt}.csv")

    def get_all_person_has_infected(self, p: Person):
        return list(filter(lambda x: x.get_infected_by() == p, self.students))

    def extract_r0_day(self):
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

    def plot_r0_and_day(self, data, plot=True):
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

    def run_transmission_with_R0_plots(self, iterations, num_days, save=True):
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


if __name__ == "__main__":
    network = Network(num_students=236, num_grades=5, num_classes=2, class_treshold=23)

    disease_transmission = Disease_transmission(network)
    disease_transmission.run_transmission_with_R0_plots(10, 50)

    # disease_transmission.plot_exposed("./data/003transmission.csv")
    # disease_transmission.plot_recovered("./data/003transmission.csv")
    # disease_transmission.plot_R0("./data/003transmission.csv")

    # disease_transmission.sympt_asympt_R_0(iterations=100)
    # disease_transmission.sympt_asympt_R_0(iterations=100, sympt=False)
