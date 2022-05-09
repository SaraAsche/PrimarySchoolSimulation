"""Description of functions in empiric_to_person_object.py

This file contains the functionality of turning the empiric network into a Network object containing 
Person objects. Is used to compare disease transmission on the empiric network with transmission on
the model. 

Author: Sara Johanne Asche
Date: 14.02.2022
File: empiric_to_person_object.py
"""

import sys

from numpy import save
from disease_transmission import Disease_transmission
from person import Person
from interaction import Interaction
from network import Network
from enums import Disease_states
import os
import pandas as pd

# Global student list to ensure the same set of students on day 1 as on day 2 for the empiric network
students = []


def generate_network(day1=True) -> Network:
    """Reads a csv file containing interactions and outputs a Network object with generated Person objects

    Csv must be in format: time of first interaction, ID_source, ID_target, class_source, class_target, count, old_source_ID, old_target_ID

    Parameters
    ----------
    day1 : bool
        Denotes wheter a Network object based on day 1 or day 2 should be the output of the model. Default is day1,
        meaning the output Network is based on day 1
    """
    # Read dayone and daytwo csv
    dayone = pd.read_csv(
        "./Empiric_network/dayOneNewIndex.csv",
        names=["time", "stud1", "stud2", "class1", "class2", "count", "old1", "old2"],
    )
    daytwo = pd.read_csv(
        "./Empiric_network/dayTwoNewIndex.csv",
        names=["time", "stud1", "stud2", "class1", "class2", "count", "old1", "old2"],
    )

    # Extracts the IDs for students that are presend on both day 1 and day 2
    ids = set(pd.concat([dayone["stud1"], dayone["stud2"]])).intersection(
        set(pd.concat([daytwo["stud1"], daytwo["stud2"]]))
    )

    # Filters through the IDs and discard interactions of individuals that are not present on both days
    dayone = dayone[dayone["stud1"].isin(ids)]
    dayone = dayone[dayone["stud2"].isin(ids)]

    daytwo = daytwo[daytwo["stud1"].isin(ids)]
    daytwo = daytwo[daytwo["stud2"].isin(ids)]

    interactions = []

    # Generates Person objects with the given IDs and Interaction objects for all interactions given in each row of the pandas dataframe
    for i, line in dayone.iterrows() if day1 else daytwo.iterrows():
        # Ignores teachers to be similar to the model
        if line["class1"] == "Teachers" or line["class2"] == "Teachers":
            continue
        # Makes sure only one Person object with a specific ID exists and is added to students
        else:
            if int(line["stud1"]) not in map(lambda x: x.get_ID(), students):
                p1 = Person(grade=int(line["class1"][0]), class_group=line["class1"][1], ID=int(line["stud1"]))
                students.append(p1)
            else:
                p1 = list(filter(lambda x: x.get_ID() == int(line["stud1"]), students))[0]

            if int(line[2]) not in map(lambda x: x.get_ID(), students):
                p2 = Person(grade=int(line["class2"][0]), class_group=line["class2"][1], ID=int(line["stud2"]))
                students.append(p2)
            else:
                p2 = list(filter(lambda x: x.get_ID() == int(line["stud2"]), students))[0]

        interaction = Interaction(p1, p2, count=int(line["count"]))
        interactions.append(interaction)

    n = Network(empiric=students.copy())
    n.generate_network(empiric=interactions)
    return n


def run_disease_transmission(iterations=10, days=100, ID=0):
    """Runs 10 iterations of transmission over 100 days for each one of the empiric days and a switch in addition to the model

    Parameters
    ----------
    iterations : int
        The number of iterations disease transmission over a given number of days. Default is 10 iterations
    days : int
        The number of days in which the disease transmission is generated. Default is 100 days
    """

    # Creates network objects for day 1, day 2 and the model
    graph1 = generate_network(day1=True)
    graph2 = generate_network(day1=False)
    net = Network(222, 5, 2)

    # Creates Disease_transmission objects for the empiric and simulated models
    dis1 = Disease_transmission(graph1)
    dis2 = Disease_transmission(net)
    dis3 = Disease_transmission(net)

    # Generates a dict with keys being (bool, bool). The keys denote whether or not run_transmission
    # should begin on day1 (else day 2) and if it should switch between day 1 and day 2 every other days.
    iterations_dict = {
        (True, False): dict([(i, {}) for i in range(days)]),
        (True, True): dict([(i, {}) for i in range(days)]),
        (False, False): dict([(i, {}) for i in range(days)]),
    }
    # Generates an empty dict for the model where the key is the number of the day and the values are empty dics
    model_int = dict([(i, {}) for i in range(days)])
    for i in range(0, iterations):
        dic2 = dis2.run_transmission(days=days, plot=False, save_to_file=f"empiric{ID}")
        dic3 = dis3.run_transmission(days=days, plot=False, save_to_file=f"empiric_static{ID}", static=True)
        for day1, switch in [(True, False), (True, True), (False, False)]:

            dic = dis1.run_transmission_empiric(
                days=days,
                graph1=graph1,
                graph2=graph2,
                day1=day1,
                switch=switch,
                plot=False,
                save_to_file=f"{day1}{switch}{ID}",
            )
            # for day in range(days):

            #     # Saves the dic to interaction_dict
            #     for disease_key in [e for e in Disease_states] + ["R_null"]:
            #         iterations_dict[(day1, switch)][day][disease_key] = (
            #             iterations_dict.get((day1, switch), {}).get(day, {}).get(disease_key, 0) + dic[day][disease_key]
            #         )

        # for day in range(days):
        #     for disease_key in [e for e in Disease_states] + ["R_null"]:
        #         model_int[day][disease_key] = model_int.get(day, {}).get(disease_key, 0) + dic2[day][disease_key]

    # # Takes the average of all the days and creates a new dict with day as key and an average disease states dicst as values
    # day1TrueSwitchFalse = {
    #     k: {k1: v1 / iterations for k1, v1 in iterations_dict[(True, False)][k].items()}
    #     for k, v in iterations_dict[(True, False)].items()
    # }

    # day1FalseSwitchFalse = {
    #     k: {k1: v1 / iterations for k1, v1 in iterations_dict[(False, False)][k].items()}
    #     for k, v in iterations_dict[(False, False)].items()
    # }

    # day1TrueSwitchTrue = {
    #     k: {k1: v1 / iterations for k1, v1 in iterations_dict[(True, True)][k].items()}
    #     for k, v in iterations_dict[(True, True)].items()
    # }

    # model = {k: {k1: v1 / iterations for k1, v1 in v.items()} for k, v in model_int.items()}

    # save_to_file(day1TrueSwitchFalse, "day1TrueSwitchFalse_average.csv")
    # save_to_file(day1FalseSwitchFalse, "day1FalseSwitchFalse_average.csv")
    # save_to_file(day1TrueSwitchTrue, "day1TrueSwitchTrue_average.csv")
    # save_to_file(model, "model_average.csv")

    # plot_recovered()


def save_to_file(d, filename):
    """Saves a disease states for each day into a csv.

    Parameters
    ----------
    d : dict
        Dict containing the day as key and the number of individuals in each disease state as value
    filename : str
        The filename in which to save the dict as
    """
    with open(f"./data/empiric_vs_model/{filename}", "w") as f:
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


def plot_recovered():
    """Plots the number of recovered as a function of the day for empiric model (day 1, day 2 and switch) against the simulated model"""
    graph1 = generate_network(day1=True)

    dis = Disease_transmission(graph1)
    directory = "./data/empiric_vs_model/"
    files = os.listdir(directory)
    colours = ["darkgoldenrod", "rosybrown", "cadetblue", "mediumseagreen"]
    labels = ["Day two", "Day one", "Switch", "Model"]

    for i, filename in enumerate(files):
        path = os.path.join(directory, filename)
        if i < len(files) - 1:
            dis.plot_recovered(path, False, lab=labels[i], colour=colours[i])
        else:
            dis.plot_recovered(path, lab=labels[i], colour=colours[i])


if __name__ == "__main__":
    ID = sys.argv[1]

    run_disease_transmission(1, 100, ID=ID)
