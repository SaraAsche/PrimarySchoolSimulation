"""Network class: Generates interactions between Person objects

Implements Networkx to create networks of interactions between Person objects. 

Typical usage example:

  network = Network(225, 5, 2) 
  network.generate_iterations(10)

Author: Sara Johanne Asche
Date: 14.02.2022
File: network.py
"""

import random
import numpy as np
import networkx as nx
import pickle
import sys

from sklearn import neighbors
from analysis import Analysis
import matplotlib.pyplot as plt

from person import Person, Interaction


class Network:
    """A class to generate network from Interaction objects between Person objects.

    Longer class information...

    Attributes
    ----------
    parameter_list : list
        List of parameters used in Person class to set the p_vector.
    students : list
        A list that contains all the Person objects that attend a certain school
    d : float
        A number that helps scale the weights
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    daily_list : list
        A list of daily nx.Graph objects.

    Methods
    -------
    generate_students(self, num_students, num_grades, num_classes, class_treshold=20)
        Returns a list of num_students Person objects that have been divided into num_grades
        and num_classes with the given class_threshold in mind.
    generate_network(self)
        Returns a network for the given students in the nx.Graph object where the Interaction
        objects are added.
    generate_interactions_for_network(self)
        Returns Interaction objects that occur for the entire school in one hour
    generate_a_day(self, hourDay=8)
        Returns a nx.Graph object where generate_network has been called hourDay times.
        The compiled interactions for hourDay hours is added to a nx.Graph object.
    generate_iterations(self, number)
        Returns the final nx.Graph object out of number days.
    """

    def __init__(self, num_students, num_grades, num_classes, class_treshold=20, parameter_list=[]):
        """Inits Network object with num_students, num_grades and num_classes parameters

        Parameters
        ----------
        num_students: int
            Denotes the number of students for a given school
        num_grades: int
            Denotes the number of grades for a school.
        num_classes: int
            Denotes the number of classes per grade per school.
        class_threshold: int
            Denotes the threshold for how many students can be in one class. Default is set to 20
        parameter_list: list
            List containing parameters used to generate p_vector in the Person class.
        """

        self.parameter_list = parameter_list
        self.students = self.generate_students(num_students, num_grades, num_classes, class_treshold=class_treshold)
        self.d = (1) * pow(10, -2)  # 4.3
        self.graph = self.generate_network()
        self.daily_list = []

    def generate_students(self, num_students, num_grades, num_classes, class_treshold=20) -> list:
        """Generates a list containing Person objects that attend a certain school

        Uses the number of students in a school (given by num_students) to generate num_grade grades
        divided into num_class classes. For each Person object that is generated and added to the
        students list, available_grade and available_classes are used to initialise the Person objects.

        Parameters
        ----------
        num_students: int
            Denotes the number of students for a given school
        num_grades: int
            Denotes the number of grades for a school.
        num_classes: int
            Denotes the number of classes per grade per school.
        class_threshold: int
            Denotes the threshold for how many students can be in one class. Default is set to 20
        """

        available_grades = [i for i in range(1, num_grades + 1)]
        available_classes = [chr(i) for i in range(97, 97 + num_classes)]

        students = []

        for grade in available_grades:  # Loop through all grades
            i = 0
            for pers in range(1, num_students // num_grades + 1):  # Loop through all Persons per grade
                students.append(Person(grade, available_classes[i]))  # Add Person to Students
                if pers % (num_students // num_grades // num_classes) == 0:
                    i += 1  # If we have reached a new class within a grade, i is updated.

                    if i >= len(available_classes) and num_students // num_grades - pers >= class_treshold:
                        # If it is not possible to have the same amount og people in each class, and we have class_threshold students left, we make a new class
                        available_classes.append(chr(ord(available_classes[-1]) + 1))
                    elif i >= len(
                        available_classes
                    ):  # If we have less than class_threshold students left, they are added in a random class
                        break

        ### Sorted many times to ensure the ID is the same as the other ones in the grade a specific individual is in.
        grade_ind = 0
        while len(students) != num_students:
            if grade_ind >= len(available_grades):
                grade_ind = 0
            students.append(Person(available_grades[grade_ind], random.choice(available_classes)))
            grade_ind += 1

        students = sorted(students, key=lambda x: x.get_class_and_grade())

        ## Generate bias_vector
        for i in range(len(students)):
            students[i].id = i
            students[i].generate_bias_vector(students)

        students = sorted(students)

        ## Generate p_vector
        for i in range(len(students)):
            students[i].generate_p_vector(students, self.parameter_list)

        return students

    def generate_interactions_for_network(self) -> Interaction:
        """Generates and returns Interaction objects between students for one hour

        Loops through the students list twice so that Interaction objects have
        a change of being created between all individuals in the school. The
        poisson distribution uses the p_vector between two individuals as well
        as the d variable to increase the weight of the interactions.
        Yields the interactions to generate_network().

        """

        for i in range(len(self.students)):
            stud = self.students[i]
            for j in range(i + 1, len(self.students)):
                pers = self.students[j]

                weight = 0

                tentative_weight = np.random.poisson(stud.p_vector[pers] * self.d)

                if stud.get_grade() == pers.get_grade():
                    if tentative_weight < 80:
                        weight = tentative_weight
                if stud.get_class_and_grade() == pers.get_class_and_grade():
                    if tentative_weight < 100:
                        weight = tentative_weight

                if tentative_weight < 50:  # 180
                    # It is only possible to interact 180 times an hour (if each interaction is maxumum 20 seconds long 60*60/20)
                    weight = tentative_weight

                if weight:
                    interaction = stud.get_interaction(pers)
                    interaction.count += weight

                    yield interaction

    def generate_network(self) -> nx.Graph:
        """Generates a hourly network

        Uses the Person objects in the students list as nodes and adds all interactions
        gathered from generate_interactions_for_network() as edges with their weight
        being set equal to the Interaction object's attribute count.

        """

        graph = nx.Graph()

        for student in self.students:
            graph.add_node(student)

        for interaction in self.generate_interactions_for_network():

            p1 = interaction.get_p1()
            p2 = interaction.get_p2()
            weight = interaction.get_count()

            graph.add_edge(p1, p2, count=weight)

        return graph

    def generate_a_day(self, hourDay=8) -> nx.Graph:
        """Generates a nx.Graph object for a day with hourDay hours

        Uses renormalise and generates a new p_vector at the start
        of the day.

        Parameters
        ----------
        hourDay : int
            Describes how many hours there is in a given schoolday
        """
        ## Renormalise bias_vector at the beginning of the day
        for i in range(len(self.students)):
            self.students[i].renormalize()

        ## Generate new p-vector with the updated bias_vector
        self.students[0].generate_p_vector(self.students, [])

        ## Empty list to append the hour by hour networks
        hourly_list = []
        for i in range(hourDay):
            hourly_list.append(self.generate_network())

        ## Empty graph based on the nodes in the nettwork. Where interactions will be added
        dayGraph = nx.empty_graph(hourly_list[0])

        for graph in hourly_list:
            for node, neighbour, attrs in graph.edges.data():
                if not dayGraph.has_edge(node, neighbour):
                    dayGraph.add_edge(node, neighbour, count=attrs["count"])
                else:
                    dayGraph[node][neighbour]["count"] += attrs["count"]

        k = 0.5

        for i in range(len(self.students)):
            stud1 = self.students[i]
            for j in range(len(self.students)):
                stud2 = self.students[j]
                if i == j:
                    continue
                if self.students[j] in dayGraph[stud1]:
                    stud1.bias_vector[stud2] += dayGraph[stud1][self.students[j]]["count"]  ## What does this do?
                    # stud1.bias_vector[stud2] -= k * (stud1.bias_vector[stud2] - stud1.bias)  ## What does this do?

        return dayGraph  ## Only returns the final hour. should it not create something based on all hours?

    def generate_iterations(self, number) -> nx.Graph:
        """Generates number number of days and returns the nx.Graph from the final day

        Parameters
        ----------
        number : int
            Number of days that should be generated
        """

        for i in range(number):
            self.daily_list.append(self.generate_a_day())

            print("-----Day " + str(i) + "------")
            print(
                "max: " + str(max(list(self.students[0].bias_vector.values())))
            )  # Prints out the highest bias_vector of student 0
            print(
                "mean: " + str(np.mean(list(self.students[0].bias_vector.values())))
            )  # prints out the mean of bias_vector of student 0
            print("bias: " + str((self.students[0].bias)))  # prints out the bias attribute of student 0

        #!!dayGraph = nx.empty_graph(self.daily_list[0])
        dayGraph = self.daily_list[0]
        d = {}
        for graph in self.daily_list:
            for node, neighbour, attrs in graph.edges.data():
                if not dayGraph.has_edge(node, neighbour):
                    #dayGraph.add_edge(node, neighbour, count=attrs["count"])
                    continue
                else:
                    dayGraph[node][neighbour]["count"] += attrs["count"]
                    d[(node, neighbour)] = d.get((node, neighbour), 0) + 1

        for node, neighbour, attrs in graph.edges.data():
            attrs["count"] = (
                attrs["count"] / d.get((node, neighbour), 1)
            )  #TODO Må finne en måte å dele på antall ganger denne spesifikke edgen har blitt plusset på

        dayNumberX = dayGraph  # self.daily_list[-1]  # Returns the final day

        return dayNumberX

    def pickle_load(self, name):
        file_to_read = open(name, "rb")
        return pickle.load(file_to_read)

    ### Estimation
    def parameterEstimation(self):
        def create_sub_graph_grade_class(graph, diagonal, gradeInteraction):
            G = nx.Graph()

            for node in graph.nodes():
                klasseAttr = node.get_class()
                for n in graph.nodes():
                    if diagonal:
                        if n.get_class() == klasseAttr and node.get_grade() == n.get_grade():
                            if n in graph.neighbors(node):
                                G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                                G.add_node(n)
                                G.add_node(node)
                            else:
                                G.add_edge(node, n, count=0)
                                G.add_node(n)
                                G.add_node(node)
                    if gradeInteraction:
                        if n.get_grade() == node.get_grade() and klasseAttr != n.get_class():
                            if n in graph.neighbors(node):
                                G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                                G.add_node(n)
                                G.add_node(node)
                            else:
                                G.add_edge(node, n, count=0)
                                G.add_node(n)
                                G.add_node(node)
                    else:
                        G.add_edge(node, n, count=0)
                        G.add_node(n)
                        G.add_node(node)

            return G

        def create_sub_graph_off_diagonal(graph, grade, klasse):
            G = nx.Graph()
            # print("this is the subgraph before anything: " + str(graph))
            for node in graph.nodes():
                for n in graph.nodes():
                    if grade and not klasse:
                        if node.get_grade() != n.get_grade():
                            if n in graph.neighbors(node):
                                G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                                G.add_node(n)
                                G.add_node(node)
                            else:
                                G.add_edge(node, n, count=0)
                                G.add_node(n)
                                G.add_node(node)
                    elif not grade:
                        if node.get_class_and_grade() != n.get_class_and_grade():
                            if n in graph.neighbors(node):
                                G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                                G.add_node(n)
                                G.add_node(node)
                            else:
                                G.add_edge(node, n, count=0)
                                G.add_node(n)
                                G.add_node(node)
                    else:
                        G.add_edge(node, n, count=0)
                        G.add_node(n)
                        G.add_node(node)

            return G

        def load_all_pixel_dist_non_cumulative():
            exp_whole = self.pickle_load("graph1_whole_pixel_not_Cumulative.pkl")
            exp_diag = self.pickle_load("graph1_off_diag_not_Cumulative.pkl")
            exp_class = self.pickle_load("graph1_class_pixel_not_Cumulative.pkl")
            exp_grade = self.pickle_load("graph1_grade_pixel_not_Cumulative.pkl")

            return exp_whole, exp_diag, exp_class, exp_grade

        def rank_interaction(pixel):
            pixel[::-1].sort()
            pixel_list = pixel.tolist()

            pixel_dict = {}

            for i in range(len(pixel_list) - 1):
                pixel_dict[i] = pixel_list[i]
            pixel_dict = np.array(list(pixel_dict.values()))
            return pixel_dict

        def setup():
            exp_whole, exp_diag, exp_class, exp_grade = load_all_pixel_dist_non_cumulative()
            exp_whole_dict = rank_interaction(exp_whole)
            exp_diag_dict = rank_interaction(exp_diag)
            exp_class_dict = rank_interaction(exp_class)
            exp_grade_dict = rank_interaction(exp_grade)

            return exp_whole_dict, exp_diag_dict, exp_class_dict, exp_grade_dict

        def toSorted(graph):
            Sim_Adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes), weight="count")
            length = len(graph.nodes())

            weights = Sim_Adj[np.triu_indices(length, k=1)].tolist()[0]

            simData = np.array(sorted(weights))

            return rank_interaction(simData)

        exp_whole, exp_diag, exp_class, exp_grade = setup()

        def parameters(X):
            def power_of_2(x):
                return np.power(x, 2)

            graph = Network(236, 5, 2, class_treshold=23, parameter_list=X).generate_a_day()

            off_diagonal = create_sub_graph_off_diagonal(graph, True, False)
            # grade_grade = create_sub_graph_grade_class(graph, False, True)
            # class_class = create_sub_graph_grade_class(graph, True, False)

            # print(f"OFf diag: {off_diagonal}, grade_grade: {grade_grade}, class_class: {class_class}")

            # whole_dict = toSorted(graph)
            diag_dict = toSorted(off_diagonal)
            # grade_dict = toSorted(grade_grade)
            # class_dict = toSorted(class_class)

            raise_2 = np.vectorize(power_of_2)

            output = raise_2(exp_diag - diag_dict)
            # raise_2(exp_whole - whole_dict)+ raise_2(exp_diag - diag_dict)+ raise_2(exp_grade - grade_dict)+ raise_2(exp_class - class_dict))
            output = output.sum()
            return output

        def loop_parameters(low, top, step):
            i = 0
            dict_results = {}
            res = 0
            for _ in range(20):
                parameter = [3, 0.1, 9, 0.01, 1, 1, 1, 1]
                res += parameters(parameter)

            print(i)
            i += 1
            dict_results[i] = [(res / 20), parameter]
            print("res: " + str(res) + " parametere: " + str(parameter))
            # print(min(dict_results.values()[0]))
            # print(min(dict_results.values()[1]))

        loop_parameters(3, 9, 0.1)
        # print(parameters([]))


if __name__ == "__main__":
    network = Network(num_students=236, num_grades=5, num_classes=2, class_treshold=23)
    # G = network.generate_iterations(8)
    network.parameterEstimation()
