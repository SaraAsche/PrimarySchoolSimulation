"""Network class: Generates Interaction object between two Person objects

Implements Networkx to create networks consisting of weighted edges defined by 
Interaction objects. Nodes are Person objects with their associated attributes.  

Typical usage example:

  network = Network(225, 5, 2) 
  network.generate_a_day()

Author: Sara Johanne Asche
Date: 14.02.2022
File: network.py
"""

import random
import numpy as np
import networkx as nx
import pickle

from enums import Traffic_light
from person import Person
from interaction import Interaction


class Network:
    """A class to generate network from Interaction objects between Person objects.

    A network object uses interaction objects between person objects to create a
    nx.Graph objects of all the interactions.

    Attributes
    ----------
    parameter_list : list
        List of parameters used in Person class to set the p_vector.
    weights : list
        List of weights for interaction generation
    students : list
        A list that contains all the Person objects that attend a certain school
    d : float
        A number that helps scale the weights
    available_grades : list
        List of available grades in the school
    available_classes : list
        List of available classes in the school
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    iteration_list : list
        A list of daily nx.Graph objects.

    Methods
    -------
    get_graph()
        Returns the nx.Graph object stored in the Network object attribute graph
    get_students()
        Returns the list of Person objects stored in the Network objects' students attribute
    generate_students(num_students : int, num_grades : int, num_classes : int, class_treshold=20)
        Returns a list of num_students Person objects that have been divided into num_grades
        and num_classes with the given class_threshold in mind.
    generate_weights(stoplight: any, stud1: Person, stud2: Person)
        Generates weights for the edges between two Person objects
     generate_interactions_for_network(stoplight=None)
        Returns Interaction objects that occur for the entire school in one hour
    get_available_classes()
        Returns the number of classes available with the given number of grades and number of classes
    def get_available_grades()
        Returns the number of grades available with the given number of grades
        and number of classes
    generate_network(stoplight=None, empiric=None)
        Returns a network for the given students in the nx.Graph object where the Interaction
        objects are added.
    generate_a_day(stoplight=None, hourDay=8)
        Returns a nx.Graph object where generate_network has been called hourDay times.
        The compiled interactions for hourDay hours is added to a nx.Graph object.
    reset_student_disease_states()
        Disease states for all students in the network is reset. Happens when a new iteration of run_transmission occurs
    generate_iterations(number : int)
        Generates num number of generated networks and returns a list of all the nx.Graph networks
    remove_all_interactions(graph: nx.Graph, person: Person)
        Removes all interaction Person person has with the other individuals, if for instance in quarantine/isolation
    pickle_load(name: str, pixel=True)
        Returns the dict of a pickle file
    """

    def __init__(
        self, num_students=236, num_grades=5, num_classes=2, class_treshold=20, parameter_list=[], empiric=None
    ):
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
        empiric : list
            Gives a list of students, if None students are generated. Default is None
        """
        self.parameter_list = parameter_list

        ## It is only possible to interact 180 times an hour (if each interaction is maxumum 20 seconds long 60*60/20)
        self.weigths = {
            "None": [100, 80, 60],
            "G": [100, 80, 60],
            "Y": [40, 30, 20],
            "YC": [60, 40, 20],
            "R": [20, 15, 10],
            "RC": [50, 30, 10],
        }

        self.d = (1) * pow(10, -2)  # 4.3

        self.available_grades = [i for i in range(1, num_grades + 1)]
        self.available_classes = [chr(i) for i in range(97, 97 + num_classes)]  # A,B,C etc is generated

        if empiric:
            self.students = empiric
        else:
            self.students = self.generate_students(num_students, num_grades, num_classes, class_treshold=class_treshold)
            self.graph = self.generate_a_day()

        self.iteration_list = []

    def get_graph(self) -> nx:
        """Returns the nx.Graph attribute of the Network object"""
        return self.graph

    def get_students(self) -> list:
        """Returns the list of Person objects stored in the Network objects' students attribute"""
        return self.students

    def generate_students(self, num_students: int, num_grades: int, num_classes: int, class_treshold=20) -> list:
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

        students = []

        for grade in self.available_grades:  # Loop through all grades
            i = 0
            for pers in range(1, num_students // num_grades + 1):  # Loop through all Persons per grade
                students.append(Person(grade, self.available_classes[i]))  # Add Person to Students
                if pers % (num_students // num_grades // num_classes) == 0:
                    i += 1  # If we have reached a new class within a grade, i is updated.

                    if i >= len(self.available_classes) and num_students // num_grades - pers >= class_treshold:
                        ## If it is not possible to have the same amount of people in each class, and we have class_threshold students left, we make a new class
                        self.available_classes.append(chr(ord(self.available_classes[-1]) + 1))
                    elif i >= len(self.available_classes):
                        break

        ## Assign studets to random class if the number of students does not divide evenly
        grade_ind = 0
        while len(students) != num_students:
            if grade_ind >= len(self.available_grades):
                grade_ind = 0
            students.append(Person(self.available_grades[grade_ind], random.choice(self.available_classes)))
            grade_ind += 1

        ### Sorted to ensure the ID is the same as the other ones in the grade a specific individual is in.
        students = sorted(students, key=lambda x: x.get_class_and_grade())

        for i in range(len(students)):
            students[i].generate_p_vector(students, self.parameter_list)

        return students

    def generate_weights(self, stoplight: any, stud1: Person, stud2: Person):
        """Generates weights for the edges between two Person objects

        Parameters
        ----------
        stoplight : Traffic_light/None
            A traffic light object or None. Says which of the stoplight values to return
        stud1 : Person
            The first Person object of the interaction
        stud2 : Person
            The second Person object of the interaction
        """

        same_cohort = stud1.get_cohort() == stud2.get_cohort() and stud1.get_cohort() is not None

        # Returns the correct list of weight parameters. For example weight['YC'] which is [90, 70, 50]
        return (
            self.weigths["None"]
            if stoplight is None
            else self.weigths[str(stoplight.value) + ("C" if same_cohort else "")]
        )

    def generate_interactions_for_network(self, stoplight=None) -> Interaction:
        """Generates and returns Interaction objects between students for one hour

        Loops through the students list twice so that Interaction objects have
        a chance of being created between all individuals in the school. The
        poisson distribution uses the p_vector between two individuals as well
        as the d variable to increase the weight of the interactions.
        Yields the Interactions to generate_network()

        Parameters
        ----------
        stoplight : Traffic_light/None
            A traffic light object or None. Denotes which Traffic light level the school is in
        """

        for i in range(len(self.students)):

            stud = self.students[i]

            # If get_tested() returns True, the individual has tested positive and will be isolated (i.e. no interactions will be generated)
            if stud.get_tested():
                continue

            for j in range(i + 1, len(self.students)):

                pers = self.students[j]

                # If get_tested() returns True, the individual has tested positive and will be isolated (i.e. no interactions will be generated)
                if pers.get_tested():
                    continue

                # Generates specific thresholds according to which Traffic light state the school is in
                weight_same_class, weight_same_grade, weight_rest = self.generate_weights(stoplight, stud, pers)

                # A tentative weight is generated
                tentative_weight = np.random.poisson(stud.p_vector[pers] * self.d)

                # Default value for weight is set
                weight = 0

                # If individuals are in the same class the tentative weight has to be lower than the set weight_same_class parameter
                if stud.get_class_and_grade() == pers.get_class_and_grade():
                    if tentative_weight < weight_same_class:
                        weight = tentative_weight

                # If individuals only in the same grade but not class, the tentative weight has to be lower than the set weight_same_grade parameter
                elif stud.get_grade() == pers.get_grade():
                    if tentative_weight < weight_same_grade:
                        weight = tentative_weight

                # For the rest of the network, the tentative_weight has to be less than weight_rest to be accepted
                elif tentative_weight < weight_rest:
                    weight = tentative_weight

                # If the weight is still 0, no interactions are added. If it is above 0, interactions are added
                if weight:
                    interaction = stud.get_interaction(pers)
                    interaction.count += weight
                    yield interaction

    def get_available_classes(self) -> list:
        """Returns the number of classes available with the given number of grades
        and number of classes"""
        return self.available_classes

    def get_available_grades(self) -> list:
        """Returns the number of grades available with the given number of grades
        and number of classes"""
        return self.available_grades

    def generate_network(self, stoplight=None, empiric=None) -> nx.Graph:
        """Generates a hourly network

        Uses the Person objects in the students list as nodes and adds all interactions
        gathered from generate_interactions_for_network() as edges with their weight
        being set equal to the Interaction object's attribute count.

        Parameters
        ----------
        stoplight :
             Traffic_light object or None. Denotes which Traffic light level the school is in
        empiric : None or list of Interaction objects
            Wheter or not a network is generated for an empiric network (with a given list of
            interactions) or generated based on students (None). Default is None

        """

        graph = nx.Graph()

        # Each person object is added as a node
        for student in self.students:
            graph.add_node(student)

        # If empiric, the interactions are taken from empiric, which is a list of interactions
        if empiric:
            for interaction in empiric:

                p1 = interaction.get_p1()
                p2 = interaction.get_p2()
                weight = interaction.get_count()

                graph.add_edge(p1, p2, count=weight)
        # If simulated, the interactions are generated based on the students present
        else:
            for interaction in self.generate_interactions_for_network(stoplight):

                p1 = interaction.get_p1()
                p2 = interaction.get_p2()
                weight = interaction.get_count()

                graph.add_edge(p1, p2, count=weight)

        self.graph = graph
        return graph

    def generate_a_day(self, stoplight=None, hourDay=8) -> nx.Graph:
        """Generates a nx.Graph object for a day with hourDay hours

        Uses renormalise and generates a new p_vector at the start
        of the day.

        Parameters
        ----------
        hourDay : int
            Describes how many hours there is in a given schoolday
        """

        # Empty list to append the hour by hour networks
        hourly_list = []
        for i in range(hourDay):
            hourly_list.append(self.generate_network(stoplight=stoplight))

        # Empty graph based on the nodes in the nettwork. Where interactions will be added
        dayGraph = nx.empty_graph(hourly_list[0])

        # Loops through all graphs, and adds edges if interaction is new for that hour, or increases the count if interaction has occured earlier
        for graph in hourly_list:
            for node, neighbour, attrs in graph.edges.data():
                if not dayGraph.has_edge(node, neighbour):
                    dayGraph.add_edge(node, neighbour, count=attrs["count"])
                else:
                    dayGraph[node][neighbour]["count"] += attrs["count"]

        # A p-vector is generated for all students
        for i in range(len(self.students)):
            self.students[i].generate_p_vector(self.students)

        self.graph = dayGraph

        return dayGraph

    def reset_student_disease_states(self):
        """Disease states for all students in the network is reset.
        Happens when a new iteration of run_transmission occurs"""
        for stud in self.students:
            stud.state = stud.disease_state_start()

    def generate_iterations(self, number: int) -> list:

        """Generates num number of generated networks and returns a list of all the nx.Graph networks

        Parameters
        ----------
        number : int
            Number of days that should be generated
        """

        for i in range(number):
            self.iteration_list.append(self.generate_a_day())

            print("-----Iteration " + str(i) + "------")
            print(
                "max: " + str(max(list(self.students[0].bias_vector.values())))
            )  # Prints out the highest bias_vector of student 0
            print(
                "mean: " + str(np.mean(list(self.students[0].bias_vector.values())))
            )  # prints out the mean of bias_vector of student 0
            print("bias: " + str((self.students[0].bias)))  # prints out the bias attribute of student 0

        return self.iteration_list

    def remove_all_interactions(self, graph: nx.Graph, person: Person) -> None:
        """Removes all interaction Person person has with the other individuals, if for instance in quarantine/isolation

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph object that describes interactions in the network
        person : Person
            The Person object person from which all interactions should be removed
        """

        person_edges = list(graph.edges(person))
        self.graph.remove_edges_from(person_edges)

    def decrease_interaction_day(self, stoplight) -> nx.Graph:
        """Returns a graph where generate_a_day is run on the Traffic_light level stoplight
        Parameters
        ----------
        stoplight : Traffic_light
            Traffic_light enum that denotes the stoplight level of the primary school
        """

        graph = self.generate_a_day(stoplight)
        return graph

    def pickle_load(self, name: str, pixel=True) -> dict:
        """Returns the dict of a pickle file

        Parameters
        ----------
        name : str
            name of the file that is to be loaded
        pixel : bool
            Denotes whether or not the file is in pixel or degree folder.
            Defauls is pixel = True, meaning it is in the pixel folder
        """

        file_to_read = open("./data/pickles/" + ("pixel/" if pixel else "degree/") + name, "rb")
        return pickle.load(file_to_read)
