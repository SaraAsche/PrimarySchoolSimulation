# from copyreg import pickle
import random
import math
from black import diff
import numpy as np
import networkx as nx
import pickle
import sys
from scipy.optimize import least_squares, dual_annealing, leastsq
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier
from analysis import Analysis
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)
from person import Person, Interaction


class Network:
    def __init__(self, num_students, num_grades, num_classes, class_treshold=20, parameterList=[]):
        self.parameterList = parameterList
        self.students = self.generate_students(num_students, num_grades, num_classes, class_treshold=class_treshold)
        self.d = (1) * pow(10, -2)  # 4.3
        self.graph = self.generate_network()
        self.daily_list = []

    def generate_students(self, num_students, num_grades, num_classes, class_treshold=20):
        available_grades = [i for i in range(1, num_grades + 1)]
        available_classes = [chr(i) for i in range(97, 97 + num_classes)]

        students = []

        for grade in available_grades:  # Loop igjennom antall grades
            i = 0
            for pers in range(1, num_students // num_grades + 1):  # Loop igjennom antall personer pr grade
                students.append(Person(grade, available_classes[i]))  # Legg til person i students
                if (
                    pers % (num_students // num_grades // num_classes) == 0
                ):  # Dersom vi er kommet oss til ny klasse innenfor grade, må vi oppdatere i
                    i += 1

                    if i >= len(available_classes) and num_students // num_grades - pers >= class_treshold:
                        # Dersom det ikke går å ha  like mange i hver klasse, og vi har igjen class_treshold antall studenter, lager vi en ny klasse
                        available_classes.append(chr(ord(available_classes[-1]) + 1))
                    elif i >= len(
                        available_classes
                    ):  # Hvis vi ikke har fler enn class_threshold studenter igjen legger vi de i en random klasse av de vi allerede har
                        break

        grade_ind = 0
        while len(students) != num_students:
            if grade_ind >= len(available_grades):
                grade_ind = 0
            students.append(Person(available_grades[grade_ind], random.choice(available_classes)))
            grade_ind += 1

        students = sorted(students, key=lambda x: x.get_class_and_grade())

        for i in range(len(students)):
            students[i].id = i
            students[i].generate_bias_vector(students)

        students = sorted(students)

        for i in range(len(students)):
            students[i].generate_p_vector(students, self.parameterList)

        return students

    def generate_network(self):
        graph = nx.Graph()

        for student in self.students:
            graph.add_node(student)

        for interaction in self.generate_interactions_for_network():

            p1 = interaction.getp1()
            p2 = interaction.getp2()
            weight = interaction.getcount()

            graph.add_edge(p1, p2, count=weight)

        return graph

    def generate_interactions_for_network(self):

        for i in range(len(self.students)):
            stud = self.students[i]
            for j in range(i + 1, len(self.students)):
                pers = self.students[j]
                # print(int(np.random.poisson(stud.p_vector[pers]*self.d))

                weight = 0

                tentative_weight = np.random.poisson(stud.p_vector[pers] * self.d)  # random.poisson

                if (
                    tentative_weight < 180
                ):  # It is only possible to interact 180 times an hour (if each interaction is maxumum 20 seconds long 60*60/20)
                    weight = tentative_weight

                if weight:
                    interaction = stud.get_interaction(pers)
                    interaction.count += weight

                    yield interaction

    def generate_a_day(self, hourDay=8, param=False):
        for i in range(len(self.students)):
            self.students[i].renormalize()

        self.students[0].generate_p_vector(self.students, [])

        hourly_list = []
        for i in range(hourDay):
            hourly_list.append(self.generate_network())

        dayGraph = nx.empty_graph(hourly_list[0])

        dayGraph = hourly_list[-1]
        k = 0.5

        for i in range(len(self.students)):
            stud1 = self.students[i]
            for j in range(len(self.students)):
                stud2 = self.students[j]
                if i == j:
                    continue
                if self.students[j] in dayGraph[stud1]:
                    stud1.bias_vector[stud2] += dayGraph[stud1][self.students[j]][
                        "count"
                    ]  # evt  += dayGraph[i][j]['count']*stortNokTall
                    stud1.bias_vector[stud2] -= k * (stud1.bias_vector[stud2] - stud1.bias)

        return dayGraph

    def generateXdays(self, numDays):
        for i in range(numDays):
            self.daily_list.append(self.generate_a_day())

            print("-----Day" + str(i) + "------")
            print("max: " + str(max(list(self.students[0].bias_vector.values()))))
            print("mean: " + str(np.mean(list(self.students[0].bias_vector.values()))))
            print("bias: " + str((self.students[0].bias)))

        dayNumberX = self.daily_list[-1]

        return dayNumberX

    def pickleLoad(self, name):
        file_to_read = open(name, "rb")
        return pickle.load(file_to_read)

    ### Estimation
    def parameterEstimation(self):
        def createSubGraphWithoutGraph(graph, diagonal, gradeInteraction):
            G = nx.Graph()

            for node in graph.nodes():
                klasseAttr = node.getClass()
                for n in graph.nodes():
                    if diagonal:
                        if n.getClass() == klasseAttr and node.getGrade() == n.getGrade():
                            if n in graph.neighbors(node):
                                G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                                G.add_node(n)
                                G.add_node(node)
                            else:
                                G.add_edge(node, n, count=0)
                                G.add_node(n)
                                G.add_node(node)
                    if gradeInteraction:
                        if n.getGrade() == node.getGrade() and klasseAttr != n.getClass():
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

        def createSubGraphWithout(graph, grade, klasse):
            G = nx.Graph()
            # print("this is the subgraph before anything: " + str(graph))
            for node in graph.nodes():
                for n in graph.nodes():
                    if grade and not klasse:
                        if node.getGrade() != n.getGrade():
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

        def load_all_pixeldist_non_cumulative():
            exp_whole = self.pickleLoad("graph1_whole_pixel_not_Cumulative.pkl")
            exp_diag = self.pickleLoad("graph1_off_diag_not_Cumulative.pkl")
            exp_class = self.pickleLoad("graph1_class_pixel_not_Cumulative.pkl")
            exp_grade = self.pickleLoad("graph1_grade_pixel_not_Cumulative.pkl")

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
            exp_whole, exp_diag, exp_class, exp_grade = load_all_pixeldist_non_cumulative()
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

            graph = Network(236, 5, 2, class_treshold=23, parameterList=X).generate_a_day()

            off_diagonal = createSubGraphWithout(graph, True, False)
            # grade_grade = createSubGraphWithoutGraph(graph, False, True)
            # class_class = createSubGraphWithoutGraph(graph, True, False)

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
    # G = network.generateXdays(8)
    network.parameterEstimation()
