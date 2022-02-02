# from copyreg import pickle
import random
import math
import numpy as np
import networkx as nx
import pickle
import sys
from scipy.optimize import least_squares
from analysis import Analysis

sys.setrecursionlimit(10000)
from person import Person, Interaction


class Network:
    def __init__(self, num_students, num_grades, num_classes, class_threshhold=20, parameterList=[]):
        self.parameterList = parameterList
        self.students = self.generate_students(num_students, num_grades, num_classes, class_treshold=class_threshhold)
        self.d = (1) * pow(10, -4.3)  # 5.1
        self.graph = self.generate_network()
        self.daily_list = []

    def generate_students(self, num_students, num_grades, num_classes, class_treshold=20, param=False):
        available_grades = [i for i in range(1, num_grades + 1)]
        available_classes = [chr(i) for i in range(97, 97 + num_classes)]

        students = []

        for grade in available_grades:  # Loop igjennom antall grades
            i = 0
            has_filled = False  # Bare et flagg for å sjekke om vi har gjort en random fylling av klasser eller ikke
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
                        has_filled = True  # "Si ifra" til loopen at vi har gjort en random fylling av studentente.
                        for _ in range(num_students // num_grades - pers):
                            students.append(
                                Person(grade, random.choice(available_classes))
                            )  # Legg til studenter i randome klasser
                if has_filled:  # Break dersom vi har fylt random
                    break

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
                # print(int(np.random.poisson(stud.p_vector[pers]*self.d)))
                weight = 1

                tentative_weight = int(np.random.poisson(stud.p_vector[pers] * self.d))
                if tentative_weight < 400:
                    weight = tentative_weight
                if weight:
                    interaction = stud.get_interaction(pers)
                    interaction.count += weight

                    yield interaction

    def generate_a_day(self, hourDay=8, param=False):
        for i in range(len(self.students)):
            self.students[i].renormalize()

        hourly_list = []
        for i in range(hourDay):
            hourly_list.append(self.generate_network())

        dayGraph = nx.empty_graph(hourly_list[0])
        # dayGraph.add_edges_from(hourly_list[0].edges(data=True)+hourly_list[1].edges(data=True))

        # edges = []
        # for i in range(len(self.students)):
        #     first_id = self.students[i]
        #     for j in range(i+1, len(self.students)):
        #         second_id = self.students[j]
        #         count = 0
        #         for graph in hourly_list:
        #             data = graph.get_edge_data(first_id, second_id)
        #             if data is not None:
        #                 count += data['count']

        #         edges.append((first_id, second_id, {'count': count}))

        # edges = []
        # for stud in self.students:
        #     for stud2, interaction in stud.interactions.items():
        #         edges.append((stud, stud2, {'count': interaction.count}))

        # dayGraph.add_edges_from(edges)
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
            # print(students[0].biasVector)
            print("-----Day" + str(i) + "------")
            print("max: " + str(max(list(self.students[0].bias_vector.values()))))
            print("mean: " + str(np.mean(list(self.students[0].bias_vector.values()))))
            print("bias: " + str((self.students[0].bias)))

        dayNumberX = self.daily_list[-1]

        return dayNumberX

    def pickleLoad(self, name):
        file_to_read = open(name, "rb")
        return pickle.load(file_to_read)

    def parameterEstimation(self):

        X0 = [1, 0.5, 110, 0.8, 4, 0.5, 25000, 0.1]

        bounds = (
            [0 for i in range(8)],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        )

        experimentalDataDict = self.pickleLoad("graph1_whole_pixel.pkl")
        exp_class_class = self.pickleLoad("graph1_class_pixel.pkl")
        exp_grade_grade = self.pickleLoad("graph1_grade_pixel.pkl")
        exp_offDiag = self.pickleLoad("graph1_off_diag.pkl")

        def createSubGraphWithoutGraph(graph, diagonal, gradeInteraction):
            G = nx.Graph()

            for node in graph:
                klasseAttr = node.getClass()
                for n in graph.neighbors(node):
                    if diagonal:
                        if n.getClass() == klasseAttr and node.getGrade() == n.getGrade():
                            G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                            G.add_node(n)
                    if gradeInteraction:
                        if n.getGrade() == node.getGrade() and node.getClass() != n.getClass():
                            G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                            G.add_node(n)
            # self.heatmap(G)
            return G

        def createSubGraphWithout(graph, grade, klasse):
            G = nx.Graph()

            for node in graph:
                for n in graph.neighbors(node):
                    if grade and not klasse:
                        if node.getGrade() != n.getGrade():
                            G.add_edge(node, n, count=graph[node][n]["count"])
                    elif not grade:
                        if node.get_class_and_grade() != n.get_class_and_grade():
                            G.add_edge(node, n, count=graph[node][n]["count"])
            # self.heatmap(G)
            return G

        def toCumulative(l):
            n = len(l)
            dictHist = {}
            for i in l:
                if i not in dictHist:
                    dictHist[i] = 1
                else:
                    dictHist[i] += 1
            cHist = {}
            cumul = 1
            for i in dictHist:
                cHist[i] = cumul
                cumul -= float(dictHist[i]) / float(n)
            return cHist

        def toArray(sim, exp):
            Sim_Adj = nx.adjacency_matrix(sim, nodelist=sorted(sim.nodes), weight="count")
            length = len(sim.nodes())

            weights = Sim_Adj[np.triu_indices(length, k=1)].tolist()[0]

            simData = sorted(weights)

            sortedSim = np.sort(simData)

            simCum = toCumulative(sortedSim)

            new_Sim = simCum.copy()

            new_exp = exp.copy()

            same = set(list(new_exp.keys())).intersection(set(list(new_Sim.keys())))

            keys = list(new_exp.keys())

            for key in keys:
                if key not in same:
                    del new_exp[key]

            f_keys = list(new_Sim.keys())
            for key in f_keys:
                if key not in same:
                    del new_Sim[key]

            y = np.array(list(new_exp.items()))
            F = np.array(list(new_Sim.items()))

            return y, F

        def objectiveFunc(X):
            ## Off-diagonal excluding lunch
            a1 = X[0]
            b1 = X[1]

            ## Off-diagonal with lunch
            a2 = X[2]
            b2 = X[3]

            ## Grade-Grade
            a3 = X[4]
            b3 = X[5]

            ## Class-Class
            a4 = X[6]
            b4 = X[7]

            # graph = self.generate_a_day(param=True)
            # graph = self.daily_list[-1]
            graph = Network(225, 5, 2, parameterList=X).generate_a_day()

            off_diagonal = createSubGraphWithout(graph, True, False)
            grade_grade = createSubGraphWithoutGraph(graph, False, True)
            class_class = createSubGraphWithoutGraph(graph, True, False)

            exp, sim = toArray(graph, experimentalDataDict)
            sim_diag, exp_diag = toArray(off_diagonal, exp_offDiag)
            sim_grade, exp_grade = toArray(grade_grade, exp_grade_grade)
            sim_class, exp_class = toArray(class_class, exp_class_class)

            print("med sær ting" + str(exp[:, 1][:85]))
            print("uten sær ting" + str(exp[:, 1][85]))

            diff_whole = exp[:, 1][:85] - sim[:, 1][:85]  # Hva betyr dette???
            diff_diag = exp_diag[:, 1][25] - sim_diag[:, 1][25]
            diff_grade = exp_grade[:, 1][20] - sim_grade[:, 1][20]
            diff_class = exp_class[:, 1][50] - sim_class[:, 1][50]

            res = diff_whole + diff_diag + diff_grade + diff_class
            print(res)
            return res

        result = least_squares(objectiveFunc, X0, method="trf", bounds=bounds, ftol=1e-10, xtol=1e-16)  # loss="cauchy"
        print(result)


if __name__ == "__main__":
    network = Network(225, 5, 2)
    # G = network.generateXdays(8)
    network.parameterEstimation()
