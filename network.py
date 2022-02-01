import random
import math
import numpy as np
import networkx as nx

from person import Person, Interaction


class Network:
    def __init__(self, num_students, num_grades, num_classes, class_threshhold=20) -> None:
        self.students = self.generate_students(num_students, num_grades, num_classes, class_treshold=class_threshhold)
        self.d = (2 / 3) * pow(10, -5.5)  # 5.1
        self.graph = self.generate_network(100)
        self.daily_list = []

    def generate_students(self, num_students, num_grades, num_classes, class_treshold=20):
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
            students[i].generate_p_vector(students)
        # for i in range(len(students)):
        #     students[i].constBias = 40*(math.log(1/random.random()))#pow(random.random(),exp) #powerlaw
        #     students[i].bias_vector = {}
        #     for j in range(len(students)):
        #         students[i].bias_vector[j] = students[i].constBias

        return students

    def generate_network(self, i):
        graph = nx.Graph()

        for student in self.students:
            graph.add_node(student)

        for interaction in self.generate_interactions_for_network(i):

            p1 = interaction.getp1()
            p2 = interaction.getp2()
            weight = interaction.getcount()

            graph.add_edge(p1, p2, count=weight)

        return graph

    def generate_interactions_for_network(self, i):

        for i in range(len(self.students)):
            stud = self.students[i]
            for j in range(i + 1, len(self.students)):
                pers = self.students[j]
                # print(int(np.random.poisson(stud.p_vector[pers]*self.d)))
                weight = int(np.random.poisson(stud.p_vector[pers] * self.d))
                if weight:
                    interaction = stud.get_interaction(pers)
                    interaction.count += weight

                    yield interaction

    def generate_a_day(self, hourDay=8):
        for i in range(len(self.students)):
            self.students[i].renormalize()

        hourly_list = []
        for i in range(hourDay):
            hourly_list.append(self.generate_network(i))

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


if __name__ == "__main__":
    network = Network(225, 5, 2)
    network.generateXdays(8)
