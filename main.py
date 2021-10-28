import random
import itertools
import enum
import cProfile


import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint


from person import Person, Interaction
from enums import Grade, Age_group
from layers import Grades, Klasse, Lunchbreak, Recess

def weightedFlip(p):
    return random.random() < p


def interaction_between_persons(p1, p2, simGrid):


    similar = len(simGrid[p1.getID()][p2.getID()])
    
    #p = similar/5
    p=similar*10;

    return np.random.poisson(similar)

def generate_students(num_students, num_grades, num_classes, class_treshold = 20):
    available_grades = [i for i in range(1, num_grades + 1)]
    available_classes = [chr(i) for i in range(97, 97 + num_classes)]

   #print(available_classes)

    students = []
    
    for grade in available_grades: # Loop igjennom antall grades
        i = 0
        has_filled = False # Bare et flagg for å sjekke om vi har gjort en random fylling av klasser eller ikke
        for pers in range(1, num_students//num_grades + 1): # Loop igjennom antall personer pr grade
                students.append(Person(grade, available_classes[i])) # Legg til person i students
                if pers % (num_students//num_grades//num_classes) == 0: # Dersom vi er kommet oss til ny klasse innenfor grade, må vi oppdatere i
                    i += 1
                    if i >= len(available_classes) and num_students//num_grades - pers >= class_treshold: 
                        # Dersom det ikke går å ha  like mange i hver klasse, og vi har igjen class_treshold antall studenter, lager vi en ny klasse
                        available_classes.append(chr(ord(available_classes[-1]) + 1))
                    elif i >= len(available_classes): # Hvis vi ikke har fler enn class_threshold studenter igjen legger vi de i en random klasse av de vi allerede har
                        has_filled = True # "Si ifra" til loopen at vi har gjort en random fylling av studentente.
                        for _ in range(num_students//num_grades - pers):
                            students.append(Person(grade, random.choice(available_classes))) # Legg til studenter i randome klasser
                if has_filled: # Break dersom vi har fylt random
                    break
    return students

def generate_network(students):
    
    graph = nx.Graph()

    for student in students:
        graph.add_node(student.getID(), grade=student.getGrade(), klasse = student.getClass(), lunchgroup = student.getLunchgroup())

    # interactions = generate_interactions_for_network(students, graph)
    
    for interaction in generate_interactions_for_network(students, graph):
        p1 = interaction.getp1().getID()
        p2 = interaction.getp2().getID()
        weight = interaction.getcount()

        graph.add_edge(p1, p2, count=weight)
    
    return graph

def generate_interactions_for_network(students, network):
    interactions = []

    checked_students = students.copy()

    simGrid = generate_similarity_Grid(network)

    for i in range(len(students)):
        stud = students[i]
        for j  in range(i + 1,len(checked_students)):
            pers = checked_students[j]
            if pers.getID in stud.find_all_interactions_for_person(pers, interactions):
                break
            else:
                weight = interaction_between_persons(stud, pers, simGrid)
                if weight:
                    # interactions.append(Interaction(stud, pers, weight))
                    yield Interaction(stud, pers, weight)
              
    # return interactions

def generate_similarity_Grid(network):
    n = network.number_of_nodes()
    print(n)
    # simGrid = ['']*n
    # for x in range(n):
    #     simGrid[x] = ['']*n

    simGrid = [['' for _ in range(n)] for _ in range(n)]


    #print(simGrid)

    similar = ['S'] #same school

    for i in network.nodes():
        for j in network.nodes():
            if (network.nodes[i]['klasse']==network.nodes[j]['klasse']) and network.nodes[i]['grade']==network.nodes[j]['grade']:
                similar.append('K')
            if network.nodes[i]['grade']==network.nodes[j]['grade']:
                similar.append('G') #samme trinn
            if network.nodes[i]['lunchgroup'] ==network.nodes[j]['lunchgroup']:
                similar.append('L')
            
            simGrid[i][j] = similar

            similar = ['S']

    return simGrid
     
def displayNetwork(graph):
    nx.draw(graph)
    plt.show()

def heatmap(graph):
    A = nx.adjacency_matrix(graph, weight='count')
    A_M = A.todense()

    sns.heatmap(A_M)
    plt.show()

def histDistribution(graph):
    degs = {}
    for n in graph.nodes ():
        deg = graph.degree(n, weight='weight')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    plt.hist(data, bins=10, color='skyblue', ec = 'black') #col = 'skyblue for day2, mediumseagreen for day1
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def plot_degree_distribution(G):
    degs = {}
    for n in G.nodes ():
        deg = G.degree(n, weight='weight')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    fig = plt.figure()

    values, base = np.histogram(data, bins=40)
    print(data)
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='skyblue')
    
    plt.title("Primary school degree distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()
    #fig.savefig("degree_distribution.png")

students = generate_students(10000, 100, 5)

cProfile.run('generate_network(students)')
heatmap(generate_network(students)) 