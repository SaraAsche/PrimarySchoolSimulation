import random
import itertools
import enum
import cProfile


import networkx as nx
from networkx.generators.small import house_graph
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
import math as math
import scipy as sp
from scipy import stats
from matplotlib.colors import LogNorm
import pickle
import functools

from person import Person, Interaction
from enums import Grade, Age_group
from layers import Grades, Klasse, Lunchbreak, Recess

def weightedFlip(p):
    return random.random() < p


def interaction_between_persons(p1, p2, simGrid):
    similarityList = simGrid[p1.getID()][p2.getID()]

    similar = len(similarityList)

    p = random.randint(0,1) #adding noise

    for element in similarityList: #add up
        if element == 'L':
            p+=random.randint(0,1)
        if element == 'G':
            p+=random.randint(0,3)
        if element == 'K':
            p+=random.randint(1,70)
    
    p=p*p1.bias*p2.bias

    #p=similar*10;

    d = (2/3)*pow(10,-5.1)

    return int(np.random.poisson(p*d))
    #return np.random.normal(p)
    #return np.random.uniform(0,p)

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
    for i in range(len(students)):
        students[i].constBias = 40*(math.log(1/random.random()))#pow(random.random(),exp) #powerlaw

    return students

def generate_network(students):
    
    graph = nx.Graph()

    for student in students:
        graph.add_node(student)

    # interactions = generate_interactions_for_network(students, graph)
    for i in range(len(students)):
        students[i].bias = students[i].constBias + 160*(math.log(1/random.random()))#pow(random.random(),exp) #powerlaw

    for interaction in generate_interactions_for_network(students, graph):
        
        p1 = interaction.getp1()
        p2 = interaction.getp2()
        weight = interaction.getcount()

        graph.add_edge(p1, p2, count=weight)
    
    
    return graph

def generate_interactions_for_network(students, network):
    interactions = []

    checked_students = students.copy()

    simGrid = generate_similarity_Grid(network)

    #exp = -2.2


    for i in range(len(students)):
        stud = students[i]
        for j  in range(i + 1,len(checked_students)):
            pers = checked_students[j]
            if pers.getID() in stud.find_all_interactions_for_person(pers, interactions):
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

    simGrid = [['' for _ in range(n)] for _ in range(n)]

    similar = ['S'] #same school

    for stud1 in network.nodes:
        i = stud1.getID()
        for stud2 in network.nodes:
            j = stud2.getID()
            if stud1.getClass() == stud2.getClass() and stud1.getGrade() == stud2.getGrade():
                similar.append('K')
            if stud1.getGrade() == stud2.getGrade():
                similar.append('G')
            if stud1.getLunchgroup() == stud2.getLunchgroup():
                similar.append('L')
            
            simGrid[i][j] = similar

            similar = ['S']

    return simGrid

#### Can and will be moved to another file as it is analysis and not a part of the document ####

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
        deg = graph.degree(n, weight='count')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    plt.hist(data, bins=10, color='skyblue', ec = 'black') #col = 'skyblue for day2, mediumseagreen for day1
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def plot_degree_distribution(G): #Not in use anymore
    degs = {}
    for n in G.nodes ():
        deg = G.degree(n, weight='count')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    fig = plt.figure()

    values, base = np.histogram(data, bins=40)
   
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='skyblue')
    
    plt.title("Primary school degree distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()
    #fig.savefig("degree_distribution.png")

def histDistributionLog(graph, logX, logY): #Cumulative distribution
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight='count')
        
        degs[n] = deg
        #degs[n] = 1-np.log(deg)
    
    def sort_rules(x, y):
        if x[0].getID() > y[0].getID():
            return 1
        elif x[0].getID() < y[0].getID():
            return -1
        return 0

    items = sorted(degs.items(), key=functools.cmp_to_key(sort_rules))
    
    data = []
    for line in items:
        data.append(line[1])
    print(data)
    N = len(data)
    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    plt.plot(d.keys(), d.values())

    if logY:
        plt.yscale('log') 
        plt.ylabel('Normalised log frequency')
    else:
        plt.yscale('linear')
        plt.ylabel('Frequency')

    if logX:
        plt.xscale('log') 
        plt.xlabel('log Degree')
    else:
        plt.xscale('linear')
        plt.xlabel('Degree')

    #plt.plot(x,y, color = 'skyblue')
    plt.show()

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
        cumul -= float(dictHist[i])/float(n)
    return cHist

def generate_a_day(students, hourDay=8):
    hourly_list = []
    for i in range(hourDay):
        hourly_list.append(generate_network(students))
    
    dayGraph = nx.empty_graph(hourly_list[0])
    # dayGraph.add_edges_from(hourly_list[0].edges(data=True)+hourly_list[1].edges(data=True))

    edges = []
    for i in range(len(students)):
        first_id = students[i]
        for j in range(i+1, len(students)):
            second_id = students[j]
            count = 0
            for graph in hourly_list:
                data = graph.get_edge_data(first_id, second_id)
                if data is not None:
                    count += data['count']
            edges.append((first_id, second_id, {'count': count}))
    
    dayGraph.add_edges_from(edges)         

    return dayGraph

students = generate_students(225, 5, 2)

#cProfile.run('generate_network(students)')
#heatmap(generate_network(students)) 

#Class and grade interaction objects
def createSubGraphWithoutGraph(graph, diagonal, gradeInteraction):  #objektene er ikke konservert med sine atributter

    G = nx.Graph()
    
    for node in graph:
        klasseAttr = node.getClass()
        for n in graph.neighbors(node):
            if diagonal:
                if (n.getClass() == klasseAttr and node.getGrade() == n.getGrade()):
                    G.add_edge(node, n, count = graph.get_edge_data(node, n)['count'])
                    G.add_node(n)
            if gradeInteraction:
                if (n.getGrade()==node.getGrade and node.getClass() != n.getClass()):
                    G.add_edge(node, n, count = graph.get_edge_data(node, n)['count'])
                    G.add_node(n)
    return G

def plot_Correlation_between_Days(day1, day2):
    degday1 = [val for (node, val) in day1.degree(weight = 'count')]
    degday2 = [val for (node, val) in day2.degree(weight = 'count')]
    
    plt.scatter(degday1, degday2)
    print("Pearson correlation:")
    print(np.corrcoef(degday1, degday2))
    print(stats.pearsonr(degday1, degday2))
    plt.show()



#Lunch (720-820). 1 hour from hour 3-4
#Entire network

#######################################################################################
'''
day1=generate_a_day(students)
day2=generate_a_day(students)

heatmap(day1)
heatmap(day2)

histDistributionLog(day1, False,True)
histDistributionLog(day2, False, True)



plot_Correlation_between_Days(day1, day2)
'''
l = generate_a_day(students) 
heatmap(l)
histDistributionLog(l, False, True)
#classInt = createSubGraphWithoutGraph(l, True, True) 
#heatmap(classInt)
#histDistributionLog(classInt, False, True)

#ClassAndGrade = createSubGraphWithoutGraph(l, True, True)
#heatmap(ClassAndGrade)
#histDistributionLog(ClassAndGrade, False, True)


#histDistributionLog(classInt, True, False)




