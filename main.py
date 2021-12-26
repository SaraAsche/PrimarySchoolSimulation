##################################################
##          Author: Sara Johanne Asche          ##
##          Date: December 2021                 ##
##          File: main.py                       ##
##          Description:                        ##
##################################################

#https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html -> header and functions format


#Importing packages
import random
random.seed(0)

import networkx as nx
from networkx.generators.small import house_graph
import seaborn as sns
import numpy as np
np.random.seed(0)

import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
import math as math
import scipy as sp
from scipy import stats
from matplotlib.colors import LogNorm
import functools
#import itertools
from itertools import combinations
#import enum
import cProfile


#Importing person and interaction objects from person.py
from person import Person, Interaction
#from enums import Grade, Age_group
#from layers import Grades, Klasse, Lunchbreak, Recess


# Generating students: Given #students in a school, #grades, #classes per grade and 
# a set threshold for individuals in class
def generate_students(num_students, num_grades, num_classes, class_treshold = 20):
    available_grades = [i for i in range(1, num_grades + 1)]
    available_classes = [chr(i) for i in range(97, 97 + num_classes)]

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
        # students[i].const_bias = 40*(math.log(1/random.random()))#pow(random.random(),exp) #powerlaw
        students[i].bias_vector = {}
        for j in range(len(students)):
            students[i].bias_vector[j] = students[i].const_bias

    return students

# Hourly generated network, interactions objects are used to create edges. Move generation of interactions to generate_interaction (add edges there). Then, simGrid is only made 1 time.
def generate_network(students):
    
    graph = nx.Graph()

    for student in students:
        graph.add_node(student)
    
    simGrid = generate_similarity_Grid(graph)
    
    for interaction in generate_interactions_for_network(students, simGrid):
        
        p1 = interaction.getp1()
        p2 = interaction.getp2()
        weight = interaction.getcount()

        graph.add_edge(p1, p2, count=weight)
    
    return graph

# Interaction objects are created
def generate_interactions_for_network(students, simGrid):
    
    interactions = []

    # for studs in combinations(students, 2):
    #     pers = studs[1]
    #     stud = studs[0]

    #     if pers.getID() in stud.find_all_interactions_for_person(pers, interactions):
    #         break
    #     weight = interaction_between_persons(stud, pers, simGrid)
    #     yield Interaction(stud, pers, weight)

    for i in range(len(students)):
        stud = students[i]
        for j  in range(i + 1,len(students)):
            pers = students[j]
            if stud.has_interacted_with(pers):
                break
            else:
                weight = interaction_between_persons(stud, pers, simGrid)
                if weight:
                    # interactions.append(Interaction(stud, pers, weight))
                    yield Interaction(stud, pers, weight)
    return interactions

def generate_similarity_Grid(network):
    n = network.number_of_nodes()
    #print(n)

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
    
    #p=p*p1.bias*p2.bias#*historie om disse to har interagert
    p=p*p1.bias_vector[p2.getID()]*p2.bias_vector[p1.getID()]
    if p1.id  == 0 and p2.id == 1:
        print(p)
        print(p1.const_bias)
        print(p1.bias_vector[p2.getID()])
    #p=similar*10;


    d = (2/3)*pow(10,-5.1) 
    # print(int(np.random.poisson(p*d)))
    return int(np.random.poisson(p*d))
    #return np.random.normal(p)
    #return np.random.uniform(0,p)

def renormalize(bias_vector, normTarget):
    oldMean = np.mean(list(bias_vector.values()))
    correction = normTarget/oldMean
    newVector = {}
    
    for i in bias_vector:
        newVector[i] = bias_vector[i]*correction
        
    return newVector

def generate_a_day(students, hourDay=8):
    for i in range(len(students)):
        students[i].bias = students[i].const_bias + 160*(math.log(1/random.random()))
        normTarget = students[i].bias
        students[i].bias_vector = renormalize(students[i].bias_vector, normTarget)
    
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

    k=0.5

    for i in range(len(students)):
        for j in range(len(students)):
            if i == j:
                continue
            students[i].bias_vector[j] += dayGraph[students[i]][students[j]]['count'] # evt  += dayGraph[i][j]['count']*stortNokTall
            students[i].bias_vector[j] -= k*(students[i].bias_vector[j]-students[i].bias)

    return dayGraph

def generateXdays(students, numDays):
    daily_list = []
    for i in range(numDays):
        daily_list.append(generate_a_day(students))
        #print(students[0].bias_vector)
        print("-----Day" + str(i)+"------")
        print("max: " + str(max(list(students[0].bias_vector.values()))))
        print("mean: " + str(np.mean(list(students[0].bias_vector.values()))))
        print("bias: " + str((students[0].bias)))
    
    # dayNumberX = daily_list[-1]

    return daily_list[2],daily_list[6]



######################## Analysis of functions ########################

#Only investigate class-class interactions or grade-grade interactions
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

#Investigate pearson correlation from day 1 to day 2 and plot the degree for each nodes on day 1 versus day 2
def plot_Correlation_between_Days(day1, day2):
    degday1 = [val for (node, val) in day1.degree(weight = 'count')]
    degday2 = [val for (node, val) in day2.degree(weight = 'count')]
    
    plt.scatter(degday1, degday2)
    print("Pearson correlation:")
    print(np.corrcoef(degday1, degday2))
    print(stats.pearsonr(degday1, degday2))
    plt.show()

#Draw the node + edge representation of the network. Axis can be used when plotting many graphs in the same plot
def displayNetwork(graph, axis=None):
    if axis:
        nx.draw(graph,ax=axis)
    else:
        nx.draw(graph)
        plt.show()

#Generate a heatmap of the adjacency matrix of a graph. Axis can be used when plotting many graphs in the same plot
def heatmap(graph, axis=None):
    
    A = nx.adjacency_matrix(graph, weight='count')
    A_M = A.todense()
    if axis:
        sns.heatmap(A_M, robust=True, ax=axis)
    else:
        sns.heatmap(A_M, robust=True)
        plt.show()

# Generates histogram of degree distribution
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

#Generates comulative distribution. For the project logX=False, logY=True for semilog
def histDistributionLog(graph, logX, logY, axis=None):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight='count')
        
        degs[n] = deg
    
    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])
    print(data)
    N = len(data)
    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    if axis:
        axis.plot(d.keys(), d.values())
    else:
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

def runAnalysis(graph):
    figure, axis = plt.subplots(2, 2)

    ######## Degree distribution ########
    print(axis)
    histDistributionLog(graph, False, True, axis[0,0])
    axis[0,0].set_title("Degree dist")

    ######## Heatmap ########
    heatmap(graph,axis[1,0])
    axis[1,0].set_title("Heatmap")
    

    ######## Weight Frequency of node 1 ########
    displayNetwork(graph,axis[0,1])
    axis[0,1].set_title("network")
    
    for ax in axis.flat:
        ## check if something was plotted 
        if not bool(ax.has_data()):
            figure.delaxes(ax) ## delete if nothing is plotted in the axes obj

    plt.show()

    return None


#######################################################################################
'''
day1=generate_a_day(students)
day2=generate_a_day(students)

heatmap(day1)
heatmap(day2)

histDistributionLog(day1, False,True)
histDistributionLog(day2, False, True)



plot_Correlation_between_Days(day1, day2)

l = generate_a_day(students) 
heatmap(l)
histDistributionLog(l, False, True)
'''
#classInt = createSubGraphWithoutGraph(l, True, True) 
#heatmap(classInt)
#histDistributionLog(classInt, False, True)

#ClassAndGrade = createSubGraphWithoutGraph(l, True, True)
#heatmap(ClassAndGrade)
#histDistributionLog(ClassAndGrade, False, True)


#histDistributionLog(classInt, True, False)
students = generate_students(225, 5, 2)



#runAnalysis(l)

day1 = generate_network(students)
heatmap(day1)
#heatmap(l)
#histDistributionLog(l, True, True)


#cProfile.run('generate_network(students)')
#heatmap(generate_network(students)) 