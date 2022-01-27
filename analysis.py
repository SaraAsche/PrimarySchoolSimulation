from cProfile import label
import pickletools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy import stats

class Analysis:
    def __init__(self, network) -> None:
        self.network = network
    
    def createSubGraphWithout(self, graph, grade, klasse):
        
        # grades = nx.get_node_attributes(graph, 'klasse')
        
        G = nx.Graph()

        # nodes = set()
        # edges = []
        # for node, klasseAttr in grades.items():
        #     print(node)
        #     print(klasseAttr)
        #     for n in graph.neighbors(node):
        #         if grade and (not klasse):
        #             if grades[n][0] != grades[node][0]:
        #                 G.add_edge(node, n, weight=graph[node][n]['count'])
        #                 G.add_node(n, klasse=graph.nodes[n]['klasse'])
        #         elif not grade:
        #             if grades[n] != grades[node]:
        #                 G.add_edge(node, n, weight=graph[node][n]['count'])
        #                 G.add_node(n, klasse=graph.nodes[n]['klasse'])
        # self.heatmap(G)
        # return G

        for node in graph:
            for n in graph.neighbors(node):
                if grade and not klasse:
                    if node.getGrade() != n.getGrade():
                        G.add_edge(node, n, count=graph[node][n]['count'])
                elif not grade:
                    if node.get_class_and_grade() != n.get_class_and_grade():
                        G.add_edge(node, n, count=graph[node][n]['count'])
        #self.heatmap(G)
        return G

    def createSubGraphWithoutGraph(self, graph, diagonal, gradeInteraction):  #objektene er ikke konservert med sine atributter

        G = nx.Graph()
        
        for node in graph:
            klasseAttr = node.getClass()
            for n in graph.neighbors(node):
                if diagonal:
                    if (n.getClass() == klasseAttr and node.getGrade() == n.getGrade()):
                        G.add_edge(node, n, count = graph.get_edge_data(node, n)['count'])
                        G.add_node(n)
                if gradeInteraction:
                    if (n.getGrade()==node.getGrade() and node.getClass() != n.getClass()):
                        G.add_edge(node, n, count = graph.get_edge_data(node, n)['count'])
                        G.add_node(n)
        self.heatmap(G)
        return G

    def pixelDist(self, graph, logX, logY, axis = None, old= False):
        if not old: 
            A = self.heatmap(graph, output = True)
            #print(A[np.triu_indices(236, k = 1)])
            length = len(graph.nodes())

            weights = A[np.triu_indices(length, k = 1)].tolist()[0]

            data = sorted(weights)
            
            sorteddata = np.sort(data)
            d = self.toCumulative(sorteddata)
        
        if old: 
            d = graph

        #print(float(sum(sorteddata))/float(len(sorteddata)))

        if axis:
            if old:
                axis.plot(d.keys(), d.values(), label='Experimental')
            else:
                axis.plot(d.keys(), d.values(), label='Simulated')

            if logX:
                axis.set_xscale('log')
                axis.set_xlabel('log Degree')
            else:
                axis.set_xscale('linear')
                axis.set_xlabel('Degree')
            if logY:
                axis.set_yscale('log')
                axis.set_ylabel('Normalised log frequency')
            else:
                axis.set_yscale('linear')
                axis.set_ylabel('Frequency')
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
        
        # print(np.triu(A.todense(), k=1)) # 

    def pickleLoad(self,name):
        file_to_read = open(name, 'rb')
        return pickle.load(file_to_read)
    
    def pixel_dist_school(self, graph, old = False):
        off_diagonal = self.createSubGraphWithout(graph, True, False)
        grade_grade = self.createSubGraphWithoutGraph(graph, False, True)
        class_class = self.createSubGraphWithoutGraph(graph,True, False)

        if old:
            old_graph = self.pickleLoad('graph1_whole_pixel.pkl')
            old_off_diagonal = self.pickleLoad('graph1_off_diag.pkl')
            old_grade = self.pickleLoad('graph1_grade_pixel.pkl')
            old_class = self.pickleLoad('graph1_class_pixel.pkl')


        figure, axis = plt.subplots(2, 2, figsize=(8,8))

        self.pixelDist(graph, False, True, axis[0,0])
        self.pixelDist(old_graph, False, True,axis[0,0],old=True)
        axis[0,0].set_title('Whole network')
        self.pixelDist(off_diagonal,False, True, axis[1,0])
        self.pixelDist(old_off_diagonal, False, True,axis[1,0],old=True)
        axis[1,0].set_title('Off-diagonal')
        self.pixelDist(grade_grade,False, True, axis[0,1])
        self.pixelDist(old_grade, False, True,axis[0,1],old=True)
        axis[0,1].set_title('grade-grade')
        self.pixelDist(class_class,False, True, axis[1,1])
        self.pixelDist(old_class, False, True,axis[1,1],old=True)
        axis[1,1].set_title('class-class')

        handles, labels = axis[1,1].get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper center')

        plt.savefig('pixelDistSimulated.png', bbox_inches='tight', dpi=150)

        plt.show()

    #Investigate pearson correlation from day 1 to day 2 and plot the degree for each nodes on day 1 versus day 2
    def plot_Correlation_between_Days(self, day1, day2):
        
        degday1 = [val for (node, val) in self.network.daily_list[day1].degree(weight = 'count')]
        degday2 = [val for (node, val) in self.network.daily_list[day2].degree(weight = 'count')]

        plt.scatter(degday1, degday2)
        print("Pearson correlation:")
        print(np.corrcoef(degday1, degday2))
        print(stats.pearsonr(degday1, degday2))
        plt.show()

    #Draw the node + edge representation of the network. Axis can be used when plotting many graphs in the same plot
    def displayNetwork(self, graph, axis=None):
        if axis:
            nx.draw(graph,ax=axis)
        else:
            nx.draw(graph)
            plt.show()

    #Generate a heatmap of the adjacency matrix of a graph. Axis can be used when plotting many graphs in the same plot
    def heatmap(self, day, output=False, axis=None): 
        A = nx.adjacency_matrix(day, nodelist=sorted(day.nodes), weight='count')

        if output:
            return A

        A_M = A.todense()
        if axis:
            sns.heatmap(A_M, robust=False, ax=axis)
        else:
            sns.heatmap(A_M, robust=False)
            plt.show()

    # Generates histogram of degree distribution
    def histDistribution(self, graph):
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

    def toCumulative(self,l):                                                                                               
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

    #Generates comulative distribution. For the project logX=False, logY=True for semilog
    def histDistributionLog(self, graph, logX=False, logY=True, axis=None):

        degs = {}
        for n in graph.nodes():
            deg = graph.degree(n, weight='count')
            
            degs[n] = deg
        
        items = sorted(degs.items())
        
        data = []
        for line in items:
            data.append(line[1])
        N = len(data)
        sorteddata = np.sort(data)

        d = self.toCumulative(sorteddata)

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

    
    def runAnalysis(self, graph):
        figure, axis = plt.subplots(2, 2)

        ######## Degree distribution ########
        self.histDistributionLog(graph, False, True, axis[0,0])
        axis[0,0].set_title("Degree dist")

        ######## Heatmap ########
        self.heatmap(graph,axis[1,0])
        axis[1,0].set_title("Heatmap")
        

        ######## Weight Frequency of node 1 ########
        self.displayNetwork(graph,axis[0,1])
        axis[0,1].set_title("network")
        
        for ax in axis.flat:
            ## check if something was plotted 
            if not bool(ax.has_data()):
                figure.delaxes(ax) ## delete if nothing is plotted in the axes obj

        plt.show()

        return None