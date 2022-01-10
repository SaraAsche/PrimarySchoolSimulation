import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

class Analysis:
    def __init__(self, network) -> None:
        self.network = network

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
                    if (n.getGrade()==node.getGrade and node.getClass() != n.getClass()):
                        G.add_edge(node, n, count = graph.get_edge_data(node, n)['count'])
                        G.add_node(n)
        return G

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
    def heatmap(self, day, axis=None):

        A = nx.adjacency_matrix(day, weight='count')
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