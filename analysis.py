"""Analysis class: Runs analysis on Network objects

Uses statistics and plotting to run analysis on network objects and compare
the results to experimental data. 

Typical usage example:

  analysis = Analysis(graph)
  analysis.pixel_dist_school(self, graph, old=False, both=False)

Author: Sara Johanne Asche
Date: 14.02.2022
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy import stats


class Analysis:
    """A class to analyse Network objects

    Longer class information...

    Attributes
    ----------
    network : Network
        Network object containing a NetworkX graph

    Method
    ------
    createSubGraphWithout(self, graph, grade, class_interaction)
        Returns a subgraph of the original graph where grade and/or class
        interactions are either removed or kept.
    createSubGraphWithoutGraph(self, graph, diagonal, grade_interaction)
        Returns a subgraph of the original graph where off-diagonal
        and/or grade interactions are removed or kept.
    pixelDist(self, graph, logX, logY, axis=None, old=False, label=None)
        Plots the distribution of all interactions that has occured
        between two individuals.
    pickleLoad(self, name)
        Loads a pickle-file with the name name.
    pixel_dist_school(self, graph, old=False, both=False)
        Plots a 2x2 pixel distribution subplot with whole graph, off-diagonal
        grade-grade interactions and class-class interactions
    plot_Correlation_between_Days(self, day1, day2)
        Plots the linear correlation between the amount of interactions a
        specific indicidual has had on the two graphs day1 and day2.
    displayNetwork(self, graph, axis=None)
        Plots a NetworkX graph with nodes and edges.
    heatmap(self, day, output=False, axis=None)
        Plots the interaction heatmap of the adjacency matrix of a NetworkX graph
    histDistribution(self, graph)
        Plots a histogram of the degree distribution of a graph
    toCumulative(self, l)
        Returns a dictionary where the degree of nodes is the values whilst their frequency
        is the frequency of that specific node.
    histDistributionLog(self, graph, logX=False, logY=True, axis=None, old=False, label=None)
        Plots the cumulative degree P(X>=x), where x is the frequency distribution of a given graph.
    histPlot(self, d, label, logX=False, logY=True, axis=None)
        Plots a dictionary d with a label on a given matplotlib axis
    plotDegreeDistSubGraphs(self, both=False, experimental=False, sim=None)
        Plots the degree distribution of both experimental and simulated graphs for the whole graph
        and the subgraphs off-diagonal, grade-grade and class-class.
    runAnalysis(self, graph)
        Plots a subplot with degree distribution, heatmap and drawn network of a graph.
    """

    def __init__(self, network) -> None:
        """Inits Analysis object with network parameter

        Parameters
        ----------
        network: Network
            A Network object describing interactions at a primary school
        """

        self.network = network

    def createSubGraphWithout(self, graph, grade, class_interaction):
        """Creates a subgraph with the exclusion/inclusion of grade and class interactions

        Loops through all nodes in a graph and filters out the interactions ones to add to a new subgraph based
        on the bool variables grade and class_interaction.

        Parameters
        ----------
        graph : NetworkX
            NetworkX object that contains interactions at a primary school between Person objects
        grade : bool
            Bool that is True if interactions between Person objects of the same grade is included. False otherwise
        class_interaction : bool
            Bool that is True if interactions between Person objects of the same class is included. False otherwise
        """

        G = nx.Graph()

        for node in graph:
            for n in graph.neighbors(node):
                if grade and not class_interaction:
                    if node.getGrade() != n.getGrade():
                        G.add_edge(node, n, count=graph[node][n]["count"])
                elif not grade:
                    if node.get_class_and_grade() != n.get_class_and_grade():
                        G.add_edge(node, n, count=graph[node][n]["count"])
        return G

    def createSubGraphWithoutGraph(self, graph, diagonal, grade_interaction):
        """Creates a subgraph with the exclusion/inclusion of the off-diagonal and grade interactions

        Loops through all nodes in a graph and filters out the desired interactions to add to a new subgraph based
        on the bool variables off_diagonal and grade_interaction.

        Parameters
        ----------
        graph : NetworkX
            NetworkX object that contains interactions at a primary school between Person objects
        diagonal : bool
            Bool that is True if interactions between Person objects on the off-diagonal is included. False otherwise
        grade_interaction : bool
            Bool that is True if interactions between Person objects of the same grade is included. False otherwise
        """

        G = nx.Graph()

        for node in graph:
            klasseAttr = node.getClass()
            for n in graph.neighbors(node):
                if diagonal:
                    if n.getClass() == klasseAttr and node.getGrade() == n.getGrade():
                        G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                        G.add_node(n)
                if grade_interaction:
                    if n.getGrade() == node.getGrade() and node.getClass() != n.getClass():
                        G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                        G.add_node(n)
        return G

    def pixelDist(self, graph, logX, logY, axis=None, old=False, label=None):
        """Plots the distribution of all interactions that has occured between two individuals.

        Parameters
        ----------
        graph : Network or dictionary
            NetworkX object that contains interactions at a primary school between Person objects.
            If old = True the graph input is a dictionary
        logX : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        logY : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        old : bool
            bool that determines whether the graph is experimental for True or simulated for False.
            Default is False.
        label : str
            A string value that can be passed as a label for the plotting using axis. Default is None
        """

        if not old:
            ## The graph is a networkX object and must be made into an adjacency matrix and made cumulative
            A = self.heatmap(graph, output=True)

            length = len(graph.nodes())

            weights = A[np.triu_indices(length, k=1)].tolist()[0]

            data = sorted(weights)

            sorteddata = np.sort(data)
            d = self.toCumulative(sorteddata)

        ## The graph is a dictionary made from toCumulative of the experimental data and can be used as is.
        if old:
            d = graph

        ## If a subplot is being made, axis is True. Then labels can be added and the plots are plotted on the axis.
        if axis:
            if old:
                ## If old, the label is specified by the input variable label. Label can be "Day 1" or "Day 2"
                axis.plot(d.keys(), d.values(), label=label)
            else:
                ## If not old, the label is simulated
                axis.plot(d.keys(), d.values(), label="Simulated")

            ## Then the x-axis and y-axis are scaled according to the logX and logY variables.
            if logX:
                axis.set_xscale("log")
                axis.set_xlabel("log Interactions")
            else:
                axis.set_xscale("linear")
                axis.set_xlabel("Interactions")
            if logY:
                axis.set_yscale("log")
                axis.set_ylabel("Normalised log frequency")
            else:
                axis.set_yscale("linear")
                axis.set_ylabel("Frequency")

        else:
            ## A single plot can be plotted by matplotlib, no axis necessary
            plt.plot(d.keys(), d.values())

            ## The x-axis and y-axis are scaled according to the logX and logY variables.
            if logY:
                plt.yscale("log")
                plt.ylabel("Normalised log frequency")
            else:
                plt.yscale("linear")
                plt.ylabel("Frequency")
            if logX:
                plt.xscale("log")
                plt.xlabel("log Interactions")
            else:
                plt.xscale("linear")
                plt.xlabel("Interactions")

            plt.show()

    def pickleLoad(self, name):
        file_to_read = open(name, "rb")
        return pickle.load(file_to_read)

    def pixel_dist_school(self, graph, old=False, both=False):
        """Plots the pixel distribution in a 2x2 plot with whole graph, off-diagonal, grade-grade and class-class interactions

        Parameters
        ----------
        graph : Network
            NetworkX object that contains interactions at a primary school between Person objects
        old : bool
            bool that determines if day 1 experimental values should be added, True if it should be added,
            False if not.
        both : bool
            bool that determines if both day 1 and day 2experimental values should be added, True if it
            should be added, False if not.
        """

        ## Generating off-diagonal, grade-grade and class-class for the simulated network
        off_diagonal = self.createSubGraphWithout(graph, True, False)
        grade_grade = self.createSubGraphWithoutGraph(graph, False, True)
        class_class = self.createSubGraphWithoutGraph(graph, True, False)

        ## Generating off-diagonal, grade-grade and class-class for day 1 experimental network
        if old:
            old_graph = self.pickleLoad("graph1_whole_pixel.pkl")
            old_off_diagonal = self.pickleLoad("graph1_off_diag_pixel.pkl")
            old_grade = self.pickleLoad("graph1_grade_pixel.pkl")
            old_class = self.pickleLoad("graph1_class_pixel.pkl")

        ## Generating off-diagonal, grade-grade and class-class for day 2 experimental network
        if both:
            old_graph2 = self.pickleLoad("graph2_whole_pixel.pkl")
            old_off_diagonal2 = self.pickleLoad("graph2_off_diag_pixel.pkl")
            old_grade2 = self.pickleLoad("graph2_grade_pixel.pkl")
            old_class2 = self.pickleLoad("graph2_class_pixel.pkl")

        ## Generate 2x2 subplot
        figure, axis = plt.subplots(2, 2, figsize=(8, 8))
        figure.tight_layout(pad=3)

        ## Setting up wether or not the x and y axis are log10 scale
        logx = True
        logy = True

        ## Adding the whole graph pixel distribution to the [0,0] position of the graph
        self.pixelDist(graph, logx, logy, axis[0, 0])
        if old:
            self.pixelDist(old_graph, logx, logy, axis[0, 0], old=True, label="Day 1")
        if both:
            self.pixelDist(old_graph2, logx, logy, axis[0, 0], old=True, label="Day 2")
        axis[0, 0].set_title("Whole network")

        ## Adding the off diagonal pixel distribution to the [1,0] position of the graph
        self.pixelDist(off_diagonal, logx, logy, axis[1, 0])
        if old:
            self.pixelDist(old_off_diagonal, logx, logy, axis[1, 0], old=True, label="Day 1")
        if both:
            self.pixelDist(old_off_diagonal2, logx, logy, axis[1, 0], old=True, label="Day 2")
        axis[1, 0].set_title("Off-diagonal")

        ## Adding the grade_grade pixel distribution to the [0,1] position of the graph
        self.pixelDist(grade_grade, logx, logy, axis[0, 1])
        if old:
            self.pixelDist(old_grade, logx, logy, axis[0, 1], old=True, label="Day 1")
        if both:
            self.pixelDist(old_grade2, logx, logy, axis[0, 1], old=True, label="Day 2")
        axis[0, 1].set_title("grade-grade")

        ## Adding the class_class pixel distribution to the [1,1] position on the graph
        self.pixelDist(class_class, logx, logy, axis[1, 1])
        if old:
            self.pixelDist(old_class, logx, logy, axis[1, 1], old=True, label="Day 1")
        if both:
            self.pixelDist(old_class2, logx, logy, axis[1, 1], old=True, label="Day 2")
        axis[1, 1].set_title("class-class")

        ## Adding a combined label according to the labels specified when plotting
        handles, labels = axis[1, 1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center")

        plt.savefig("pixelDistSimulated.png", bbox_inches="tight", dpi=150)

        plt.show()

    def plot_Correlation_between_Days(self, day1, day2):
        """Plots the linear correlation between the amount of interactions a specific indicidual
        has had on the two graphs day1 and day2.

        Parameters
        ----------
        day1 : Network
            NetworkX object that contains interactions at a primary school between Person objects on day 1
        day2 : Network
            NetworkX object that contains interactions at a primary school between Person objects on day 2
        """
        ##Creates two lists of all the degrees on the two days
        degday1 = [val for (node, val) in self.network.daily_list[day1].degree(weight="count")]
        degday2 = [val for (node, val) in self.network.daily_list[day2].degree(weight="count")]

        ## Scatter the degree each node has against each other and observe the correlation
        plt.scatter(degday1, degday2)
        print("Pearson correlation:")
        print(np.corrcoef(degday1, degday2))
        print(stats.pearsonr(degday1, degday2))
        plt.show()

    def displayNetwork(self, graph, axis=None):
        """Plots a NetworkX graph with nodes and edges.

        Parameters
        ----------
        graph : Network
            NetworkX object that contains interactions at a primary school between Person objects
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        """

        if axis:
            nx.draw(graph, ax=axis)
        else:
            nx.draw(graph)
            plt.show()

    def heatmap(self, day, output=False, axis=None):
        """Plots the interaction heatmap of the adjacency matrix of a NetworkX graph

        Parameters
        ----------
        day : Network
            NetworkX object that contains interactions at a primary school between Person objects
        output : bool
            bool variable that is True if the function should return a numpy adjacencymatric and False
            if the function should draw a heatmap.
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        """

        A = nx.adjacency_matrix(day, nodelist=sorted(day.nodes), weight="count")

        ## If output=True, there is no need to draw the heatmap
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
        """Plots a histogram of the degree distribution of a graph

        Parameters
        ----------
        graph : Network
            NetworkX object that contains interactions at a primary school between Person objects
        """

        ## Creates an empty dictionary where the degree of each node is added to a dictionary with the key being n (0-len(graph.nodes))
        degs = {}
        for n in graph.nodes():
            deg = graph.degree(n, weight="count")
            degs[n] = deg

        ## Sort the dictionary values and keys based on weight
        items = sorted(degs.items())

        ## Extract the values (weights)
        data = []
        for line in items:
            data.append(line[1])

        ## Plot the values in a histogram
        plt.hist(data, bins=10, color="skyblue", ec="black")  # col = 'skyblue for day2, mediumseagreen for day1
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

    def toCumulative(self, l):
        """Returns a dictionary where the degree of nodes is the values whilst their frequency
        is the frequency of that specific node.

        Parameters
        ----------
        l : list
            list of degrees a specific graph has
        """

        n = len(l)
        dictHist = {}

        ## Adds to the frequency is the same degree is found multiple times
        for i in l:
            if i not in dictHist:
                dictHist[i] = 1
            else:
                dictHist[i] += 1
        cHist = {}
        cumul = 1

        ## Normalise the frequency
        for i in dictHist:
            cHist[i] = cumul
            cumul -= float(dictHist[i]) / float(n)
        return cHist

    def histDistributionLog(self, graph, logX=False, logY=True, axis=None, old=False, label=None):
        """Plots the cumulative degree P(X>=x), where x is the frequency distribution of a given graph.

        Parameters
        ----------
        graph : Network
            NetworkX object that contains interactions at a primary school between Person objects
        logX : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        logY : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        old : bool
            bool that determines whether the graph is experimental for True or simulated for False.
            Default is False.
        label : str
            A string value that can be passed as a label for the plotting using axis. Default is None
        """

        ## Extract all degrees from the nodes of the graph
        degs = {}
        for n in graph.nodes():
            deg = graph.degree(n, weight="count")

            degs[n] = deg

        items = sorted(degs.items())

        ## Extract degrees from the dictionary
        data = []
        for line in items:
            data.append(line[1])
        N = len(data)
        sorteddata = np.sort(data)

        d = self.toCumulative(sorteddata)

        ##Load in degreedistributions from experimental data
        old1 = self.pickleLoad("Degreedistribution_Day1.pkl")
        old2 = self.pickleLoad("Degreedistribution_Day2.pkl")

        ##If subplot is created label can be added
        if axis:
            if label:
                axis.plot(d.keys(), d.values(), label=label)
            else:
                axis.plot(d.keys(), d.values())

        ## Else creates normal matplotlib object for a single plot
        else:
            plt.plot(d.keys(), d.values(), label="Simulated")
            plt.plot(old1.keys(), old1.values(), label="Empiric day 1")
            plt.plot(old2.keys(), old2.values(), label="Empiric day 2")

            if logY:
                plt.yscale("log")
                plt.ylabel("Normalised log frequency")
            else:
                plt.yscale("linear")
                plt.ylabel("Frequency")

            if logX:
                plt.xscale("log")
                plt.xlabel("log Degree")
            else:
                plt.xscale("linear")
                plt.xlabel("Degree")

            plt.show()

    def histPlot(self, d, label, logX=False, logY=True, axis=None):
        """Plots a dictionary d with a label on a given matplotlib axis
        Parameters
        ----------
        d : dictionary
            Dictionary containing frequency of a number as key and the number as key
        logX : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        logY : bool
            bool value that determins whether the x-axis will be log10 for True or normal for False
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        """

        if axis:
            axis.plot(d.keys(), d.values(), label=label)

    def plotDegreeDistSubGraphs(self, both=False, experimental=False, sim=None):
        """Plots the degree distribution of both experimental and simulated graphs for the whole graph
         and the subgraphs off-diagonal, grade-grade and class-class.


        Parameters
        ----------
         both : bool
             bool that determines if both day 1 and day 2 experimental values should be added, True if it
             should be added, False if not. Default is False
         experimental : bool
             True if simulated data should be added to the plot. False otherwise. Default is False.
         sim : Network
             NetworkX object that contains interactions at a primary school between Person objects. Default is None
        """

        ## Load day 1 dictionaries
        graph1 = self.pickleLoad("DegreeDictwhole1.pkl")
        off_diag1 = self.pickleLoad("DegreeDictOffDiag1.pkl")
        grade1 = self.pickleLoad("DegreeDictgrade1.pkl")
        class1 = self.pickleLoad("DegreeDictclass1.pkl")

        ## Load day 2 dictionaries
        if both:
            graph2 = self.pickleLoad("DegreeDictwhole2.pkl")
            off_diag2 = self.pickleLoad("DegreeDictOffDiag2.pkl")
            grade2 = self.pickleLoad("DegreeDictgrade2.pkl")
            class2 = self.pickleLoad("DegreeDictclass2.pkl")

        ## load experimental graphs
        if experimental:
            graph = sim
            off_diagE = self.createSubGraphWithout(graph, True, False)
            gradeE = self.createSubGraphWithoutGraph(graph, False, True)
            classE = self.createSubGraphWithoutGraph(graph, True, False)

        ## Create figure and axis objects for subplot
        figure, axis = plt.subplots(2, 2, figsize=(10, 8))
        figure.tight_layout(pad=4)

        ## Adding the whole graph degree distribution to the [0,0] position of the graph
        self.histPlot(graph1, logX=False, logY=True, axis=axis[0, 0], label="Day 1")
        if both:
            self.histPlot(graph2, logX=False, logY=True, axis=axis[0, 0], label="Day 2")
        if experimental:
            self.histDistributionLog(graph, False, True, axis=axis[0, 0], label="Simulated")
        axis[0, 0].set_title("Whole graph")

        ## Adding the off diagonal degree distribution to the [1,0] position of the graph
        self.histPlot(off_diag1, logX=False, logY=True, axis=axis[1, 0], label="Day 1")
        if both:
            self.histPlot(off_diag2, logX=False, logY=True, axis=axis[1, 0], label="Day 2")
        if experimental:
            self.histDistributionLog(off_diagE, logX=False, logY=True, axis=axis[1, 0], label="Simulated")
        axis[1, 0].set_title("Off diagonal")

        ## Adding the grade-grade degree distribution to the [0,1] position of the graph
        self.histPlot(grade1, logX=False, logY=True, axis=axis[0, 1], label="Day 1")
        if both:
            self.histPlot(grade2, logX=False, logY=True, axis=axis[0, 1], label="Day 2")
        if experimental:
            self.histDistributionLog(gradeE, logX=False, logY=True, axis=axis[0, 1], label="Simulated")
        axis[0, 1].set_title("Grade")

        ## Adding the class-class degree distribution to the [1,1] position of the graph
        self.histPlot(class1, logX=False, logY=True, axis=axis[1, 1], label="Day 1")
        if both:
            self.histPlot(class2, logX=False, logY=True, axis=axis[1, 1], label="Day 2")
        if experimental:
            self.histDistributionLog(classE, logX=False, logY=True, axis=axis[1, 1], label="Simulated")
        axis[1, 1].set_title("Class")

        ## Adding a combined label according to the labels specified when plotting
        handles, labels = axis[1, 1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="center")  # loc="lower center"

        plt.show()

    def runAnalysis(self, graph):
        """Plots a subplot with degree distribution, heatmap and drawn network of a graph

        Parameters
        ----------
        graph : Network
            NetworkX object that contains interactions at a primary school between Person objects.
        """
        figure, axis = plt.subplots(2, 2)

        ## Degree distribution
        self.histDistributionLog(graph, False, True, axis[0, 0])
        axis[0, 0].set_title("Degree dist")

        ## Heatmap
        self.heatmap(graph, axis[1, 0])
        axis[1, 0].set_title("Heatmap")

        ## Display of the network
        self.displayNetwork(graph, axis[0, 1])
        axis[0, 1].set_title("network")

        for ax in axis.flat:
            ## check if something was plotted
            if not bool(ax.has_data()):
                figure.delaxes(ax)  ## delete if nothing is plotted in the axes obj

        plt.show()

        return None
