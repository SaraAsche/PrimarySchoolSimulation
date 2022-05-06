"""Analysis class: Runs analysis on Network objects

Uses statistics and plotting to run analysis on network objects and compare
the results to experimental data. 

Typical usage example:

  analysis = Analysis(graph)
  analysis.pixel_dist_school(self, graph, old=False, both=False)
  analysis.degree_distribution_layers(self, both=False, experimental=False, sim=None)

Author: Sara Johanne Asche
Date: 14.02.2022
File: analysis.py

"""


import functools
import os
import networkx as nx
from networkx.algorithms import community
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy import stats


class Analysis:
    """A class to analyse Network objects

    Longer class information.

    Attributes
    ----------
    network : nx.Graph
        Network object containing a nx.GraphX graph

    Methods
    ------
    create_sub_graph_off_diagonal(self, graph, grade, class_interaction)
        Returns a subgraph of the original graph where grade and/or class
        interactions are either removed or kept.
    create_sub_graph_grade_class(self, graph, diagonal, grade_interaction)
        Returns a subgraph of the original graph where off-diagonal
        and/or grade interactions are removed or kept.
    pixel_dist(self, graph, logX, logY, axis=None, old=False, label=None)
        Plots the distribution of all interactions that has occured
        between two individuals.
    pickle_load(self, name)
        Loads a pickle-file with the name name.
    pixel_dist_school(self, graph, old=False, both=False)
        Plots a 2x2 pixel distribution subplot with whole graph, off-diagonal
        grade-grade interactions and class-class interactions
    plot_correlation_between_days(self, day1, day2)
        Plots the linear correlation between the amount of interactions a
        specific indicidual has had on the two graphs day1 and day2.
    display_network(self, graph, axis=None)
        Plots a Network graph with nodes and edges.
    heatmap(self, day, output=False, axis=None)
        Plots the interaction heatmap of the adjacency matrix of a Network graph
    hist_distribution(self, graph)
        Plots a histogram of the degree distribution of a graph
    to_cumulative(self, l)
        Returns a dictionary where the degree of nodes is the values whilst their frequency
        is the frequency of that specific node.
    cumulative_distribution_log(self, graph, logX=False, logY=True, axis=None, old=False, label=None)
        Plots the cumulative degree P(X>=x), where x is the frequency distribution of a given graph.
    hist_plot(self, d, label, logX=False, logY=True, axis=None)
        Plots a dictionary d with a label on a given matplotlib axis
    degree_distribution_layers(self, both=False, experimental=False, sim=None)
        Plots the degree distribution of both experimental and simulated graphs for the whole graph
        and the subgraphs off-diagonal, grade-grade and class-class.
    run_analysis(self, graph)
        Plots a subplot with degree distribution, heatmap and drawn network of a nx.Graph.
    """

    def __init__(self, network) -> None:
        """Inits Analysis object with network parameter

        Parameters
        ----------
        network: nx.Graph
            A nx.Graph object describing interactions at a primary school
        """

        self.network = network

    def create_sub_graph_off_diagonal(self, graph, grade, class_interaction) -> nx.Graph:
        """Creates a subgraph with the exclusion/inclusion of grade and class interactions

        Loops through all nodes in a graph and filters out the interactions ones to add to a new subgraph based
        on the bool variables grade and class_interaction.

        Parameters
        ----------
        graph : nx.GraphX
            nx.GraphX object that contains interactions at a primary school between Person objects
        grade : bool
            Bool that is True if interactions between Person objects of the same grade is included. False otherwise
        class_interaction : bool
            Bool that is True if interactions between Person objects of the same class is included. False otherwise
        """

        G = nx.Graph()

        for node in graph:
            for n in graph.neighbors(node):
                if grade and not class_interaction:
                    if node.get_grade() != n.get_grade():
                        G.add_edge(node, n, count=graph[node][n]["count"])
                elif not grade:
                    if node.get_class_and_grade() != n.get_class_and_grade():
                        G.add_edge(node, n, count=graph[node][n]["count"])
        return G

    def create_sub_graph_grade_class(self, graph, diagonal, grade_interaction) -> nx.Graph:
        """Creates a subgraph with the exclusion/inclusion of the off-diagonal and grade interactions

        Loops through all nodes in a graph and filters out the desired interactions to add to a new subgraph based
        on the bool variables off_diagonal and grade_interaction.

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
        diagonal : bool
            Bool that is True if interactions between Person objects on the off-diagonal is included. False otherwise
        grade_interaction : bool
            Bool that is True if interactions between Person objects of the same grade is included. False otherwise
        """

        G = nx.Graph()

        for node in graph:
            klasseAttr = node.get_class()
            for n in graph.neighbors(node):
                if diagonal:
                    if n.get_class() == klasseAttr and node.get_grade() == n.get_grade():
                        G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                        G.add_node(n)
                if grade_interaction:
                    if n.get_grade() == node.get_grade() and node.get_class() != n.get_class():
                        G.add_edge(node, n, count=graph.get_edge_data(node, n)["count"])
                        G.add_node(n)
        return G

    def pixel_dist(
        self, graph, logX, logY, axis=None, old=False, label=None, wait=False, replica=False, colour="grey"
    ) -> None:
        """Plots the distribution of all interactions that has occured between two individuals.

        Parameters
        ----------
        graph : nx.Graph or dictionary
            nx.Graph object that contains interactions at a primary school between Person objects.
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
            ## The graph is a nx.Graph object and must be made into an adjacency matrix and made cumulative
            A = self.heatmap(graph, output=True)

            length = len(graph.nodes())

            weights = A[np.triu_indices(length, k=1)].tolist()[0]

            data = sorted(weights)

            sorteddata = np.sort(data)
            d = self.to_cumulative(sorteddata)

        ## The graph is a dictionary made from to_cumulative of the experimental data and can be used as is.
        if old:
            d = graph

        ## If a subplot is being made, axis is True. Then labels can be added and the plots are plotted on the axis.
        if axis:
            if old:
                ## If old, the label is specified by the input variable label. Label can be "Day 1" or "Day 2"
                axis.plot(d.keys(), d.values(), label=label, color=colour)
            else:
                ## If not old, the label is simulated
                axis.plot(d.keys(), d.values(), label="Simulated", color=colour)

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
            if replica:
                plt.plot(d.keys(), d.values(), "--", label=label, alpha=0.4)
            else:
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

            if not wait:
                plt.show()

    def pickle_load(self, name, pixel=True) -> dict:
        file_to_read = open("./pickles" + ("/pixel/" if pixel else "/degree/") + name, "rb")
        return pickle.load(file_to_read)

    def pixel_dist_school(self, graph, old=False, both=False) -> None:
        """Plots the pixel distribution in a 2x2 plot with whole graph, off-diagonal, grade-grade and class-class interactions

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
        old : bool
            bool that determines if day 1 experimental values should be added, True if it should be added,
            False if not.
        both : bool
            bool that determines if both day 1 and day 2experimental values should be added, True if it
            should be added, False if not.
        """

        ## Generating off-diagonal, grade-grade and class-class for the simulated network
        off_diagonal = self.create_sub_graph_off_diagonal(graph, True, False)
        grade_grade = self.create_sub_graph_grade_class(graph, False, True)
        class_class = self.create_sub_graph_grade_class(graph, True, False)

        ## Generating off-diagonal, grade-grade and class-class for day 1 experimental network
        pixel = "./pickle/pixel_dist/"
        if old:
            old_graph = self.pickle_load("graph1_whole_pixel.pkl")
            old_off_diagonal = self.pickle_load("graph1_off_diag_pixel.pkl")
            old_grade = self.pickle_load("graph1_grade_pixel.pkl")
            old_class = self.pickle_load("graph1_class_pixel.pkl")

        ## Generating off-diagonal, grade-grade and class-class for day 2 experimental network
        if both:
            old_graph2 = self.pickle_load("graph2_whole_pixel.pkl")
            old_off_diagonal2 = self.pickle_load("graph2_off_diag_pixel.pkl")
            old_grade2 = self.pickle_load("graph2_grade_pixel.pkl")
            old_class2 = self.pickle_load("graph2_class_pixel.pkl")

        for i in range(0, 2):
            print(i)
            ## Generate 2x2 subplot
            figure, axis = plt.subplots(2, 2, figsize=(8, 8))
            figure.tight_layout(pad=3)

            ## Setting up wether or not the x and y axis are log10 scale
            logx = True
            logy = True

            ## Adding the whole graph pixel distribution to the [0,0] position of the graph
            self.pixel_dist(graph, logx, logy, axis[0, 0], colour="darkseagreen")
            if old:
                self.pixel_dist(old_graph, logx, logy, axis[0, 0], old=True, label="Day 1", colour="rosybrown")
            if both:
                self.pixel_dist(old_graph2, logx, logy, axis[0, 0], old=True, label="Day 2", colour="cadetblue")
            axis[0, 0].set_title("Whole network")

            ## Adding the off diagonal pixel distribution to the [1,0] position of the graph
            self.pixel_dist(off_diagonal, logx, logy, axis[1, 0], colour="darkseagreen")
            if old:
                self.pixel_dist(old_off_diagonal, logx, logy, axis[1, 0], old=True, label="Day 1", colour="rosybrown")
            if both:
                self.pixel_dist(old_off_diagonal2, logx, logy, axis[1, 0], old=True, label="Day 2", colour="cadetblue")
            axis[1, 0].set_title("Off-diagonal")

            ## Adding the grade_grade pixel distribution to the [0,1] position of the graph
            self.pixel_dist(grade_grade, logx, logy, axis[0, 1], colour="darkseagreen")
            if old:
                self.pixel_dist(old_grade, logx, logy, axis[0, 1], old=True, label="Day 1", colour="rosybrown")
            if both:
                self.pixel_dist(old_grade2, logx, logy, axis[0, 1], old=True, label="Day 2", colour="cadetblue")
            axis[0, 1].set_title("Grade")

            ## Adding the class_class pixel distribution to the [1,1] position on the graph
            self.pixel_dist(class_class, logx, logy, axis[1, 1], colour="darkseagreen")
            if old:
                self.pixel_dist(old_class, logx, logy, axis[1, 1], old=True, label="Day 1", colour="rosybrown")
            if both:
                self.pixel_dist(old_class2, logx, logy, axis[1, 1], old=True, label="Day 2", colour="cadetblue")
            axis[1, 1].set_title("Class")

            if i == 0:
                ## Adding a combined label according to the labels specified when plotting
                handles, labels = axis[1, 1].get_legend_handles_labels()
                figure.legend(handles, labels, loc="upper center")

                plt.savefig("./fig_master/pixelDist_simulated_with.png", transparent=True, dpi=500)

                plt.show()

            if i == 1:

                figure.legend([])
                plt.savefig("./fig_master/pixelDist_simulated_without.png", transparent=True, dpi=500)

                plt.show()

    def plot_correlation_between_days(self, day1, day2) -> None:
        """Plots the linear correlation between the amount of interactions a specific indicidual
        has had on the two graphs day1 and day2.

        Parameters
        ----------
        day1 : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects on day 1
        day2 : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects on day 2
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

    def display_network(self, graph, axis=None) -> None:
        """Plots a Network graph with nodes and edges.

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
        axis : axis object (Matplotlib)
            axis object that determines the position of the plot in a subgraph. Default is None
        """

        if axis:
            nx.draw(graph, ax=axis)
        else:
            nx.draw(graph)
            plt.show()

    def heatmap(self, day, output=False, axis=None):
        """Plots the interaction heatmap of the adjacency matrix of a nx.Graph

        Parameters
        ----------
        day : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
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
            for i in range(0, 2):
                if i == 0:
                    sns.heatmap(A_M, robust=True)
                    plt.savefig("./fig_master/heatmap_sim.png", transparent=True, dpi=500)
                    plt.show()
                else:
                    sns.heatmap(A_M, robust=False)
                    plt.savefig("./fig_master/heatmap_sim_not_robust.png", transparent=True, dpi=500)
                    plt.show()

    # Generates histogram of degree distribution
    def hist_distribution(self, graph) -> None:
        """Plots a histogram of the degree distribution of a graph

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
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

    def to_cumulative(self, l) -> dict:
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

    def cumulative_distribution_log(
        self,
        graph,
        logX=False,
        logY=True,
        axis=None,
        old=False,
        label=None,
        cap=0,
        wait=False,
        replica=False,
        colour="grey",
    ) -> None:
        """Plots the cumulative degree P(X>=x), where x is the frequency distribution of a given graph.

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects
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
        if cap > 0:
            for n in graph.nodes():
                if graph.degree(n, weight="count") > cap:
                    deg = graph.degree(n, weight="count")
                    degs[n] = deg
        else:
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

        d = self.to_cumulative(sorteddata)

        ##Load in degreedistributions from experimental data
        if old:
            old1 = self.pickle_load("Degreedistribution_Day1.pkl", pixel=False)
            old2 = self.pickle_load("Degreedistribution_Day2.pkl", pixel=False)

        ##If subplot is created label can be added
        if axis:
            if label:
                axis.plot(d.keys(), d.values(), label=label, color=colour)
            else:
                axis.plot(d.keys(), d.values(), color=colour)

        ## Else creates normal matplotlib object for a single plot
        else:
            if replica:
                plt.plot(d.keys(), d.values(), "--", label=label, alpha=0.4)
            else:
                plt.plot(d.keys(), d.values(), label="Simulated")
            if old:
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
            if not wait:
                plt.show()

    def hist_plot(self, d, label, logX=False, logY=True, axis=None, colour="grey") -> None:
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
            axis.plot(d.keys(), d.values(), label=label, color=colour)
            if logX:
                axis.set_xscale("log")
                axis.set_xlabel("log Degree")
            else:
                axis.set_xlabel("Degree")
            if logY:
                axis.set_yscale("log")
                axis.set_ylabel("log Frequency")
            else:
                axis.set_ylabel("Frequency")
        else:
            plt.plot(d.keys(), d.values(), label=label, color=colour)
            if logX:
                plt.xscale("log")
                plt.xlabel("log Degree")
            else:
                plt.xlabel("Degree")
            if logY:
                plt.yscale("log")
                plt.ylabel("log Frequency")
            else:
                plt.ylabel("Frequency")

    def degree_distribution_layers(self, both=False, experimental=False, sim=None) -> None:
        """Plots the degree distribution of both experimental and simulated graphs for the whole graph
         and the subgraphs off-diagonal, grade-grade and class-class.


        Parameters
        ----------
         both : bool
             bool that determines if both day 1 and day 2 experimental values should be added, True if it
             should be added, False if not. Default is False
         experimental : bool
             True if simulated data should be added to the plot. False otherwise. Default is False.
         sim : nx.Graph
             nx.Graph object that contains interactions at a primary school between Person objects. Default is None
        """

        ## Load day 1 dictionaries
        graph1 = self.pickle_load("DegreeDictwhole1.pkl", pixel=False)
        off_diag1 = self.pickle_load("DegreeDictOffDiag1.pkl", pixel=False)
        grade1 = self.pickle_load("DegreeDictgrade1.pkl", pixel=False)
        class1 = self.pickle_load("DegreeDictclass1.pkl", pixel=False)

        ## Load day 2 dictionaries
        if both:
            graph2 = self.pickle_load("DegreeDictwhole2.pkl", pixel=False)
            off_diag2 = self.pickle_load("DegreeDictOffDiag2.pkl", pixel=False)
            grade2 = self.pickle_load("DegreeDictgrade2.pkl", pixel=False)
            class2 = self.pickle_load("DegreeDictclass2.pkl", pixel=False)

        ## load experimental graphs
        if experimental:
            graph = sim
            off_diagE = self.create_sub_graph_off_diagonal(self.network.get_graph(), True, False)
            gradeE = self.create_sub_graph_grade_class(self.network.get_graph(), False, True)
            classE = self.create_sub_graph_grade_class(self.network.get_graph(), True, False)
        for i in range(0, 2):
            ## Create figure and axis objects for subplot
            figure, axis = plt.subplots(2, 2, figsize=(10, 8))
            figure.tight_layout(pad=4)

            xlog = False
            ylog = True

            #  self.iterator_colours = iter(["rosybrown", "sienna", "tan", "darkgoldenrod", "olivedrab"])

            ## Adding the whole graph degree distribution to the [0,0] position of the graph
            self.hist_plot(graph1, logX=xlog, logY=ylog, axis=axis[0, 0], label="Day 1", colour="rosybrown")
            if both:
                self.hist_plot(graph2, logX=xlog, logY=ylog, axis=axis[0, 0], label="Day 2", colour="cadetblue")
            if experimental:
                self.cumulative_distribution_log(
                    graph, xlog, ylog, axis=axis[0, 0], label="Simulated", colour="darkseagreen"
                )
            axis[0, 0].set_title("Whole graph")

            ## Adding the off diagonal degree distribution to the [1,0] position of the graph
            self.hist_plot(off_diag1, logX=xlog, logY=ylog, axis=axis[1, 0], label="Day 1", colour="rosybrown")
            if both:
                self.hist_plot(off_diag2, logX=xlog, logY=ylog, axis=axis[1, 0], label="Day 2", colour="cadetblue")
            if experimental:
                self.cumulative_distribution_log(
                    off_diagE, logX=xlog, logY=ylog, axis=axis[1, 0], label="Simulated", colour="darkseagreen"
                )
            axis[1, 0].set_title("Off diagonal")

            ## Adding the grade-grade degree distribution to the [0,1] position of the graph
            self.hist_plot(grade1, logX=xlog, logY=ylog, axis=axis[0, 1], label="Day 1", colour="rosybrown")
            if both:
                self.hist_plot(grade2, logX=xlog, logY=ylog, axis=axis[0, 1], label="Day 2", colour="cadetblue")
            if experimental:
                self.cumulative_distribution_log(
                    gradeE, logX=xlog, logY=ylog, axis=axis[0, 1], label="Simulated", colour="darkseagreen"
                )
            axis[0, 1].set_title("Grade")

            ## Adding the class-class degree distribution to the [1,1] position of the graph
            self.hist_plot(class1, logX=xlog, logY=ylog, axis=axis[1, 1], label="Day 1", colour="rosybrown")
            if both:
                self.hist_plot(class2, logX=xlog, logY=ylog, axis=axis[1, 1], label="Day 2", colour="cadetblue")
            if experimental:
                self.cumulative_distribution_log(
                    classE, logX=xlog, logY=ylog, axis=axis[1, 1], label="Simulated", colour="darkseagreen"
                )
            axis[1, 1].set_title("Class")

            if i == 0:
                ## Adding a combined label according to the labels specified when plotting
                handles, labels = axis[1, 1].get_legend_handles_labels()
                figure.legend(handles, labels, loc="center")  # loc="lower center"
                plt.savefig("./fig_master/degree_dist_sim_with.png", transparent=True, dpi=500)
                plt.show()
            if i == 1:
                figure.legend([])
                plt.savefig("./fig_master/degree_dist_sim_no_lable.png", transparent=True, dpi=500)
                plt.show()

    def outlierDist(self, graph) -> None:
        """Get the distribution of all interactions a max node has with others

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects.
        """

        dictionary = {}

        for (node, val) in graph.degree(weight="count"):
            dictionary[node] = val

        sortedDegrees = dict(sorted(dictionary.items(), key=lambda item: item[1]))

        highest_degree_node = max(dictionary, key=dictionary.get)
        highest_degree = sortedDegrees[highest_degree_node]
        print(f"Highest degree node is {highest_degree_node} and its degree is {highest_degree}")

        list_of_interactions = []
        list_of_same_class = []
        list_of_same_grade = []

        for edge in graph.edges(highest_degree_node, data="count"):
            list_of_interactions.append(edge[2])

            i = edge[0]
            j = edge[1]

            if i.get_class_and_grade() == j.get_class_and_grade():
                list_of_same_class.append(edge[2])
            if i.get_grade() == j.get_grade():
                list_of_same_grade.append(edge[2])

        def Average(lst):
            return sum(lst) / len(lst)

        list_of_interactions.sort()
        print(list_of_interactions)
        print(f"length: {len(list_of_interactions)}")
        print(f"Average interaction: {Average(list_of_interactions)}")

        print(f"Interactions within same grade: {len(list_of_same_grade)}")
        print(f"Same grade average interaction: {Average(list_of_same_grade)}")

        print(f"Interactions within same class: {len(list_of_same_class)}")
        print(f"Same Class average interaction: {Average(list_of_same_class)}")

    def run_analysis(self, graph) -> None:
        """Plots a subplot with degree distribution, heatmap and drawn network of a graph

        Parameters
        ----------
        graph : nx.Graph
            nx.Graph object that contains interactions at a primary school between Person objects.
        """
        figure, axis = plt.subplots(2, 2)

        ## Degree distribution
        self.cumulative_distribution_log(graph, False, True, axis[0, 0])
        axis[0, 0].set_title("Degree dist")

        ## Heatmap
        self.heatmap(graph, axis[1, 0])
        axis[1, 0].set_title("Heatmap")

        ## Display of the network
        self.display_network(graph, axis[0, 1])
        axis[0, 1].set_title("network")

        for ax in axis.flat:
            ## check if something was plotted
            if not bool(ax.has_data()):
                figure.delaxes(ax)  ## delete if nothing is plotted in the axes obj

        plt.show()

    def runAnalysis2(self, G):
        color_map = []

        for node in G.nodes():
            print(node)
            if node.get_grade() == 1:
                color_map.append("rosybrown")
            elif node.get_grade() == 2:
                color_map.append("sienna")
            elif node.get_grade() == 3:
                color_map.append("tan")
            elif node.get_grade() == 4:
                color_map.append("darkgoldenrod")
            elif node.get_grade() == 5:
                color_map.append("olivedrab")
            else:
                color_map.append("slategrey")

        degree_sequence = sorted([d for n, d in G.degree(weight="weight")], reverse=False)

        fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])

        Gcc = G
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, node_color=color_map)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Day 1 network")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, 2:])

        degs = {}
        for n in G.nodes():
            deg = G.degree(n, weight="count")
            degs[n] = deg

        items = sorted(degs.items())

        data = []
        for line in items:
            data.append(line[1])

        sorteddata = np.sort(data)

        d = self.to_cumulative(sorteddata)

        ax1.plot(d.keys(), d.values(), color="seagreen")

        ax1.set_title("Cumulative degree distribution P(X > x)")
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel("Degree")

        ax2 = fig.add_subplot(axgrid[3:, :2])

        ax2.set_title("Histogram of degree distribution")
        ax2.hist(data, bins=20, color="seagreen", ec="black")  # col = 'skyblue for day2, mediumseagreen for day1
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Frequency")

        fig.tight_layout()
        plt.savefig("./fig_master/full_analysis_simulated.png", transparent=True, dpi=500)
        plt.show()

    def getClasses(self, G):  # -> [{1a}, {1b}, {2a}, {2b}, {3a}, {3b}, {4a}, {4b}, {5a}, {5b}]
        d = {}
        for node in G.nodes():
            if node.get_class_and_grade() not in d:
                d[node.get_class_and_grade()] = {node}
            else:
                d[node.get_class_and_grade()].add(node)

        return [classSet[1] for classSet in sorted(list(d.items()), key=lambda x: x[0])]

    def modularity(self, G):
        communities = self.getClasses(G)
        M = community.modularity(G, communities, "weight")
        print(M)

    def replica_degree(self, G, network):
        graph1 = self.pickle_load("DegreeDictwhole1.pkl", pixel=False)
        graph2 = self.pickle_load("DegreeDictwhole2.pkl", pixel=False)
        self.hist_plot(graph1, label="Empiric day 1")
        self.hist_plot(graph2, label="Empiric day 2")
        self.cumulative_distribution_log(G, wait=True, label="0", replica=True)
        for i in range(0, 10):
            graph = network.generate_a_day()
            self.cumulative_distribution_log(graph, wait=True, label=str(i + 1), replica=True)
            print(graph)
        plt.legend()
        plt.show()

    def replica_pixel(self, G, network):
        old_graph = self.pickle_load("graph1_whole_pixel.pkl")
        self.pixel_dist(old_graph, logX=True, logY=True, label="Day 1", wait=True, old=True)
        old_graph = self.pickle_load("graph2_whole_pixel.pkl")
        self.pixel_dist(old_graph, logX=True, logY=True, label="Day 2", wait=True, old=True)

        for i in range(0, 10):
            self.pixel_dist(network.generate_a_day(), wait=True, logX=True, logY=True, replica=True)
        plt.legend()
        plt.show()

    def heatmap_asymptomatic_calibration(self):
        pkl = open("./pickles/asymptomatic_calibration35.pickle", "rb")
        asymptotic_dict = pickle.load(pkl)
        print(asymptotic_dict)
        df = pd.DataFrame.from_dict(asymptotic_dict, orient="index")
        # df = df.pivot("%Asymptomatic", "Day")
        plt.figure(figsize=(10, 7))
        sns.heatmap(df, cmap=sns.cm.rocket_r, linewidths=0.5, annot=False)
        plt.xlabel("Day")
        plt.ylabel("%Asymptomatic")
        plt.show()
        return df

    def pie_chart(self, filename, save_as):
        df = pd.read_csv(filename)
        list_of_R_null = df.values.tolist()

        list_no_1 = []
        for item in list_of_R_null:
            list_no_1.append(item[1])
        list_no_1.sort()
        print(len(list_no_1))

        new_list = []
        for item in list_no_1:
            if item > 6:
                new_list.append("$\geq 7$")
            else:
                new_list.append("=" + str(item))

        new_dict = {}
        R_0 = "$R_{0}$"
        for item in new_list:
            if R_0 + str(item) not in new_dict:
                new_dict[R_0 + str(item)] = 1
            else:
                new_dict[R_0 + str(item)] += 1
        labels = []
        sizes = []
        print(new_dict)
        for x, y in new_dict.items():
            labels.append(x)

            sizes.append(y)

        # Plot
        plt.pie(sizes, labels=labels)
        colors = ["#7F4F24", "#936639", "#A68A64", "#B6AD90", "#C2C5AA", "#A4AC86", "#656D4A", "#414833"][::-1]
        # colors = ["#CB997E", "#EDDCD2", "#FFF1E6", "#F0EFEB", "#DDBEA9", "#A5A58D", "#B7B7A4", "#414833"]
        # Set your custom color palette
        sns.set_palette(sns.color_palette(colors))

        # colors = sns.color_palette("dark")
        plt.pie(sizes, labels=labels, colors=colors)  # autopct="%0.0f%%"
        plt.tight_layout()
        plt.savefig(f"./fig_master/R0_{save_as}.png", transparent=True, dpi=500)

        plt.axis("equal")
        plt.show()

    def average_of_simulations(self, networkType: str) -> pd.DataFrame:
        dfs = []
        for filename in filter(
            lambda x: networkType in x and "average" not in x and "p0" not in x, os.listdir("./data/empiric_vs_model2/")
        ):
            dfs.append(pd.read_csv(f"./data/empiric_vs_model2/{filename}", header=0))

        new_df = functools.reduce(lambda a, b: a.add(b, fill_value=0), dfs)

        new_df = new_df / len(dfs)
        new_df.to_csv(f"./data/empiric_vs_model2/{networkType}_average.csv", index=False)
        return new_df

    def accumulate_R0(self, networkType: str) -> pd.DataFrame:
        l = []
        for filename in sorted(
            filter(
                lambda x: networkType in x and "average" not in x and "transmission" not in x,
                os.listdir("./data/weekly_testing"),
            )
        ):
            with open(f"./data//weekly_testing/{filename}") as f:
                l.extend([int(x.split(",")[-1].strip("\n")) for x in f.readlines()])
                print(filename)

        with open(f"{networkType}_average_by_p0.csv", "w") as f:
            f.writelines(map(lambda x: str(x) + "\n", l))
