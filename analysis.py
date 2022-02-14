from cProfile import label

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

    def createSubGraphWithoutGraph(self, graph, diagonal, gradeInteraction):

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

    def pixelDist(self, graph, logX, logY, axis=None, old=False, label=None):
        if not old:
            A = self.heatmap(graph, output=True)

            length = len(graph.nodes())

            weights = A[np.triu_indices(length, k=1)].tolist()[0]

            data = sorted(weights)

            sorteddata = np.sort(data)
            d = self.toCumulative(sorteddata)

        if old:
            d = graph

        if axis:
            if old:
                axis.plot(d.keys(), d.values(), label=label)
            else:
                axis.plot(d.keys(), d.values(), label="Simulated")

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
            plt.plot(d.keys(), d.values())
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
        off_diagonal = self.createSubGraphWithout(graph, True, False)
        grade_grade = self.createSubGraphWithoutGraph(graph, False, True)
        class_class = self.createSubGraphWithoutGraph(graph, True, False)

        if old:
            old_graph = self.pickleLoad("graph1_whole_pixel.pkl")
            old_off_diagonal = self.pickleLoad("graph1_off_diag_pixel.pkl")
            old_grade = self.pickleLoad("graph1_grade_pixel.pkl")
            old_class = self.pickleLoad("graph1_class_pixel.pkl")

        if both:
            old_graph2 = self.pickleLoad("graph2_whole_pixel.pkl")
            old_off_diagonal2 = self.pickleLoad("graph2_off_diag_pixel.pkl")
            old_grade2 = self.pickleLoad("graph2_grade_pixel.pkl")
            old_class2 = self.pickleLoad("graph2_class_pixel.pkl")

        figure, axis = plt.subplots(2, 2, figsize=(8, 8))
        figure.tight_layout(pad=3)

        logx = True
        logy = True

        self.pixelDist(graph, logx, logy, axis[0, 0])
        if old:
            self.pixelDist(old_graph, logx, logy, axis[0, 0], old=True, label="Day 1")
        if both:
            self.pixelDist(old_graph2, logx, logy, axis[0, 0], old=True, label="Day 2")
        axis[0, 0].set_title("Whole network")

        self.pixelDist(off_diagonal, logx, logy, axis[1, 0])
        if old:
            self.pixelDist(old_off_diagonal, logx, logy, axis[1, 0], old=True, label="Day 1")
        if both:
            self.pixelDist(old_off_diagonal2, logx, logy, axis[1, 0], old=True, label="Day 2")
        axis[1, 0].set_title("Off-diagonal")

        self.pixelDist(grade_grade, logx, logy, axis[0, 1])
        if old:
            self.pixelDist(old_grade, logx, logy, axis[0, 1], old=True, label="Day 1")
        if both:
            self.pixelDist(old_grade2, logx, logy, axis[0, 1], old=True, label="Day 2")
        axis[0, 1].set_title("grade-grade")

        self.pixelDist(class_class, logx, logy, axis[1, 1])
        if old:
            self.pixelDist(old_class, logx, logy, axis[1, 1], old=True, label="Day 1")
        if both:
            self.pixelDist(old_class2, logx, logy, axis[1, 1], old=True, label="Day 2")
        axis[1, 1].set_title("class-class")

        handles, labels = axis[1, 1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center")

        plt.savefig("pixelDistSimulated.png", bbox_inches="tight", dpi=150)

        plt.show()

    # Investigate pearson correlation from day 1 to day 2 and plot the degree for each nodes on day 1 versus day 2
    def plot_Correlation_between_Days(self, day1, day2):

        degday1 = [val for (node, val) in self.network.daily_list[day1].degree(weight="count")]
        degday2 = [val for (node, val) in self.network.daily_list[day2].degree(weight="count")]

        plt.scatter(degday1, degday2)
        print("Pearson correlation:")
        print(np.corrcoef(degday1, degday2))
        print(stats.pearsonr(degday1, degday2))
        plt.show()

    # Draw the node + edge representation of the network. Axis can be used when plotting many graphs in the same plot
    def displayNetwork(self, graph, axis=None):
        if axis:
            nx.draw(graph, ax=axis)
        else:
            nx.draw(graph)
            plt.show()

    # Generate a heatmap of the adjacency matrix of a graph. Axis can be used when plotting many graphs in the same plot
    def heatmap(self, day, output=False, axis=None):
        A = nx.adjacency_matrix(day, nodelist=sorted(day.nodes), weight="count")

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
        for n in graph.nodes():
            deg = graph.degree(n, weight="count")
            degs[n] = deg

        items = sorted(degs.items())

        data = []
        for line in items:
            data.append(line[1])

        plt.hist(data, bins=10, color="skyblue", ec="black")  # col = 'skyblue for day2, mediumseagreen for day1
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

    def toCumulative(self, l):
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

    # Generates comulative distribution. For the project logX=False, logY=True for semilog
    def histDistributionLog(self, graph, logX=False, logY=True, axis=None, old=False, label=None):

        degs = {}
        for n in graph.nodes():
            deg = graph.degree(n, weight="count")

            degs[n] = deg

        items = sorted(degs.items())

        data = []
        for line in items:
            data.append(line[1])
        N = len(data)
        sorteddata = np.sort(data)

        d = self.toCumulative(sorteddata)

        old1 = self.pickleLoad("Degreedistribution_Day1.pkl")
        old2 = self.pickleLoad("Degreedistribution_Day2.pkl")

        if axis:
            if label:
                axis.plot(d.keys(), d.values(), label=label)
            else:
                axis.plot(d.keys(), d.values())
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
        if axis:
            axis.plot(d.keys(), d.values(), label=label)

    def plotDegreeDistSubGraphs(self, both=False, experimental=False, exp=None):
        graph1 = self.pickleLoad("DegreeDictwhole1.pkl")
        off_diag1 = self.pickleLoad("DegreeDictOffDiag1.pkl")
        grade1 = self.pickleLoad("DegreeDictgrade1.pkl")
        class1 = self.pickleLoad("DegreeDictclass1.pkl")

        if both:
            graph2 = self.pickleLoad("DegreeDictwhole2.pkl")
            off_diag2 = self.pickleLoad("DegreeDictOffDiag2.pkl")
            grade2 = self.pickleLoad("DegreeDictgrade2.pkl")
            class2 = self.pickleLoad("DegreeDictclass2.pkl")

        if experimental:
            graph = exp
            off_diagE = self.createSubGraphWithout(graph, True, False)
            gradeE = self.createSubGraphWithoutGraph(graph, False, True)
            classE = self.createSubGraphWithoutGraph(graph, True, False)

        figure, axis = plt.subplots(2, 2, figsize=(10, 8))
        figure.tight_layout(pad=4)
        self.histPlot(graph1, logX=False, logY=True, axis=axis[0, 0], label="Day 1")
        if both:
            self.histPlot(graph2, logX=False, logY=True, axis=axis[0, 0], label="Day 2")
        if experimental:
            self.histDistributionLog(graph, False, True, axis=axis[0, 0], label="Simulated")
        axis[0, 0].set_title("Whole graph")

        self.histPlot(off_diag1, logX=False, logY=True, axis=axis[1, 0], label="Day 1")
        if both:
            self.histPlot(off_diag2, logX=False, logY=True, axis=axis[1, 0], label="Day 2")
        if experimental:
            self.histDistributionLog(off_diagE, logX=False, logY=True, axis=axis[1, 0], label="Simulated")
        axis[1, 0].set_title("Off diagonal")

        self.histPlot(grade1, logX=False, logY=True, axis=axis[0, 1], label="Day 1")
        if both:
            self.histPlot(grade2, logX=False, logY=True, axis=axis[0, 1], label="Day 2")
        if experimental:
            self.histDistributionLog(gradeE, logX=False, logY=True, axis=axis[0, 1], label="Simulated")
        axis[0, 1].set_title("Grade")

        self.histPlot(class1, logX=False, logY=True, axis=axis[1, 1], label="Day 1")
        if both:
            self.histPlot(class2, logX=False, logY=True, axis=axis[1, 1], label="Day 2")
        if experimental:
            self.histDistributionLog(classE, logX=False, logY=True, axis=axis[1, 1], label="Simulated")
        axis[1, 1].set_title("Class")

        handles, labels = axis[1, 1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="center")  # loc="lower center"

        plt.show()

    def runAnalysis(self, graph):
        figure, axis = plt.subplots(2, 2)

        ######## Degree distribution ########
        self.histDistributionLog(graph, False, True, axis[0, 0])
        axis[0, 0].set_title("Degree dist")

        ######## Heatmap ########
        self.heatmap(graph, axis[1, 0])
        axis[1, 0].set_title("Heatmap")

        ######## Weight Frequency of node 1 ########
        self.displayNetwork(graph, axis[0, 1])
        axis[0, 1].set_title("network")

        for ax in axis.flat:
            ## check if something was plotted
            if not bool(ax.has_data()):
                figure.delaxes(ax)  ## delete if nothing is plotted in the axes obj

        plt.show()

        return None
