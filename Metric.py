import statistics
import itertools
from statistics import StatisticsError
import math

'''
this class implements the following metrics: Recall, DCG, and MRR
'''

class Metric(object):
    def __init__(self):

        self.metric_values = []
        # available metrics
        self.metrics = {"mrr": self.compute_mean_reciprocal_rank,
                        "recall":self.compute_recall_N,
                        "dcg":self.compute_dcg_N,
                        }

    #check the name of the metric in order to call the corresponding method
    def compute_metric(self, metric_name, item, sorted_score, N):
        value = 0
        if "recall@" in metric_name:
            value = self.metrics["recall"](sorted_score,N,item)
        elif "dcg@" in metric_name:
            value = self.metrics["dcg"](sorted_score,N,item)
        else:
            value = self.metrics[metric_name](sorted_score,N,item)
        return value

    def compute_mean_reciprocal_rank(self,sorted_score,N,item):
        #return 1/rank(i)
        rank = list(sorted_score.keys()).index(item)
        return 1/(rank+1)

    def compute_recall_N(self,sorted_score,N,item):
        #Recall@N = 0 or 1 (1 if the item “i” is included in the recommendation list)
        top_N = dict(itertools.islice(sorted_score.items(), int(N)))
        if item in top_N.keys():
            return 1
        return  0

    def compute_dcg_N(self,sorted_score,N,item):
        top_N = dict(itertools.islice(sorted_score.items(), int(N)))
        if item in top_N.keys():
            return 1/math.log((top_N[item]+1),2)
        return  0

    def store_metric_value(self,value):
        self.metric_values.append(value)

    def statistical_results(self):
        metric_statistics = {}
        try:
            mean_value = statistics.mean(self.metric_values)
            metric_statistics["mean"] = mean_value

        except StatisticsError:
            metric_statistics["mean"] = ""

        try:

            median_value = statistics.median(self.metric_values)
            metric_statistics["median"] = median_value

        except StatisticsError:
            metric_statistics["median"] = ""


        try:
            standered_deviation_value = statistics.pstdev(self.metric_values)
            metric_statistics["standered_deviation"] = standered_deviation_value

        except StatisticsError:
            metric_statistics["standered_deviation"] = ""

        try:
            min_value = min(self.metric_values)
            metric_statistics["minimum"] = min_value

        except ValueError:
            metric_statistics["minimum"] = ""

        try:
            max_value=max(self.metric_values)
            metric_statistics["maximum"] = max_value

        except ValueError:
            metric_statistics["maximum"] = ""

        try:
            mode_value = statistics.mode(self.metric_values)
            metric_statistics["mode"] = mode_value

        except StatisticsError:

            try:
                metric_statistics["mode"] = max(set(self.metric_values), key = self.metric_values.count)

            except ValueError:
                metric_statistics["maximum"] = ""


        return metric_statistics
