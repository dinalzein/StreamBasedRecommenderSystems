import json
from pathlib import Path
import sys
from Evaluation import Evaluation

'''
runs a certain model using the Evaluation.py class
'''

class RunModel(object):

    def __init__(self, model_name, nb_neighbors, nb_features, epoch, batch, regularization, learning_rate, metrics, nb_items, nb_users, train, test):
        self.model_name = model_name
        self.epoch = epoch[0]
        self.batch = batch[0]
        self.regularization = regularization[0]
        self.learning_rate = learning_rate[0]
        self.nb_features = nb_features[0]
        self.nb_neighbors = nb_neighbors[0]
        self.metrics = metrics
        self.nb_items = nb_items
        self.nb_users = nb_users
        self.train = train
        self.test = test

    def run_model(self):
        eval = Evaluation()
        metrics_statistics = eval.evaluation(self.model_name, self.nb_neighbors, self.nb_features, self.epoch, self.batch, self.regularization, self.learning_rate, self.metrics, self.nb_items, self.nb_users, self.train, self.test)

        #print results:
        for key in metrics_statistics:
            print(key + ":" + str(metrics_statistics[key])+"\n")
