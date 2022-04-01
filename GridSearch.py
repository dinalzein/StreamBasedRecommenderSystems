import json
from Evaluation import Evaluation
import itertools
from collections import OrderedDict
from collections import  defaultdict

'''
this class applies gridsearch on the imput parameters
'''

class GridSearch(object):
    #initilize all the required parameters to test the evaluation class
    def __init__(self, model_name, nb_neighbors, nb_features, epoch, batch, regularization, learning_rate, metrics, nb_items, nb_users, train, test):

        self.model_name = model_name
        self.epoch = epoch
        self.batch = batch
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.nb_features = nb_features
        self.nb_neighbors = nb_neighbors
        self.metrics = metrics
        self.nb_items = nb_items
        self.nb_users = nb_users
        self.train = train
        self.test = test

        self.parameters = {
        'k': self.nb_neighbors,
        'n': self.nb_features,
        'l': self.learning_rate,
        'r': self.regularization,
        'epoch': self.epoch,
        'batch': self.batch,
        }

    def run_grid(self):
        best_performance=defaultdict(list) # best parameters for each metric
        for metric in self.metrics:
            best_performance[metric]=['best_comb:',-1]


        # compute all possible combinations of the input parameters
        parameters_permutations = self.combinations(self.dict_to_list(OrderedDict(sorted(self.parameters.items()))))
        parameters_keys = sorted(OrderedDict(self.parameters))

        # for each combination of the input parameters: call the Evaluation.py class
        for comb in parameters_permutations:
            parameters_comb={}
            for i in range(len(comb)):
                parameters_comb[parameters_keys[i]] = comb[i]

            #print("The metric results for parameters:")
            #for key, value in parameters_comb.items():
            #    print(f"{key}: {value}")

            res = ", ".join(f"{key}: {value}" for key, value in parameters_comb.items())
            print(res)


            #running the model
            eval = Evaluation()
            metrics_statistics = eval.evaluation(self.model_name, parameters_comb['k'], parameters_comb['n'], parameters_comb['epoch'], parameters_comb['batch'], parameters_comb['r'], parameters_comb['l'], self.metrics, self.nb_items, self.nb_users, self.train, self.test)

            for metric in self.metrics:
                try:
                    if best_performance[metric][1] < metrics_statistics[metric]['mean']:
                        best_performance[metric] = [parameters_comb, metrics_statistics[metric]['mean']]
                except TypeError:
                    pass


            for key in metrics_statistics:
                print("%s : %s \n" % (key, str(metrics_statistics[key])))


        # replace -1 by empty string incase of empty test set
        for metric in self.metrics:
            if best_performance[metric][1] == -1:
                    best_performance[metric][1] = ""


        print("Best combination of parameters per metric")
        for key in best_performance:
            print(key + ": " + str(best_performance[key][0]) + "\n")


    def dict_to_list(self, dict):
        lst=[]
        for key in dict.keys():
            lst.append(dict[key])
        return lst

    def combinations(self, lst):
        return list(itertools.product(*lst))
