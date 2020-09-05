from Model import Model
from KnnI import KnnI
from KnnU import KnnU
from MostPopular import MostPopular
from MF import MF
from MFNF import MFNF
from MFBPR import MFBPR
from Random import Random
from Metric import Metric

'''
the main function of the class can be summarized in the following:
For every item in the catalog:
   Compute f(u,i) and store it in list “Score” (this method is implemented in each model class)
   Sort “Score” in inverse order (done in the corresponding model class)
   Evaluate the recommendations (Evaluate):
    evaluate_metric(Recommendations, i) = compute meric(u, i, Score), Store value of metric = append value to MRRs[]
   Update the model based on the observation (Model):
    Update the model given the current model and the observation (u,i)
'''

class Evaluation(object):


    def __init__(self):
        pass


    def evaluation(self, model_name, nb_neighbors, nb_features, epoch, batch, regularization, learning_rate, metrics, nb_items, nb_users, train, test):
        # create an instance of the model
        self.create_instance(model_name, nb_neighbors, nb_features, epoch, batch, regularization, learning_rate, nb_items, nb_users)
        self.model_instance.batch_training(train)

        metrics_statistics={}
        metrics_instances={}

        for met in metrics:
            metrics_instances[met.lower()]=Metric()

        count=0
        for index, row in test.iterrows():
            sorted_score = self.model_instance.get_recommendations(row[0])
            count = count + 1
            self.model_instance.incremental_training(row[0],row[1])
            for met in metrics:
                N = 0
                if "@" in met:
                    N = met.split("@")[1]
                met_val = metrics_instances[met.lower()].compute_metric(met.lower(), row[1], sorted_score, N)
                metrics_instances[met.lower()].store_metric_value(round(met_val, 2))

        for met in metrics:
            metric_statistics = metrics_instances[met.lower()].statistical_results()
            metrics_statistics[met] = metric_statistics

        return metrics_statistics

    #create an instance of the chosen model
    def create_instance(self, model_name, nb_neighbors, nb_features, epoch, batch, regularization, learning_rate, nb_items, nb_users):
        if model_name == 'Rand':
            self.model_instance = Random(nb_items)

        elif model_name == 'MP':
            self.model_instance = MostPopular()

        elif model_name == 'KnnI':
            self.model_instance = KnnI(nb_neighbors, nb_items)

        elif model_name == 'KnnU':
            self.model_instance = KnnU(nb_neighbors, nb_items, nb_users)

        elif model_name == 'MF':
            self.model_instance = MF(nb_features, epoch, regularization, learning_rate, nb_items, nb_users)

        elif model_name == 'MFNF':
            self.model_instance = MFNF(nb_features, epoch, regularization, learning_rate, nb_items, nb_users)

        elif model_name == 'MFBPR':
            self.model_instance = MFBPR(nb_features, epoch, batch, regularization, learning_rate, nb_items, nb_users)
