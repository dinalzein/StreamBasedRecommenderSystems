from collections import  defaultdict
from collections import OrderedDict
from Model import Model
import sys
import random

# this class implements the Random model: recommending random items for each user

class Random(Model):
    def __init__(self, nb_items):

        self.nb_items = nb_items


    # give each item a random score
    def get_recommendation_rand(self,user):
        score = defaultdict(list)
        for i in range(0,self.nb_items):
            score[i] = random.random()

        sorted_score = OrderedDict(sorted(score.items(), key=lambda x: x[1], reverse=True))
        return sorted_score

    def batch_training(self,training_data):
        pass


    def get_recommendations(self,user):
        sorted_score = self.get_recommendation_rand(user)
        return sorted_score

    def incremental_training(self,user,item):
        pass
