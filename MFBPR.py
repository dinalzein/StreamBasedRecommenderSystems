import numpy as np
from collections import OrderedDict
from Model import Model
import copy
import random
from math import copysign
from numpy.random import choice
from scipy.spatial import distance

'''
this class implements the Matrix Factorization algorithm using negtaive feedback with BPR approach
referenced paper: "BPR: Bayesian personalized ranking from implicit feedback"
'''



class MFBPR(Model):
    def __init__(self, nb_features, epoch, batch, regularization, learning_rate, nb_items, nb_users):

        self.epoch = int(epoch) #number of iterations for the batch training
        self.nb_features = int(nb_features)

        self.reg_user = regularization #regularization parameter for the user
        self.reg_pos_item = regularization#regularization parameter for the positive items
        self.reg_neg_item = regularization #regularization parameter for the negative items

        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate

        self.init_mean = 1
        self.init_stdv = 1

        self.batch = int(batch) # batch update
        self.reservoir_length = 1000000
        self.epoch_inc = int(epoch) # number of iterations for the incremental update (till convergence)


        self.nb_items = nb_items
        self.nb_users = nb_users

        self.users_latent_model = np.random.normal(self.init_mean, self.init_stdv, (self.nb_users, self.nb_features)) #n*k
        self.items_latent_model = np.random.normal(self.init_mean, self.init_stdv,(self.nb_items,self.nb_features)) #m*k   R is n*m

        self.reservoir = list()
        self.negative_items_num = 59 #number of negaive items we are considering for each positive item
        self.counter = 0 #this is a temp variable used to check when should we update the model(in the incremental version)


    def batch_matrixFact(self, train):
        for count in range(self.epoch):
            for index, row in train.iterrows():
                self.update_user_item_models_batch(row[0], row[1])


    def update_user_item_models_batch(self, user, item):
        self.update_reservoir(user, item)
        self.update_model()


    def update_user_item_models_incremental(self, user, item):
        self.update_reservoir(user, item)
        self.counter = self.counter+1
        if self.counter == self.batch:
            self.update_model()
            self.counter = 0

    def update_reservoir(self, user, item):
        if len(self.reservoir) >= self.reservoir_length:
            self.reservoir.pop(random.randrange(len(self.reservoir)))
        self.reservoir.append((user,item))

    def update_model(self):

        for count in range(self.epoch_inc):
            user,pos_item = random.choice(self.reservoir)
            negative_items_lst = self.sample_candidate_negative_items(user,pos_item)
            neg_item = self.sample_negative_item(pos_item, negative_items_lst)

            user_vector = copy.copy(self.users_latent_model[user])
            item_vector_pos = copy.copy(self.items_latent_model[pos_item])

            item_vector_neg = copy.copy(self.items_latent_model[neg_item]) if neg_item is not None else np.zeros(self.nb_features)

            xui = np.dot(user_vector, item_vector_pos)
            xuj = np.dot(user_vector, item_vector_neg)

            yuij = self.sign_func((xui - xuj))

            #wu ←wu +ηyuij (hi −hj)−ηλW wu
            self.users_latent_model[user] = user_vector+ (self.learning_rate*yuij*(item_vector_pos - item_vector_neg))- (self.learning_rate*self.reg_user*user_vector)

            #hi ← hi + η yuij wu − η λH+ hi
            self.items_latent_model[pos_item] = item_vector_pos+ (self.learning_rate*yuij*user_vector)- (self.learning_rate*self.reg_pos_item)+item_vector_pos

            #hj ←hj +ηyuij (−wu)−ηλH− hj
            self.items_latent_model[neg_item] = item_vector_neg+ (self.learning_rate*yuij*(-user_vector))- (self.learning_rate*self.reg_neg_item)-item_vector_neg

            #η=α·η
            self.learning_rate = self.learning_rate*self.learning_rate_schedule


    def sign_func(self, value):
        sign = lambda x : copysign(1, x)
        return sign(value)

    def sample_candidate_negative_items(self, user, pos_item):

        positive_items_lst = list()
        negative_items_lst = list()
        #check the positive items for the user (items the user interacted with)
        for pair in self.reservoir:
            if pair[0] == user:
                positive_items_lst.append(pair[1])
        #check the negative items for the user (items the user didn't interacte with)
        for pair in self.reservoir:
            if pair[1] not in positive_items_lst:
                negative_items_lst.append(pair[1])

        if len(negative_items_lst) <= self.negative_items_num:
            return negative_items_lst
        else:
            return random.sample(negative_items_lst,self.negative_items_num) if len(negative_items_lst) != 0 else []

    # sample one negative item from the list of negative items for the user
    def sample_negative_item(self,positive_item,negative_items_lst):
        item_vector_pos = self.items_latent_model[positive_item]

        prob_dist=list()

        for item in negative_items_lst:
            item_vector_neg=self.items_latent_model[item]
            #the probabilty distribution is defined based on the euclidean distance between the positive item and every negative item
            dist = distance.euclidean(item_vector_pos,item_vector_neg)
            val = 1/dist if dist!=0 else 0
            prob_dist.append(val)

        negative_item = (choice(negative_items_lst, 1, prob_dist))[0] if len(negative_items_lst)!=0 else None

        return negative_item

    def get_recommendations_matrixFact(self, user):
        user_vector = self.users_latent_model[user]
        score = np.dot(self.items_latent_model, user_vector)
        dict_score = {index: value for index, value in enumerate(score)}

        sorted_score = OrderedDict(sorted(dict_score.items(), key=lambda x: x[1], reverse=True))

        return sorted_score

    def batch_training(self,train):
        self.batch_matrixFact(train)


    def get_recommendations(self,user):
        sorted_score = self.get_recommendations_matrixFact(user)
        return sorted_score

    def incremental_training(self,user,item):
        self.update_user_item_models_incremental(user,item)
