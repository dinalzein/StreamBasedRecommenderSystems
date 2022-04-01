import numpy as np
from collections import  defaultdict
from collections import OrderedDict
from math import sqrt
from Model import Model
import sys
from scipy.sparse import lil_matrix

'''
this class implements the incremental version of item based approach for the KNN algorithm
referenced paper: Item-Based and User-Based Incremental Collaborative Filtering for Web Recommendations
'''

class KnnI(Model):
    def __init__(self, nb_neighbors, nb_items):

        self.K = int(nb_neighbors)
        self.nb_items = nb_items
        self.freq_matrix = lil_matrix((self.nb_items, self.nb_items))
        self.sim_matrix = lil_matrix((self.nb_items,self.nb_items))
        self.users_history = defaultdict(list) # a dictionary with the items each user interacted with
        self.items_history = defaultdict(list) # a dictionary with the users interacted with each item

    # the frequency matrix represents number of users interacted with each pair of items
    def compute_frequency_matrix(self):

        for i in range(self.nb_items):
            for j in range(self.nb_items):
                lst = self.intersection(self.items_history[i], self.items_history[j])
                self.freq_matrix[i,j] = len(lst)



    # the similarity matrix represents the similarity between each pair of items using the cosine similarity measure
    def calculate_similarity_matrix(self):
        for i in range(self.nb_items):
            for j in range(self.nb_items):
                self.sim_matrix[i,j]=self.cosine_similarity(i,j)


    # get the recommendations for each user
    def get_recommendations_KnnI(self, user):
        user_history = self.get_user_history(user)
        items_lst = list(range(0, self.nb_items))
        user_unseen_items = set(items_lst) - set(user_history)
        user_unseen_items = list(user_unseen_items)

        score={}
        for item in user_unseen_items:
            item_neighbors = self.get_item_neighbors(item)

            value=self.compute_score_item(user_history,item_neighbors,item)
            score[item] = value

        item_min_score = min(score.keys(), key=(lambda k: score[k]))
        value_min_score=score[item_min_score]
        for item in user_history:
            score[item] = value_min_score
        sorted_score = OrderedDict(sorted(score.items(), key=lambda x: x[1], reverse=True))

        return sorted_score

    def compute_score_item(self, user_history, item_neighbors, item):
        item_neighbors_in_user_history=self.intersection(item_neighbors,user_history)
        activation_weight_num = 0
        activation_weight_den = 0

        for i in item_neighbors_in_user_history:
            activation_weight_num = activation_weight_num+self.sim_matrix[item,i]

        for i in item_neighbors:
            activation_weight_den = activation_weight_den+self.sim_matrix[item,i]

        if activation_weight_den == 0 :
            return 0
        return activation_weight_num/activation_weight_den

    #returns the intersection between two lists
    def intersection(self,lst1, lst2):
        lst3 = set(lst1) & set(lst2)
        lst3 = list(lst3)
        return lst3


    def cosine_similarity(self,item_i, item_j):
        #sim(i, j) = cos( i , j ) = #(I ∩ J)/(√#I × √#J)
        if(self.freq_matrix[item_i,item_i] != 0 and self.freq_matrix[item_j,item_j] != 0):
            return self.freq_matrix[item_i,item_j]/((sqrt(self.freq_matrix[item_i,item_i]))*(sqrt(self.freq_matrix[item_j,item_j])))
        return 0

    def get_item_neighbors(self, item):
          #get the row for a specific item
          item_sim = self.sim_matrix[item]
          #sort them according to their indices in an ascending order
          item_sim = np.argsort(item_sim)
          #return top k indices of the sorted array
          item_neighbors = item_sim[::-1][:self.K]

          return item_neighbors


    def get_user_history(self,user):
        return self.users_history[user]

    '''
    returns two dictionaries:
    1. users_history: users as keys, for each user the list of items interacted with
    2. items_history: items as keys, for each item the list of users interacted with that item
    '''
    def compute_users_items_history(self):

        for index, row in self.train.iterrows():
            #userId is the first column so row["userID"] is equivalent to row[0]
            if row[0] not in self.users_history:
                self.users_history[row[0]] = [row[1]]

            elif type(self.users_history[row[0]]) == list:
                self.users_history[row[0]].append(row[1])
            else:
                self.users_history[row[0]] = [self.users_history[row[0]],row[1]]


            if row[1] not in self.items_history:
                self.items_history[row[1]] = [row[0]]
            elif type(self.items_history[row[1]]) == list:
                self.items_history[row[1]].append(row[0])
            else:
                self.items_history[row[1]] = [self.items_history[row[1]],row[0]]


    # increment the frequency of each of the items the user has interacted with
    def update_frequency_matrix(self,user,item):
        user_history = self.get_user_history(user)
        self.freq_matrix[item,item] = self.freq_matrix[item,item]+1
        for item_ in user_history:
            self.freq_matrix[item,item_] = self.freq_matrix[item,item_]+1
            self.freq_matrix[item_,item] = self.freq_matrix[item_,item]+1


    # update the similarity measure of the entries that has been updated in the frequency matrix
    def update_similarity_matrix(self,user,item):
        user_history = self.get_user_history(user)
        self.sim_matrix[item,item] = self.cosine_similarity(item,item)

        for item_ in user_history:
            self.sim_matrix[item,item_] = self.cosine_similarity(item,item_)
            self.sim_matrix[item_,item] = self.cosine_similarity(item_,item)


    def update_users_history(self,user, item):
        if user not in self.users_history:
            self.users_history[user] = [item]

        elif type(self.users_history[user]) == list:
            self.users_history[user].append(item)
        else:
            self.users_history[user] = [self.users_history[user],item]



    def batch_training(self,train):
        self.train = train
        self.compute_users_items_history()
        self.compute_frequency_matrix()
        self.calculate_similarity_matrix()


    def get_recommendations(self,user):
        sorted_score = self.get_recommendations_KnnI(user)
        return sorted_score

    def incremental_training(self,user,item):
        self.update_frequency_matrix(user,item)
        self.update_similarity_matrix(user,item)
        self.update_users_history(user,item)
