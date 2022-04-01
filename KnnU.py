import numpy as np
from collections import  defaultdict
from collections import OrderedDict
from math import sqrt
from Model import Model
from scipy.sparse import lil_matrix

'''
this class implements the incremental version of user based approach for the KNN algorithm
referenced paper: Item-Based and User-Based Incremental Collaborative Filtering for Web Recommendations
'''

class KnnU(Model):
    def __init__(self, nb_neighbors, nb_items, nb_users):

        self.K = int(nb_neighbors)
        self.nb_users = nb_users
        self.nb_items = nb_items
        self.freq_matrix = lil_matrix((self.nb_users, self.nb_users))
        self.sim_matrix = lil_matrix((self.nb_users,self.nb_users))
        self.users_history = defaultdict(list)
        self.items_history = defaultdict(list)

    # frequency matrix represents number of items each pair of users interacted with
    def compute_frequency_matrix(self):
        count = 0
        for i in range(self.nb_users):
            for j in range(self.nb_users):
                lst = self.intersection(self.users_history[i], self.users_history[j])
                self.freq_matrix[i,j] = len(lst)
                count = count+1

    # similarity matrix represents the similarity between each pair of users using the cosine similarity measure
    def compute_similarity_matrix(self):
        for i in range(self.nb_users):
            for j in range(self.nb_users):
                self.sim_matrix[i,j] = self.cosine_similarity(i,j)

    # get the recommendations for each user
    def get_recommendations_KnnU(self, user):
        user_history = self.get_user_history(user)

        items_lst = list(range(0, self.nb_items))

        user_unseen_items = set(items_lst) - set(user_history)
        user_unseen_items = list(user_unseen_items)

        user_neighbors = self.get_user_neighbors(user)

        score = {}
        for item in user_unseen_items:
            item_history = self.get_item_history(item)
            val = self.compute_score_user(item_history,user_neighbors,user)
            score[item] = val

        item_min_score = min(score.keys(), key=(lambda k: score[k]))
        value_min_score = score[item_min_score]
        for item in user_history:
            score[item] = value_min_score

        sorted_score = OrderedDict(sorted(score.items(), key=lambda x: x[1], reverse=True))
        return sorted_score

    def compute_score_user(self, item_history, user_neighbors, user):
        user_neighbors_in_item_history=self.intersection(user_neighbors,item_history)
        activation_weight_num = 0
        activation_weight_den = 0

        for i in user_neighbors_in_item_history:
            activation_weight_num = activation_weight_num+self.sim_matrix[user,i]
        for i in user_neighbors:
            activation_weight_den = activation_weight_den+self.sim_matrix[user,i]

        if activation_weight_den == 0 :
            return 0
        return activation_weight_num/activation_weight_den


    #returns the intersection between two lists
    def intersection(self,lst1, lst2):
        lst3 = set(lst1) & set(lst2)
        lst3 = list(lst3)
        return lst3


    def cosine_similarity(self,user_i, user_j):
        #sim(i, j) = cos( i , j ) = #(I ∩ J)/(√#I × √#J)
        if(self.freq_matrix[user_i,user_i] != 0 and self.freq_matrix[user_j,user_j] != 0):
            return self.freq_matrix[user_i,user_j]/((sqrt(self.freq_matrix[user_i,user_i]))*(sqrt(self.freq_matrix[user_j,user_j])))
        return 0

    def get_user_neighbors(self, user):
        # get the row for a specific user
        user_sim = self.sim_matrix[user]
        # sort them according to indices in an ascending order
        user_sim = np.argsort(user_sim)
        # return top k indices of sorted array
        user_neighbors = user_sim[::-1][:self.K]

        return user_neighbors



    def get_user_history(self,user):
        return self.users_history[user]

    def get_item_history(self,item):
        return self.items_history[item]

    '''
    returns two dictionaries:
    1. users_history: users as keys, for each user the list of items interacted with
    2. items_history: items as keys, for each item the list of users interacted with that item
    '''
    def compute_users_items_history(self):

        for index, row in self.train.iterrows():
            #computing users history
            #userId is the first column so row["userID"] is equivalent to row[0]
            if row[0] not in self.users_history:
                self.users_history[row[0]] = [row[1]]
            elif type(self.users_history[row[0]]) == list:
                self.users_history[row[0]].append(row[1])
            else:
                self.users_history[row[0]] = [self.users_history[row[0]],row[1]]


            #computing items history
            if row[1] not in self.items_history:
                self.items_history[row[1]] = [row[0]]

            elif type(self.items_history[row[1]]) == list:
                self.items_history[row[1]].append(row[0])
            else:
                self.items_history[row[1]] = [self.items_history[row[1]],row[0]]

    # increment the frequency of each of the users that has been interacted with the new item
    def update_frequency_matrix(self,user,item):
        item_history=self.get_item_history(item)
        self.freq_matrix[user,user] = self.freq_matrix[user,user]+1
        for user_ in item_history:
            self.freq_matrix[user,user_] = self.freq_matrix[user,user_]+1
            self.freq_matrix[user_,user] = self.freq_matrix[user_,user]+1

    # update the similarity measure of the entries that has been updated in the frequency matrix
    def update_similarity_matrix(self,user,item):
        item_history = self.get_item_history(item)
        self.sim_matrix[user,user] = self.cosine_similarity(user,user)
        for user_ in item_history:
            self.sim_matrix[user,user_] = self.cosine_similarity(user,user_)
            self.sim_matrix[user_,user] = self.cosine_similarity(user_,user)



    def update_users_history(self,user, item):
        if user not in self.users_history:
            self.users_history[user] = [item]

        elif type(self.users_history[user]) == list:
            self.users_history[user].append(item)
        else:
            self.users_history[user] = [self.users_history[user],item]

    def update_items_history(self,user, item):
        if item not in self.items_history:
            self.items_history[item] = [user]

        elif type(self.items_history[item]) == list:
            self.items_history[item].append(user)
        else:
            self.items_history[item] = [self.items_history[item],user]



    def batch_training(self,train):
        self.train = train
        self.compute_users_items_history()
        self.compute_frequency_matrix()
        self.compute_similarity_matrix()


    def get_recommendations(self,user):
        sorted_score = self.get_recommendations_KnnU(user)
        return sorted_score


    def incremental_training(self,user,item):
        self.update_frequency_matrix(user,item)
        self.update_similarity_matrix(user,item)
        self.update_users_history(user,item)
        self.update_items_history(user,item)
