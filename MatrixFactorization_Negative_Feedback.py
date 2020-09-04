import numpy as np
from collections import OrderedDict
from Model import Model
import copy
from queue import Queue

'''
this class implements the Matrix Factorization algorithm using negtaive feedback
referenced paper: "Collaborative filtering with recency-based negative feedback"
'''

class MatrixFactorization_Negative_Feedback(Model):
    def __init__(self,  nb_features, epoch, regularization, learning_rate, nb_items, nb_users):


        self.epoch = int(epoch)
        self.nb_features = int(nb_features)
        self.reg = regularization #regularization parameter
        self.learning_rate = learning_rate
        self.init_mean = 1
        self.init_stdv = 1
        self.nb_items = nb_items
        self.nb_users = nb_users
        self.negative_items = 5


        self.users_latent_model=np.random.normal(self.init_mean, self.init_stdv, (self.nb_users, self.nb_features)) #n*k
        self.items_latent_model=np.random.normal(self.init_mean, self.init_stdv,(self.nb_items,self.nb_features)) #m*k   R is n*m

        self.queue=Queue()

    def batch_matrixFact(self, train):
        for count in range(self.epoch):
            for index, row in train.iterrows():
                user = row[0]
                item = row[1]
                self.update_user_item_models_with_NF(user,item)


    # get the recommendations for the identified user
    def get_recommendations_matrixFact(self, user):
        user_vector = self.users_latent_model[user]
        score = np.dot(self.items_latent_model, user_vector)
        dict_score = {index: value for index, value in enumerate(score)}
        sorted_score = OrderedDict(sorted(dict_score.items(), key=lambda x: x[1], reverse=True))

        return sorted_score

    # updates the model using for positive items
    def update_user_item_models(self, user, item):
        user_vector=copy.copy(self.users_latent_model[user])
        item_vector=copy.copy(self.items_latent_model[item])

        error= 1- np.dot(user_vector,item_vector)
        self.users_latent_model[user]=np.add(user_vector , (self.learning_rate*(np.subtract(error* item_vector , self.reg * user_vector))))
        self.items_latent_model[item]=np.add(item_vector , (self.learning_rate*(np.subtract(error* user_vector , self.reg * item_vector))))


    # updates the model using negative items
    def update_user_item_models_with_NF(self, user, item):
        for k in range(0,min(self.negative_items, self.queue.qsize())):
            item_j = self.queue.get()
            user_vector = copy.copy(self.users_latent_model[user])
            item_j_vector = copy.copy(self.items_latent_model[item_j])
            error_u_j = 0- np.dot(user_vector,item_j_vector)
            self.users_latent_model[user] = np.add(user_vector , (self.learning_rate*(np.subtract(error_u_j* item_j_vector , self.reg * user_vector))))
            self.queue.put(item_j)
        #update the model with positive items for the user
        self.update_user_item_models(user, item)
        if item in list(self.queue.queue):
            #remove item from queue
            queue_temp = Queue()
            for i in range(0,self.queue.qsize()):
                y = self.queue.get()
                if (y != item):
                    queue_temp.put(y)
            self.queue = queue_temp
        self.queue.put(item)



    def batch_training(self,train):
        self.batch_matrixFact(train)


    def get_recommendations(self,user):
        sorted_score = self.get_recommendations_matrixFact(user)
        return sorted_score

    def incremental_training(self,user,item):
        self.update_user_item_models_with_NF(user,item)
