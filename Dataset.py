import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import os

'''
this class loads the data sets and does the train test split
'''


class Dataset(object):

    def __init__(self, interactions_file_name, items_file_name, users_file_name, train_pct, test_pct):
        #initilze the variables
        self.interactions_file_name = interactions_file_name
        self.items_file_name = items_file_name
        self.users_file_name = users_file_name
        self.train_pct = int(train_pct)
        self.test_pct = int(test_pct) if train_pct + test_pct <= 100 else 100 - train_pct
        self.dataset_path = "data/"



    def load_interactions(self):
        interactions = pd.read_csv(self.dataset_path+self.interactions_file_name, names=["userId","itemId","rating","timestamp"], sep=';', engine='python',na_values="?", header=None)

        for index, row in interactions.iterrows():
            row[0]=self.converted_IDs_users[row[0]]
            row[1]=self.converted_IDs_items[row[1]]

        self.interactions=interactions



    def load_items(self):
        '''
        loads the items from the item.csv file in the data folder
        '''
        items = pd.read_csv(self.dataset_path+self.items_file_name, names=["movieId", "movie-title", "release-date", "video-release-date", "IMDb-URL", "unknown", "action",
         "adventure", "animation", "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical",
          "mystery", "romance","sci-fi", "thriller", "war", "western"], sep=';', engine='python',na_values="?", header=None)

        for index, row in items.iterrows():
            items.at[index, 'movieId'] = self.converted_IDs_items[row[0]]

        self.items = items

    def load_users(self):
        '''
        loads the users from the user.csv file in the data folder
        '''

        users = pd.read_csv(self.dataset_path+self.users_file_name, names=["userId", "age", "gender", "occupation", "zip-code"], sep=';', engine='python',na_values="?", header=None)

        for index, row in users.iterrows():
            users.at[index, 'userId']=self.converted_IDs_users[row[0]]

        self.users = users

    #load the main interaction datset from the data.csv file in the data file
    #convert the ids of the of the users and items to be 0,1 ..

    def load_interactions_convert_IDs(self):
        '''
        loads the interaction dataset file in the data folder
        '''
        interactions = pd.read_csv(self.dataset_path+self.interactions_file_name, names = ["userId", "itemID", "rating", "timestamp"], sep=';', engine='python',na_values="?", header=None)
        #taking the data with rating = 5
        indexNames = interactions[interactions['rating']!= 5].index
        interactions.drop(indexNames , inplace = True)
        interactions = interactions.sort_values(by = "timestamp", ascending = True)
        self.converted_IDs_users = {}
        self.converted_IDs_items = {}
        user_id = 0
        item_id = 0

        for index, row in interactions.iterrows():
            #userId is the first column so row["userID"] is equivalent to row[0]
            if row[0] not in self.converted_IDs_users:
                self.converted_IDs_users[row[0]]=user_id
                row[0] = user_id
                user_id = user_id+1

            else:
                row[0] = self.converted_IDs_users[row[0]]
            #itemId is the second column so row["itemID"] is equivalent to row[1]
            if row[1] not in self.converted_IDs_items:
                self.converted_IDs_items[row[1]] = item_id
                row[1] = item_id
                item_id = item_id+1
            else:
                row[1] = self.converted_IDs_items[row[1]]


        self.nb_users = user_id
        self.nb_items = item_id
        self.interactions = interactions


    def split_train_test(self, grid_search):
        # for the grid search, we use a train valid split
        if grid_search:
            train_size = int(self.train_pct*len(self.interactions)/100)
            test_size = int(self.test_pct*len(self.interactions)/100)
            self.train = self.interactions[:train_size]
            self.test = self.interactions[train_size:test_size+train_size]

        else:
            train_size = int(self.train_pct*len(self.interactions)/100)
            self.train = self.interactions[:train_size]
            self.test = self.interactions[train_size:(len(self.interactions)+1)]



    def get_train_test_set(self, grid_search):
        self.load_interactions_convert_IDs()
        self.split_train_test(grid_search)
        return self.train, self.test

    def get_nb_users(self):
        if self.users_file_name != 'None':
            self.load_users()
            self.nb_users = self.users['userId'].nlargest(1).iloc[-1]+1
        return self.nb_users

    def get_nb_items(self):
        if self.items_file_name != 'None':
            self.load_items()
            self.nb_items = self.items['movieId'].nlargest(1).iloc[-1]+1
        return self.nb_items
