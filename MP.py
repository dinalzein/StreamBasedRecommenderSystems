from collections import  defaultdict
from collections import OrderedDict
from Model import Model

# this class implements the Most Popular model: recommending most popular items for each user


class MP(Model):
    def __init__(self):
        pass


    def batch_mostPop(self):
        self.item_frequency()
        self.sorted_score = OrderedDict(sorted(self.items_freq.items(), key = lambda x: x[1], reverse = True))

    # calulates the frequency for each item (how many users interated with each item)
    def item_frequency(self):
        self.items_freq = defaultdict(list)
        for index, row in self.train.iterrows():
            if row[1] not in self.items_freq:
                self.items_freq[row[1]]=1
            else:
                self.items_freq[row[1]]=self.items_freq[row[1]]+1


    def get_recommedations_mostPop(self,user):
        return self.sorted_score

    #increments the frequency of the item in the <user,item> observation
    def update_mostPop(self, user, item):
        if item not in self.sorted_score:
            self.sorted_score[item] = 1
        else:
            self.sorted_score[item] = self.sorted_score[item]+1
        self.sorted_score = OrderedDict(sorted(self.sorted_score.items(), key=lambda x: x[1], reverse=True))



    def batch_training(self, train):
        self.train = train
        self.batch_mostPop()


    def get_recommendations(self, user):
        sorted_score = self.get_recommedations_mostPop(user)
        return sorted_score

    def incremental_training(self, user, item):
        self.update_mostPop(user,item)
