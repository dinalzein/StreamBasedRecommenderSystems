from abc import ABC, abstractmethod
# abstract class for the model classes
class Model(ABC):
    @abstractmethod
    def batch_training(self,training_data):
        pass
    @abstractmethod
    def get_recommendations(self,user):
        pass
    @abstractmethod
    def incremental_training(self,user,item):
        pass
