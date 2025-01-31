from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def inference(self, query, sources):
        pass
