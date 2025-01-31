from abc import ABC, abstractmethod


class Search(ABC):

    @abstractmethod
    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        pass
