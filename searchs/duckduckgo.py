from .search import Search

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


class DuckDuckGoSearch(Search):

    def search(self, query, limit):
        return self._search(query, limit)

    def _search(self, query, limit):
        api_wrapper = DuckDuckGoSearchAPIWrapper(
            region="ru-ru", max_results=limit)
        search = DuckDuckGoSearchResults(
            api_wrapper=api_wrapper, output_format="list", num_results=limit)
        result = search.invoke(query)
        return result
