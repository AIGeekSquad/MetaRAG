# Description: This file contains the implementation base Loader component of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from abc import ABC, abstractmethod
from llama_index.core.schema import BaseNode
from typing import List


## TODO: Add interface and implementation for load data that calls function on correct data loader
class DataLoaderBaseComponent(ABC):
    """
    Abstract base class for data loader components.
    This class defines the interface for data loader components that are responsible for loading data content,
    storing documents in a knowledge graph, and creating documents from loaded data. Subclasses must implement
    the following abstract methods:
    Methods
    -------
    load_data_content(**kwargs) -> List[BaseNode]
        Return a list of BaseNode objects representing the loaded data content.
    store_document(**kwargs) -> bool
        Store a document in the knowledge graph and return True if the operation is successful.
    load_documents_from_data(**kwargs) -> bool
        Create documents from the loaded data and return True if the operation is successful.
    """
    @abstractmethod
    def load_data_content(
        self,
        **kwargs
    ) -> List[BaseNode]:
        """Return List of BaseNodes"""

    @abstractmethod
    def store_document(
        self,
        **kwargs
    ) -> bool:
        """Store document in knowledge graph, return true if successful"""

    @abstractmethod
    def load_documents_from_data(
        self,
        **kwargs
    ) -> bool:
        """Create Documents from Data Loaded, return true if successful"""