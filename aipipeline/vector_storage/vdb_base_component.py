# Description: This file contains the base implementation of VectorDB storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document, BaseNode

from aipipeline.utilities.constants import VECTORDB_TYPE, SIMILARITY_TYPE

@dataclass
class VectorDBBaseComponentPoint(): 
    """
    A class representing a point in a vector database.
    Attributes:
        score (float): The score associated with the point.
        payload (Dict[str, Any]): The payload containing additional data for the point.
    """
    score: float
    payload: Dict[str, Any]


class VectorDBBaseComponent(ABC):
    """
    Abstract base class for a Vector Database Component.
    Methods
    -------
    create_collection(collection_name: str, vdb_type: VECTORDB_TYPE, similarity: Optional[SIMILARITY_TYPE] = SIMILARITY_TYPE.COSINE)
        Abstract method to create a collection for vectors.
    store_vectors(vdb_type: VECTORDB_TYPE, collection_name: str, vector: List[float], metadata: Dict[str, Any]) -> Any
        Upsert vectors into the vector database, return true if successful.
    get_collection_data(**kwargs) -> Any
        Get collection and vector data.
    search_vectors(collection_name: str, query_vector: List[float], query_vector_name: Optional[str] = None, top_k: Optional[int] = 5) -> List[VectorDBBaseComponentPoint]
        Perform Approximate Nearest Neighbor (ANN) search by vector.
    """
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vdb_ype: VECTORDB_TYPE,
        similiarity: Optional[SIMILARITY_TYPE] = SIMILARITY_TYPE.COSINE
    ):
        """Create Collection for Vectors"""

    def store_vectors(
        self,
        vdb_type: VECTORDB_TYPE,
        collection_name: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> Any:
        """Upsert vectors into vectordb, return true if successful"""

    def get_collection_data(
        self,
        **kwargs
    ) -> Any:
        """Get Collection and Vector data"""

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        query_vector_name: Optional[str] = None,
        top_k: Optional[int] = 5, 
    ) -> List[VectorDBBaseComponentPoint]:
         """Do ANN Search by vector"""
