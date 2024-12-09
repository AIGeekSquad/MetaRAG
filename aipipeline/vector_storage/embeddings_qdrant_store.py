
from aipipeline.utilities.constants import SIMILARITY_TYPE, VECTORDB_TYPE
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent, VectorDBBaseComponentPoint

from qdrant_client import QdrantClient
from typing import Any, Dict, List,Optional
  

class Qdrant_store(VectorDBBaseComponent):
    """
    Qdrant_store is a class that provides an interface to interact with a Qdrant vector database.
    Methods
    -------
    __init__(client: QdrantClient)
        Initializes the Qdrant_store with a QdrantClient instance.
    create_collection(collection_name: str, vdb_type: VECTORDB_TYPE, similiarity: Optional[SIMILARITY_TYPE] = SIMILARITY_TYPE.COSINE)
        Creates a collection for storing vectors in the vector database.
    store_vectors(vdb_type: VECTORDB_TYPE, collection_name: str, vector: List[float], metadata: Dict[str, Any]) -> Any
        Upserts vectors into the vector database and returns true if successful.
    get_collection_data(**kwargs) -> Any
        Retrieves collection and vector data from the vector database.
    search_vectors(collection_name: str, query_vector: List[float], query_vector_name: Optional[str] = None, top_k: Optional[int] = 5) -> List[VectorDBBaseComponentPoint]
        Performs an approximate nearest neighbor (ANN) search by vector and returns a list of VectorDBBaseComponentPoint objects.
    """

    def __init__(self, client:QdrantClient):
        self._client = client
    
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
         points = self._client.search(collection_name = collection_name, query_vector=  query_vector,limit= top_k)
         return[VectorDBBaseComponentPoint(score=point.score, payload=point.payload) for point in points]

  