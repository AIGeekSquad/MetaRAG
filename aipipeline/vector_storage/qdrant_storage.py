# Description: This file contains the implementation of Qdrant Vector storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint

from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

class QdrantStorage(VectorDBBaseComponent):
    """
    QdrantStorage is a component responsible for storing and searching vectors in Qdrant.
    """
    @staticmethod
    def store_node_references(client:QdrantClient, collection_name:str, nodes:List[BaseNode], embed_model:BaseEmbedding):
        """
        Store node references in a Qdrant collection.
        This function processes a list of nodes, extracts their content, generates embeddings
        using the provided embedding model, and stores the resulting points in the specified
        Qdrant collection.
        Args:
            client (QdrantClient): The Qdrant client used to interact with the Qdrant service.
            collection_name (str): The name of the collection where the points will be stored.
            nodes (List[BaseNode]): A list of nodes to be processed and stored.
            embed_model (BaseEmbedding): The embedding model used to generate dense vectors
                                            from the node content.
        Returns:
            None
        """

        points: List[PointStruct] = []
        for node in nodes:
            type = node.class_name()
            if type == "TextNode":
                text = node.get_content()
                if text is None or text.strip() == "":
                    continue
                dense_vector = embed_model.get_text_embedding(text)
                point = PointStruct(
                    id = node.node_id,
                    vector=dense_vector,
                    payload={"nodeType": type}
                )
                points.append(point)
               
        client.upsert(collection_name, points)
    
    def search(client:QdrantClient, collection_name:str, query:List[float], top_k:int = 10) -> List[ScoredPoint]:
        """
        Searches for the top-k most similar points in a specified collection.
        Args:
            client (QdrantClient): The Qdrant client instance used to perform the search.
            collection_name (str): The name of the collection to search in.
            query (List[float]): The query vector to search for similar points.
            top_k (int, optional): The number of top similar points to return. Defaults to 10.
        Returns:
            List[ScoredPoint]: A list of scored points representing the most similar points to the query vector.
        """
        search_result = client.search(collection_name, query_vector= query, limit=top_k)
        return search_result
