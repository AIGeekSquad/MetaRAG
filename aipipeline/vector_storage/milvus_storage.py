# Description: This file contains the placeholder of Milvus vector storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import List

from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

## Use Base VectorDB or use llamaIndex base - Diego and Tara to Discuss

class MilvusStorage(VectorDBBaseComponent):
    """
    MilvusStorage is a component responsible for storing and searching vectors in Milvus.
    """
    
    @staticmethod
    def store_node_references(client:any, collection_name:str, nodes:List[BaseNode], embed_model:BaseEmbedding):
        # client.upsert(collection_name, points)
        pass
    
    def search(client:any, collection_name:str, query:List[float], top_k:int = 10) -> List[float]:
        search_result = client.search(collection_name, query_vector= query, limit=top_k)
        return search_result
