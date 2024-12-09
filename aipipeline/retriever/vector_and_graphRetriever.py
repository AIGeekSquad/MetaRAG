# Description: This file contains the implementation of Vector and Graph Retreiver of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import asyncio
from typing import List, Optional

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model

from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

import logging

logger = logging.getLogger(__name__)

class VectorAndGraphRetriever(BaseRetriever):
    """
    A retriever that combines vector search and graph-based retrieval.
    Args:
        doc_graph (DocumentGraphBaseComponent): The document graph component.
        vector_store (VectorDBBaseComponent): The vector database component.
        vector_collection (str): The name of the vector collection to search in.
        similarity_top_k (int, optional): The number of top similar results to retrieve. Defaults to DEFAULT_SIMILARITY_TOP_K.
        embed_model (Optional[BaseEmbedding], optional): The embedding model to use. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): The callback manager for handling callbacks. Defaults to None.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
    Methods:
        _aretrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Asynchronously retrieves nodes based on the query bundle using vector search and graph-based retrieval.
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Synchronously retrieves nodes based on the query bundle using vector search and graph-based retrieval.
    """
    def __init__(
        self,
        doc_graph: DocumentGraphBaseComponent,
        vector_store: VectorDBBaseComponent,
        vector_collection: str,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        embed_model: Optional[BaseEmbedding] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        ) -> None:
        self._doc_graph = doc_graph
        self._vector_store = vector_store
        self._vector_collection = vector_collection
        self._similarity_top_k = similarity_top_k
        self._embed_model = resolve_embed_model(embed_model)
        super().__init__(
            callback_manager=callback_manager,
            verbose=verbose,        
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronously retrieves a list of nodes with scores based on the provided query bundle.
        Args:
            query_bundle (QueryBundle): The query bundle containing the query information, including embeddings and embedding strings.
        Returns:
            List[NodeWithScore]: A list of nodes with their corresponding scores.
        Notes:
            - If the query bundle does not have an embedding but has embedding strings, it generates the embedding using the embedding model.
            - Searches for vectors in the vector store using the query embedding.
            - Searches for corresponding nodes in the document graph based on the vector search results.
        """
        result: List[NodeWithScore] = []
        if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
            embed_model = self._embed_model
            query_bundle.embedding = (
                await embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
            
        logger.info("Searching vectors")
        points = self._vector_store.search_vectors(self._vector_collection, query_bundle.embedding, self._similarity_top_k )
            
        logger.info("Searching graph")
        #TODO: Update to KG Base
        for point in points:
            node = self._doc_graph.get_chunk_node_for_id(point.payload["nodeId"])
            if(node is not None):
                result.append(NodeWithScore(node=node, score=point.score))

        logger.info("Retrieved {} nodes".format(len(result)))
        return result
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = asyncio.run( self._aretrieve(query_bundle))
        return results
