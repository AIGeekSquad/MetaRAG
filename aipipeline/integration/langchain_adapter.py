# Description: This file contains the implementation of the Langchain adapter for integration w/LlamaIndex of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import asyncio
from typing import Any, Dict, List, cast

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

from aipipeline.query.query_helpers import QuerySvcEnvs
from aipipeline.query.provider_factory import ProviderFactory, QuerySvcProvider


## Update to use source nodes from retriever vs. query - TEW

class LlamaIndexLangChainRetriever(BaseRetriever):
    """
    `LlamaIndexLangChainRetriever` is a retriever that integrates LlamaIndex with Langchain for question-answering with sources over a LlamaIndex data structure.
    Attributes:
        index (Any): LlamaIndex index to query.
        index_retriever (Any): LlamaIndex retriever to get nodes from.
        query_kwargs (Dict): Keyword arguments to pass to the query method.
    Methods:
        _aget_relevant_documents(query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
            Asynchronously retrieves documents relevant for a query.
        _get_relevant_documents(query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
            Retrieves documents relevant for a query by running the asynchronous method synchronously.
    """
    index: Any
    index_retriever: Any
    query_kwargs: Dict = Field(default_factory=dict)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        """
        Asynchronously retrieve documents relevant to a given query.
        This method uses the LlamaIndex retriever to fetch documents that are relevant
        to the provided query string. If the retriever is not already set, it initializes
        one using the QuerySvcProvider.
        Args:
            query (str): The query string for which relevant documents are to be retrieved.
            run_manager (AsyncCallbackManagerForRetrieverRun): The run manager for handling
                asynchronous callbacks during the retrieval process.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            List[Document]: A list of Document objects that are relevant to the query.
        Raises:
            ImportError: If the required `llama-index` package is not installed.
        """
        try:
            from llama_index.core.base.response.schema import Response
            from llama_index.core.indices.base import BaseGPTIndex
            from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )

        llamaidx_retriever: LlamaBaseRetriever = None

        if self.index_retriever is None:
            provider: QuerySvcProvider = ProviderFactory.create()
            QuerySvcEnvs.set_querysvc_config(configuration=None)
            llamaidx_retriever = provider.retriever
        else:    
            llamaidx_retriever = cast (LlamaBaseRetriever, self.index_retriever)
        
        response_nodes = llamaidx_retriever.retrieve(query)
        
        #index = cast(BaseGPTIndex, self.index)
        #response = index.query(query, **self.query_kwargs)
        #response = cast(Response, response)
        
        # parse source nodes
        docs = []
        for source_node in response_nodes:
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.get_content(), metadata=metadata)
            )
        return docs

    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun , **kwargs: Any,
    ) -> List[Document]:
        docs = asyncio.run( self._aget_relevant_documents(query=query, run_manager=run_manager, **kwargs))
        return docs
