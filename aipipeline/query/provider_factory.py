# Description: This file contains the implementation of Query Server factory class of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from dataclasses import dataclass
from typing import Optional, cast
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import BaseQueryEngine
from qdrant_client import QdrantClient

from aipipeline.query.query_helpers import QuerySvcEnvs, get_query_engine, get_retriever
from aipipeline.utilities.constants import GRAPHDB_TYPE, VECTORDB_TYPE
from aipipeline.utilities.gdb_utilities import get_document_graph_store
from aipipeline.utilities.vdb_utilities import get_vDBClient
from aipipeline.vector_storage.embeddings_qdrant_store import Qdrant_store



import logging

logger = logging.getLogger(__name__)
@dataclass
class QuerySvcProvider:
    """
    QuerySvcProvider is a service provider class that encapsulates the functionality
    of a retriever and a query engine.
    Attributes:
        retriever (BaseRetriever): An instance of a class that implements the BaseRetriever interface.
        query_engine (BaseQueryEngine): An instance of a class that implements the BaseQueryEngine interface.
    """
    retriever: BaseRetriever
    query_engine: BaseQueryEngine
        
class ProviderFactory:
        @staticmethod
        def create(vector_db_collection_name: Optional[str] = None, kg_db_type: GRAPHDB_TYPE = GRAPHDB_TYPE.NEO4J) -> QuerySvcProvider :
            """
            Creates a QuerySvcProvider instance.
            This function initializes the configuration for the Query Service, connects to the specified vector database,
            and sets up the knowledge graph database. It then creates a retriever and query engine based on the provided
            configurations and returns a QuerySvcProvider instance.
            Args:
                vector_db_collection_name (Optional[str]): The name of the vector database collection. If not provided,
                                                            it defaults to the collection name specified in the configuration.
                kg_db_type (GRAPHDB_TYPE): The type of knowledge graph database to use. Defaults to GRAPHDB_TYPE.NEO4J.
            Returns:
                QuerySvcProvider: An instance of QuerySvcProvider initialized with the configured retriever and query engine.
            """

            conf = QuerySvcEnvs.CONFIGURATION
            if conf is None:
                QuerySvcEnvs.set_querysvc_config(configuration=None, model_configuration=None)
                
            vector_client = None
            match QuerySvcEnvs.CONFIGURATION.vector_database: 
                case "QDRANT":
                    logger.info("Connecting to Qdrant")
                    vector_client = cast(QdrantClient, get_vDBClient(VECTORDB_TYPE.QDRANT))
                    vector_db = Qdrant_store(vector_client)
                case "MILVUS":
                    logger.info("Connecting to Milvus")
                    vector_client = get_vDBClient(VECTORDB_TYPE.MILVUS)
                    #vector_db = Milvus_store(vector_client)
                case "WEAVIATE":
                    logger.info("Connecting to Weaviate")
                    vector_client = get_vDBClient(VECTORDB_TYPE.WEAVIATE)
                    #vector_db = Weaviate_store(vector_client)
                case "AZAISEARCH":
                    logger.info("Connecting to Azure AI Search")
                    vector_client = get_vDBClient(VECTORDB_TYPE.AZUREAISEARCH)
                    #vector_db = AzAI_store(vector_client)
        
            #Switch this to use the KG option once other storage types available
            #kg_db = get_document_graph_store(GRAPHDB_TYPE.NEO4J)
            kg_db = get_document_graph_store(kg_db_type)
            collection_name = vector_db_collection_name

            if (collection_name is None):
                collection_name = QuerySvcEnvs.CONFIGURATION.vector_database_collection_name

            logger.info(f"Creating retriever for collection {collection_name}")
            retriever = get_retriever(kg_db, vector_db, collection_name)
            query_engine = get_query_engine(retriever = retriever)

            return QuerySvcProvider( retriever, query_engine)