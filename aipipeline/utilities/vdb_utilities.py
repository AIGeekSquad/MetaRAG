# Description: This file contains the implementation of Data loader utilities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

## TODO: Diego and Tara decide to use llamaindex vector abstraction or each vector DB client, GH GH GH

from aipipeline.utilities.constants import VECTORDB_TYPE, IngestionEnvs

from qdrant_client import QdrantClient
import logging
logger = logging.getLogger(__name__)

def get_vDBClient(vectorDB:VECTORDB_TYPE):
    """
    Returns a client instance for the specified vector database type.
    Parameters:
    vectorDB (VECTORDB_TYPE): The type of vector database to connect to.
    Returns:
    QdrantClient or None: A client instance for the specified vector database type, or None if the type is not supported.
    Supported vector database types:
    - VECTORDB_TYPE.QDRANT: Returns a QdrantClient instance.
    - VECTORDB_TYPE.MILVUS: Returns None (not implemented).
    - VECTORDB_TYPE.WEAVIATE: Returns None (not implemented).
    - VECTORDB_TYPE.AZUREAISEARCH: Returns None (not implemented).
    - Any other type: Returns None.
    """
    match vectorDB.value:
        case VECTORDB_TYPE.QDRANT:
            url = IngestionEnvs.QDRANT_HOST
            logger.info(f"Connecting to Qdrant at {url}")
            return QdrantClient(url=url)
        case VECTORDB_TYPE.MILVUS:
            return None
        case VECTORDB_TYPE.WEAVIATE:
            return None
        case VECTORDB_TYPE.AZUREAISEARCH:
            return None
        case _:
            return None 
