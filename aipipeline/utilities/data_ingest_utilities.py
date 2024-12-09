# Description: This file contains the implementation of ingestion utilities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import json
import os
import magic
import pandas as pd
from typing import List, cast
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM

from aipipeline.embeddings.embedding_generator import EmbeddingGenerator
from aipipeline.embeddings.embeddings_utilities import EMBED_MODEL_TYPE, create_embed_model
from aipipeline.utilities.constants import DATALOADER_TYPE, IngestionEnvs, VECTORDB_TYPE, GRAPHDB_TYPE
from aipipeline.config.config_param_types import AzureOpenAIConfigParam, LLMConfigParam, IngestionConfigurationParm
from aipipeline.utilities.vdb_utilities import get_vDBClient
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.utilities.gdb_utilities import get_document_graph_store

import logging


logger = logging.getLogger(__name__)

logger.info("Initializing Data Ingestion pipeline")
ada_embed_model = create_embed_model(embed_model_type= EMBED_MODEL_TYPE.AZURE_ADA)

qdrant_client = cast(QdrantClient, get_vDBClient(VECTORDB_TYPE.QDRANT))
kg_doc_store: DocumentGraphBaseComponent = get_document_graph_store(graphDB=GRAPHDB_TYPE.NEO4J)

## Discern file type & set args
def get_data_type(filetoprocess: os.DirEntry):
    """
    Determines the type of data loader to use based on the file type of the given file.
    Args:
        filetoprocess (os.DirEntry): The file to process.
    Returns:
        tuple: A tuple containing:
            - type_of_loader (DATALOADER_TYPE): The type of data loader determined.
            - args_for_loader (dict): The arguments required for the data loader.
    Supported file types and corresponding loaders:
        - "application/pdf": DATALOADER_TYPE.PDF
        - "video/mp4": DATALOADER_TYPE.VIDEO
        - "text/csv": DATALOADER_TYPE.CSV
        - "application/json": DATALOADER_TYPE.WEB
    If the file type is not supported, an error is logged.
    """
    type_of_loader: DATALOADER_TYPE = DATALOADER_TYPE.NONE
    args_for_loader = {}

    file_type = magic.from_file(filetoprocess.path, mime=True).lower()

    if file_type == 'text/plain':
        try :
            df = pd.read_csv(filetoprocess.path)
            file_type = "text/csv"
        except Exception as e:
            pass

    match file_type:
        case "application/pdf":
            type_of_loader = DATALOADER_TYPE.PDF
            args_for_loader = {"file_path": filetoprocess.path}
        case "video/mp4":
            type_of_loader = DATALOADER_TYPE.VIDEO
            args_for_loader = {"vid_filepath": filetoprocess.path}
        case "text/csv":
            type_of_loader = DATALOADER_TYPE.CSV
            args_for_loader = {"csv_rootdir": filetoprocess.path}
        case "application/json":
            web_urls=[]
            with open(filetoprocess, 'r') as f:
                url_data = f.read()
                url_json = json.loads(url_data)
                for weburl in url_json["urls"]: 
                    web_urls.append(weburl["url"])
            type_of_loader = DATALOADER_TYPE.WEB
            args_for_loader = {"web_urls": web_urls}
        case _:
            logger.error(f"File type {filetoprocess} not supported")
            ## check if empty

    return type_of_loader, args_for_loader

def get_data_file_list(directory: str) -> List[os.DirEntry]:
    """
    Recursively retrieves a list of all files in the given directory and its subdirectories.
    Args:
        directory (str): The path to the directory to scan for files.
    Returns:
        List[os.DirEntry]: A list of os.DirEntry objects representing the files found in the directory and its subdirectories.
    """
    file_list: List[os.DirEntry] = []

    data = [f for f in os.scandir(directory)]
    for d in data:
        print(d)
        if d.is_dir():
            dir_file_list: List[os.DirEntry] = []
            dir_file_list = get_data_file_list(d.path)
            file_list.extend(dir_file_list)        
        else: file_list.append(d)
    
    return file_list

def write_embedding(nodes:List[BaseNode], collection_name: str, embed_model: BaseEmbedding =None, llm: LLM = None):
    """
    Generates and writes embeddings for a list of nodes into a specified collection.
    Args:
        nodes (List[BaseNode]): A list of nodes for which embeddings need to be generated.
        collection_name (str): The name of the collection where the embeddings will be stored.
        embed_model (BaseEmbedding, optional): The embedding model to use for generating embeddings. Defaults to None.
        llm (LLM, optional): The language model to use for processing nodes. Defaults to None.
    Returns:
        None
    """

    if embed_model is not None:
        ada_embed_model = embed_model

    EmbeddingGenerator.generate_vectors(
         embed_model= ada_embed_model,
         document_graph_store=kg_doc_store, 
         client= qdrant_client, 
         collection_name= collection_name, 
         llm=llm,
         nodes_to_process=nodes)

## Change to accept json file not just json text string
def get_ingest_config_from_json(config: str):
    """
    Parses a JSON string to create an IngestionConfigurationParm object. If the JSON string is empty or None,
    it creates a default configuration with predefined values.
    Args:
        config (str): A JSON string representing the ingestion configuration.
    Returns:
        IngestionConfigurationParm: An object containing the ingestion configuration parameters.
    """
    configuration: IngestionConfigurationParm = None
    
    if config is not None and config != "":
        configuration = IngestionConfigurationParm.model_validate_json(config)
    else:
        configuration = IngestionConfigurationParm()
        configuration.LLMModelConfig = LLMConfigParam()
        configuration.LLMModelConfig.model = IngestionEnvs.DEFAULT_LLM_MODEL
        configuration.LLMModelConfig.llm_model_configuration = AzureOpenAIConfigParam(
            temperature=0.0,
            deployment_name=IngestionEnvs.DEFAULT_LLM_MODEL,
            endpoint= IngestionEnvs.AZURE_ENDPOINT,
            api_key= IngestionEnvs.OAI_API_KEY,
            version=IngestionEnvs.OAI_API_VERSION
            )
    
        configuration.EmbedModelConfig = LLMConfigParam()
        configuration.EmbedModelConfig.model = IngestionEnvs.DEFAULT_EMBED_MODEL
        configuration.EmbedModelConfig.llm_model_configuration = AzureOpenAIConfigParam(
            temperature=0.0,
            deployment_name=IngestionEnvs.DEFAULT_EMBED_MODEL,
            endpoint= IngestionEnvs.AZURE_ENDPOINT,
            api_key= IngestionEnvs.OAI_API_KEY,
            version=IngestionEnvs.OAI_EMBED_API_VERSION
            )
    
    return configuration

# def get_model_from_config(config: BaseModelConfigParam) -> LLM:
#     #llm_model = create_llm_model(LLM_MODEL_TYPE.AZURE_GPT4)
#     pass

# def get_embed_model_from_config(config: BaseModelConfigParam)-> LLM:
#     #llm_model = create_llm_model(LLM_MODEL_TYPE.AZURE_GPT4)
#     pass

def validate_vdb_collection(collection_name: str, vector_size: int = 1536):
    """
    Validates the existence of a collection in the VectorDB and creates it if it does not exist.
    This function checks if a collection with the specified name exists in the VectorDB. If the collection
    does not exist, it creates a new collection with the given name and vector size.
    Args:
        collection_name (str): The name of the collection to validate or create.
        vector_size (int, optional): The size of the vectors in the collection. Defaults to 1536.
    Returns:
        None
    """

    logger.info(f"Validating collection '{collection_name}' in VectorDB")
    list_of_collections = [x.name for x in  qdrant_client.get_collections().collections]
    logger.info(f"Existing collections: {list_of_collections}")

    result = collection_name in list_of_collections
    if result is False:
        logger.info(f"Collection '{collection_name}' does not exist, creating collection")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, 
                                        distance=Distance.COSINE)
        )
        
    logger.info(f"Collection '{collection_name}' is ready for use.")
    