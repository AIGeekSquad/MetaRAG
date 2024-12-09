# Description: This file contains the main entry point of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import json
import sys
import os


from aipipeline.utilities.constants import DATALOADER_TYPE, IngestionEnvs
from aipipeline.utilities.data_ingest_utilities import get_data_file_list, get_data_type, get_ingest_config_from_json, validate_vdb_collection, write_embedding
from aipipeline.utilities.loader_utilities import process_data_files
from aipipeline.config.config_param_types import IngestionConfigurationParm
from aipipeline.query.query_helpers import get_embed_model, get_llm_model

import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)

def process_data(directory: str, collection_name: str, config_type: str=None, config: IngestionConfigurationParm=None, **datargs):
    """
    Processes data from a specified directory or Wikipedia topics and writes embeddings to a collection.
    Args:
        directory (str): The directory containing data files to process.
        collection_name (str): The name of the collection to write embeddings to.
        config_type (str, optional): The type of configuration to use. Defaults to None.
        config (IngestionConfigurationParm, optional): The configuration parameters for ingestion. Defaults to None.
        **datargs: Additional arguments for data processing.
    Returns:
        None
    """
    loadertype: DATALOADER_TYPE = DATALOADER_TYPE.NONE
    loader_args = {}
    data_files = []


    llm_model = get_llm_model(config.LLMModelConfig)
    multimodal_llm_model = get_llm_model(config.MultiModalModelConfig) if config.MultiModalModelConfig is not None else None
    if config.MultiModalModelConfig is not None:
        multimodal_llm_model = get_llm_model(config.MultiModalModelConfig)
    embeddings_model = get_embed_model(config.EmbedModelConfig)

    ## Check if loading data via wikipedia
    ## Process data from wikipedia categories
    if datargs.get("topics") is not None:
        loadertype = DATALOADER_TYPE.WIKIPEDIA
        loader_args = {"topics": datargs.get("topics"), 
                       "max_page_per_topic": datargs.get("page_count_per_topic")}
        
        logger.info(f"Processing data from wikipedia topics: {loader_args.get('topics')}")
        
        nodes = process_data_files(
            None, 
            loadertype, 
            llm_model_to_use=llm_model,
            embed_model_to_use=embeddings_model,
            multi_modal_llm_model_to_use=multimodal_llm_model,
            **loader_args)
        
        write_embedding(nodes, collection_name=collection_name, embed_model=embeddings_model)
    
    ## Check if loading data via files in directory
    if directory is None: return

    ## Get file paths from directory and subdirectories
    data_files = get_data_file_list(directory=directory)

    ## Discern file type & set args
    for file in data_files:
        
        loadertype, loader_args = get_data_type(file)
        logger.info(f"Processing data from file: {file.path} of type {loadertype.name}")
        nodes = process_data_files(
            str(file.path),
            loadertype, 
            llm_model_to_use=llm_model,
            embed_model_to_use=embeddings_model,
            multi_modal_llm_model_to_use=multimodal_llm_model,
            **loader_args)
        
        write_embedding(nodes, collection_name=collection_name, embed_model=embeddings_model)

## Take in the path from sessions or anything else as directory
## Take in categories from sessions or anything else
directory_to_process = os.getenv('DATA_DIR')
topics_to_process_json_string = os.getenv('TOPIC_LIST') if os.getenv('TOPIC_LIST') is not None else None
vdb_collection_name = os.getenv('COLLECTION_NAME') if os.getenv('COLLECTION_NAME') is not None else IngestionEnvs.vDB_COLLECTION_NAME
ingest_config_json = os.getenv('CONFIG') if os.environ.get('CONFIG') is not None else None
ingest_config_type = os.getenv('CONFIG_TYPE') if os.environ.get('CONFIG_TYPE') is not None else None

ingest_config: IngestionConfigurationParm = None

topics_to_process = None
if topics_to_process_json_string is not None:
    topics_to_process = json.loads(topics_to_process_json_string)

logger.info(f"loading config: {ingest_config_json}")
ingest_config = get_ingest_config_from_json(ingest_config_json)
validate_vdb_collection(collection_name = vdb_collection_name, vector_size=3072)
logger.info(f"Processing data from directory: {directory_to_process}")
process_data(directory=directory_to_process, 
             collection_name=vdb_collection_name, 
             config_type=ingest_config_type,
             config=ingest_config,
             topics=topics_to_process)