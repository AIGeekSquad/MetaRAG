# Description: This file contains the implementation of Data loader utilities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import List

from llama_index.core.schema import BaseNode
from llama_index.core.llms.llm import LLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from aipipeline.data_loaders.csv_text_loader import CsvAndTextLoader
from aipipeline.data_loaders.loaders_base_component import DataLoaderBaseComponent
from aipipeline.data_loaders.pdf_loader import PdfLoader
from aipipeline.data_loaders.video_transcribe_loader import VideoTranscribeLoader
from aipipeline.data_loaders.web_loader import WebLoader
from aipipeline.data_loaders.wikipedia_loader import WikipediaLoader
from aipipeline.utilities.constants import DATALOADER_TYPE, GRAPHDB_TYPE, LLM_MULTI_MODAL_MODEL_TYPE
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.utilities.llm_utilities import create_multi_modal_llm_model
#from aipipeline.embeddings.embeddings_utilities import EMBED_MODEL_TYPE, create_embed_model

import logging

from aipipeline.utilities.gdb_utilities import get_document_graph_store

logger = logging.getLogger(__name__)

def process_data_files(filepath: str, loader_type: DATALOADER_TYPE, llm_model_to_use: LLM, embed_model_to_use: BaseEmbedding, multi_modal_llm_model_to_use: MultiModalLLM = None, **kwargs) -> List[BaseNode]:
    """
    Processes data files based on the specified loader type and models.
    Args:
        filepath (str): The path to the file to be processed.
        loader_type (DATALOADER_TYPE): The type of data loader to use.
        llm_model_to_use (LLM): The language model to use for processing.
        embed_model_to_use (BaseEmbedding): The embedding model to use for processing.
        multi_modal_llm_model_to_use (MultiModalLLM, optional): The multi-modal language model to use for processing. Defaults to None.
        **kwargs: Additional keyword arguments for specific loader types.
    Returns:
        List[BaseNode]: A list of processed data nodes.
    Raises:
        ValueError: If no appropriate DataLoader is found for the specified data type.
    Loader Types and Corresponding kwargs:
        - DATALOADER_TYPE.PDF: 
            - file_path (str): Path to the PDF file.
            - filename_override (str, optional): Override for the filename.
            - uri (str, optional): URI of the file.
        - DATALOADER_TYPE.VIDEO: 
            - vid_filepath (str): Path to the video file.
        - DATALOADER_TYPE.WEB: 
            - web_urls (List[str]): List of web URLs to process.
        - DATALOADER_TYPE.WIKIPEDIA: 
            - topics (List[str]): List of Wikipedia topics to process.
            - max_page_per_topic (int, optional): Maximum number of pages per topic.
        - DATALOADER_TYPE.CSV: 
            - csv_rootdir (str): Root directory of CSV files.
    """
    dataloader: DataLoaderBaseComponent = None 
    results: List[BaseNode] = [] 

    kg_store: DocumentGraphBaseComponent = get_document_graph_store(GRAPHDB_TYPE.NEO4J)
    llm_model : LLM = llm_model_to_use
    multi_modal_llm_model : MultiModalLLM = None
    embeddings_model: BaseEmbedding = embed_model_to_use

    ##TODO: add kg and model from Config, use defaults right now to test
    #llm_model = create_llm_model(LLM_MODEL_TYPE.AZURE_GPT4)
    if multi_modal_llm_model_to_use is not None:
        multi_modal_llm_model = multi_modal_llm_model_to_use
    else:
        multi_modal_llm_model = create_multi_modal_llm_model(LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4O_MINI, None)
        #multi_modal_llm_model = create_multi_modal_llm_model(LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4V, None)
    #embeddings_model = create_embed_model(EMBED_MODEL_TYPE.AZURE_ADA)

    logger.info(f"Processing file {filepath} with loader type {loader_type.name}")

    match loader_type.value:
        case DATALOADER_TYPE.PDF:
            dataloader = PdfLoader(
                kg_doc_store=kg_store, 
                llm=llm_model, 
                multi_modal_llm=multi_modal_llm_model, 
                embed_model = embeddings_model,
                verbose= True
                )
            
            loader_args = {"file_path": filepath, 
                           "filename_override" : kwargs.get("filename_override"),
                            "uri" : kwargs.get("uri") }
            logger.info(f"Processing PDF file: {loader_args.get('file_path')}")
        
        case DATALOADER_TYPE.VIDEO:
            dataloader = VideoTranscribeLoader(
                kg_doc_store=kg_store, 
                llm=llm_model, 
                embed_model = embeddings_model,
                verbose=True
                )
            
            loader_args = {"vid_filepath": kwargs.get("vid_filepath")}
            logger.info(f"Processing Video file: {loader_args.get('vid_filepath')}")

        case DATALOADER_TYPE.WEB:
            dataloader =  WebLoader(
                kg_doc_store=kg_store, 
                llm=llm_model, 
                embed_model = embeddings_model
                )
            
            loader_args = {"web_urls": kwargs.get("web_urls")}
            logger.info(f"Processing Web file: {loader_args.get('web_urls')}")

        case DATALOADER_TYPE.WIKIPEDIA:
            dataloader = WikipediaLoader(
                kg_doc_store=kg_store, 
                llm=llm_model, 
                multi_modal_llm=multi_modal_llm_model,
                embed_model = embeddings_model,
                verbose=True
                )
            
            loader_args = {"topics": kwargs.get("topics"),
                           "max_page_per_topic": kwargs.get("page_count_per_topic")}
            logger.info(f"Processing Wikipedia data for topics: {loader_args.get('topics')}")

        case DATALOADER_TYPE.CSV:
            dataloader = CsvAndTextLoader(
                kg_doc_store=kg_store, 
                llm=llm_model, 
                embed_model = embeddings_model
                )
            
            loader_args = {"csv_rootdir": kwargs.get("csv_rootdir")}
            logger.info(f"Processing CSV file: {loader_args.get('csv_rootdir')}")

        case _:
            dataloader = None

    if dataloader is not None: 
        results = dataloader.load_data_content(**loader_args)
    else: 
        logger.error(f"Error: No DataLoader found for this data type")
    
    return results
        
