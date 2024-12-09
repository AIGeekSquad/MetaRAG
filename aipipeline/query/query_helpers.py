# Description: This file contains the implementation of Query Server helpers of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from dataclasses import dataclass
from typing import List
import re
from llama_index.core.llms.llm import LLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine, FLAREInstructQueryEngine, BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, BaseModelConfigParam, ConfigurationParam, CustomModelConfigParam, LLMConfigParam, OpenAIConfigParam, QueryEngineConfigParam, RetrieverConfigParam

from aipipeline.embeddings.embeddings_utilities import EMBED_MODEL_TYPE, create_embed_model
from aipipeline.utilities.llm_utilities import create_llm_model, create_multi_modal_llm_model
from aipipeline.utilities.constants import LLM_MODEL_TYPE, LLM_MULTI_MODAL_MODEL_TYPE, IngestionEnvs
from aipipeline.retriever.vector_and_graphRetriever import VectorAndGraphRetriever
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

import logging

logger = logging.getLogger(__name__)

#TODO: Add KG option

@dataclass(frozen=False)
class QuerySvcEnvs: 
    """
    A class to manage and configure the environment settings for the Query Service.
    Attributes:
    -----------
    LLM_MODEL : LLM | MultiModalLLM
        The language model to be used.
    EMBED_MODEL : BaseEmbedding
        The embedding model to be used.
    CONFIGURATION : ConfigurationParam
        The configuration parameters for the Query Service.
    Methods:
    --------
    set_querysvc_config(configuration: ConfigurationParam, model_configuration: BaseModelConfigParam = None) -> ConfigurationParam
        Sets the configuration for the Query Service environment.
        Parameters:
        -----------
        configuration : ConfigurationParam
            The configuration parameters to be set.
        model_configuration : BaseModelConfigParam, optional
            The model configuration parameters to be set. If not provided, default values will be used.
        Returns:
        --------
        ConfigurationParam
            The updated configuration parameters.
    """

    LLM_MODEL: LLM| MultiModalLLM = None
    EMBED_MODEL: BaseEmbedding = None
    CONFIGURATION: ConfigurationParam = None

    @staticmethod
    def set_querysvc_config(configuration: ConfigurationParam, model_configuration: BaseModelConfigParam = None):
        """
        Sets the configuration for the Query Service.
        This function initializes and sets the configuration parameters for the Query Service, including
        the language model (LLM), embedding model, retriever, and query engine configurations. If any of 
        the configurations are not provided, default values are used.
        Args:
            configuration (ConfigurationParam): The main configuration parameters for the Query Service.
            model_configuration (BaseModelConfigParam, optional): The model-specific configuration parameters.
                If not provided, default values are used.
        Returns:
            ConfigurationParam: The updated configuration for the Query Service.
        """
        default_llm_name: str = IngestionEnvs.DEFAULT_LLM_MODEL
        default_embed_model_name: str = IngestionEnvs.DEFAULT_EMBED_MODEL
        default_collection_name = IngestionEnvs.vDB_COLLECTION_NAME

        if configuration is not None:
            QuerySvcEnvs.CONFIGURATION = configuration
        else:
            QuerySvcEnvs.CONFIGURATION = ConfigurationParam()
            configuration = QuerySvcEnvs.CONFIGURATION

        model_configuration_default = model_configuration if model_configuration is not None else AzureOpenAIConfigParam(
                temperature=0.0,
                endpoint= IngestionEnvs.AZURE_ENDPOINT,
                api_key= IngestionEnvs.OAI_API_KEY,
                api_version=IngestionEnvs.OAI_API_VERSION
                )
        

        # setup defualt values for llm and embed model configuration
        llm_config: LLMConfigParam =  configuration.llm if configuration.llm is not None else LLMConfigParam()
        llm_config.model = llm_config.model if llm_config.model is not None else default_llm_name
        llm_config.token_threshold = llm_config.token_threshold if llm_config.token_threshold is not None else None

        llm_config.llm_model_configuration = _set_model_configuration_values(llm_config.model, llm_config.llm_model_configuration, model_configuration_default)

        embed_model_config: LLMConfigParam = configuration.embed_model if configuration.embed_model is not None else LLMConfigParam()
        embed_model_config.model = embed_model_config.model if embed_model_config.model is not None else default_embed_model_name
        embed_model_config.token_threshold = embed_model_config.token_threshold if embed_model_config.token_threshold is not None else None

        embed_model_config.llm_model_configuration = _set_model_configuration_values(embed_model_config.model, embed_model_config.llm_model_configuration, model_configuration_default)

        # setup default values for retriever
        retriever_config: RetrieverConfigParam = configuration.retriever if configuration.retriever is not None else RetrieverConfigParam()
        retriever_config.llm = retriever_config.llm if retriever_config.llm is not None else llm_config
        retriever_config.embed_model = retriever_config.embed_model if retriever_config.embed_model is not None else embed_model_config
        retriever_config.type = retriever_config.type if retriever_config.type is not None else "VectorAndGraph" ## Need to set/tell strings to send in
    
        # setup default values for query engine
        query_config: QueryEngineConfigParam = configuration.query_engine if configuration.query_engine is not None else  QueryEngineConfigParam()
        query_config.type = query_config.type if query_config.type is not None else "FLARE" ## Need to set string to send in
        query_config.processor  = query_config.processor if query_config.processor is not None else None
        query_config.llm = query_config.llm if query_config.llm is not None else llm_config

        # set the configuration values
        QuerySvcEnvs.CONFIGURATION.vector_database = configuration.vector_database if configuration.vector_database is not None else  "QDRANT"
        QuerySvcEnvs.CONFIGURATION.domain_ontology =  configuration.domain_ontology if configuration.domain_ontology is not None else  None
        QuerySvcEnvs.CONFIGURATION.embed_model =  configuration.embed_model if configuration.embed_model is not None else  embed_model_config
        QuerySvcEnvs.CONFIGURATION.vector_database_collection_name =  configuration.vector_database_collection_name if configuration.vector_database_collection_name is not None else  default_collection_name
                 
        QuerySvcEnvs.CONFIGURATION.query_engine = query_config
        QuerySvcEnvs.CONFIGURATION.retriever = retriever_config
        QuerySvcEnvs.CONFIGURATION.llm = llm_config
        QuerySvcEnvs.CONFIGURATION.embed_model = embed_model_config

        QuerySvcEnvs.LLM_MODEL = get_llm_model(QuerySvcEnvs.CONFIGURATION.llm)
     
        QuerySvcEnvs.EMBED_MODEL = get_embed_model(QuerySvcEnvs.CONFIGURATION.embed_model)
        
        return QuerySvcEnvs.CONFIGURATION
    
def _set_model_configuration_values(model_name:str, config: BaseModelConfigParam, default_model_configuration: BaseModelConfigParam = None):
        if config is None :
            if default_model_configuration is  None:
                config = AzureOpenAIConfigParam(
                deployment_name= model_name,
                temperature=0.0,
                endpoint= IngestionEnvs.AZURE_ENDPOINT,
                api_key= IngestionEnvs.OAI_API_KEY,
                api_version=IngestionEnvs.OAI_API_VERSION
                )
            else:
                default_model_configuration_clone = default_model_configuration.model_copy(deep=True)
                if (model_name is not None):
                                if isinstance(default_model_configuration_clone, AzureOpenAIConfigParam):
                                    default_model_configuration_clone.deployment_name = model_name
                                    default_model_configuration_clone.version = default_model_configuration_clone.version if default_model_configuration_clone.version is not None else IngestionEnvs.OAI_API_VERSION
                                elif isinstance(default_model_configuration_clone, OpenAIConfigParam):
                                    default_model_configuration_clone.name_of_model = model_name
                                elif isinstance(default_model_configuration_clone, CustomModelConfigParam):
                                    default_model_configuration_clone.name_of_model = model_name
                return default_model_configuration_clone
        else:
            configClone = config.model_copy(deep=True)
            
            if isinstance(config, AzureOpenAIConfigParam):
                _set_azure_openai_model_configuration_values(model_name,configClone, default_model_configuration)
            elif isinstance(config, OpenAIConfigParam):
                _set_openai_model_configuration_values(model_name, configClone, default_model_configuration)
            elif isinstance(config, CustomModelConfigParam):
                _set_custom_model_configuration_values(model_name, configClone, default_model_configuration)
            
            return configClone
        
def _set_base_model_configuration_values(model_name:str, config: BaseModelConfigParam, default_model_configuration: BaseModelConfigParam = None):
    if default_model_configuration is not None:
        if (config.llm_usage_type is None):
            config.llm_usage_type = default_model_configuration.llm_usage_type
        if (config.token_threshold is None):
            config.token_threshold = default_model_configuration.token_threshold

def _set_azure_openai_model_configuration_values(model_name:str, config: AzureOpenAIConfigParam, default_model_configuration: BaseModelConfigParam = None):
    _set_base_model_configuration_values(config, default_model_configuration)
    if default_model_configuration is not None and isinstance(default_model_configuration, AzureOpenAIConfigParam):
        if (config.deployment_name is None):
            config.deployment_name = model_name if model_name is not None else  default_model_configuration.deployment_name
        if (config.endpoint is None):
            config.endpoint = default_model_configuration.endpoint
        if (config.api_key is None):
            config.api_key = default_model_configuration.api_key
        if (config.version is None):
            config.version = default_model_configuration.version

def _set_openai_model_configuration_values(model_name:str, config: OpenAIConfigParam, default_model_configuration: BaseModelConfigParam = None):
    _set_base_model_configuration_values(config, default_model_configuration)
    if default_model_configuration is not None and isinstance(default_model_configuration, OpenAIConfigParam):
        if (config.name_of_model is None):
            config.name_of_model = model_name if model_name is not None else default_model_configuration.name_of_model
        if (config.api_key is None):
            config.api_key = default_model_configuration.api_key
        if (config.version is None):
            config.version = default_model_configuration.version

def _set_custom_model_configuration_values(model_name:str, config: CustomModelConfigParam, default_model_configuration: BaseModelConfigParam = None):
    _set_base_model_configuration_values(config, default_model_configuration)
    if default_model_configuration is not None and isinstance(default_model_configuration, CustomModelConfigParam):
        if (config.name_of_model is None):
            config.name_of_model = model_name if model_name is not None else  default_model_configuration.name_of_model
        if (config.endpoint is None):
            config.endpoint = default_model_configuration.endpoint
        if (config.api_key is None):
            config.api_key = default_model_configuration.api_key
        if (config.path is None):
            config.path = default_model_configuration.path
        if (config.host is None):
            config.host = default_model_configuration.host

def _get_llm_model_name_from_config(config: BaseModelConfigParam, default:str = None) -> str:
    if isinstance(config, AzureOpenAIConfigParam):
                if config.deployment_name is  None:
                    config.deployment_name = default
                return config.deployment_name
    elif isinstance(config, OpenAIConfigParam):
                if config.name_of_model is  None:
                    config.name_of_model = default
                return config.name_of_model
    elif isinstance(config, CustomModelConfigParam):
                if config.name_of_model is  None:
                    config.name_of_model = default
                return config.name_of_model
    return None

def get_llm_model(llm_config: LLMConfigParam) -> LLM|MultiModalLLM:
    """
    Retrieves the appropriate LLM (Large Language Model) or MultiModalLLM based on the provided configuration.
    Args:
        llm_config (LLMConfigParam): The configuration parameters for the LLM model.
    Returns:
        LLM | MultiModalLLM: The instantiated LLM or MultiModalLLM based on the configuration.
    Raises:
        ValueError: If the llm_config is None or if the model configuration is not set.
    Notes:
        - If the configuration specifies a multi-modal model, a MultiModalLLM is created.
        - If the model name matches certain patterns (e.g., GPT-4, GPT-3.5 Turbo, Mistral-7B), the corresponding LLM is created.
    """
    if llm_config is None:
        raise ValueError("LLM model configuration is not set")
        
    llm_model = _get_llm_model_name_from_config(llm_config.llm_model_configuration, llm_config.model)
    gpt4_pattern = re.compile(r'gpt-?4(?:-?32k|-?turbo)?',re.IGNORECASE)
    model_config = llm_config.llm_model_configuration
    llm : LLM = None
    if llm_config.llm_model_configuration.is_multi_modal:
        # todo : find better ways to create orther ones
        llm = create_multi_modal_llm_model(LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4V, model_config) 
    else:
        if gpt4_pattern.search(llm_model):
            llm = create_llm_model(LLM_MODEL_TYPE.AZURE_GPT4, model_config)
        elif llm_model == "gpt-35-turbo":
            llm = create_llm_model(LLM_MODEL_TYPE.AZURE_GPT35, model_config)
        elif llm_model == "mistral-7b":
            llm = create_llm_model(LLM_MODEL_TYPE.MISTRAL_7B, model_config)

    return llm

def get_embed_model(embed_model_config: LLMConfigParam) -> BaseEmbedding:
    """
    Retrieves the embedding model based on the provided configuration.
    Args:
        embed_model_config (LLMConfigParam): Configuration parameters for the embedding model.
    Returns:
        BaseEmbedding: An instance of the embedding model based on the configuration.
    Raises:
        ValueError: If the model configuration is invalid or unsupported.
    """
    embed_model = _get_llm_model_name_from_config(embed_model_config.llm_model_configuration, embed_model_config.model)
    ada_pattern = re.compile(r'ada',re.IGNORECASE)
    embed_model_config = embed_model_config.llm_model_configuration
    embed: LLM = None
    if ada_pattern.search(embed_model):
        embed = create_embed_model(EMBED_MODEL_TYPE.AZURE_ADA,embed_model_config)
    elif embed_model == "TEI":
        embed = create_embed_model(EMBED_MODEL_TYPE.TEI,embed_model_config)
    else: 
        embed = create_embed_model(EMBED_MODEL_TYPE.AZURE_LARGE,embed_model_config)
    return embed


def get_retriever(documentGraph: DocumentGraphBaseComponent, vectorStore: VectorDBBaseComponent, vectorCollection: str)-> VectorAndGraphRetriever: 
    """
    Creates and returns a QueryFusionRetriever instance configured with a VectorAndGraphRetriever.
    Args:
        documentGraph (DocumentGraphBaseComponent): The document graph component.
        vectorStore (VectorDBBaseComponent): The vector store component.
        vectorCollection (str): The name of the vector collection.
    Returns:
        VectorAndGraphRetriever: A configured QueryFusionRetriever instance.
    Raises:
        ValueError: If the LLM model or Embed model is not set in QuerySvcEnvs.
    """

    if QuerySvcEnvs.LLM_MODEL is None:
        raise ValueError("LLM model is not set")
    
    if QuerySvcEnvs.EMBED_MODEL is None:
        raise ValueError("Embed model is not set")

    logger.info("Creating VectorAndGraphRetriever")
    inner_retriever = VectorAndGraphRetriever(documentGraph, 
                                        vectorStore,
                                        vectorCollection,
                                        QuerySvcEnvs.CONFIGURATION.retriever.top_k,
                                        QuerySvcEnvs.EMBED_MODEL,
                                        verbose=True
                                        )  
    
    logger.info("Creating QueryFusionRetriever")
    fusion_retriever = QueryFusionRetriever(
        [inner_retriever],
        llm = QuerySvcEnvs.LLM_MODEL,
        similarity_top_k=QuerySvcEnvs.CONFIGURATION.retriever.top_k,
        num_queries=QuerySvcEnvs.CONFIGURATION.retriever.query_iteration_max,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        # query_gen_prompt="...",  # we could override the query generation prompt here
        verbose = True,
    )

    return fusion_retriever  

def get_query_engine(retriever: BaseRetriever) -> BaseQueryEngine:
    """
    Creates and returns a query engine based on the provided retriever and configuration settings.
    Args:
        retriever (BaseRetriever): The retriever instance to be used by the query engine.
    Returns:
        BaseQueryEngine: An instance of a query engine configured according to the environment settings.
    Raises:
        ValueError: If the LLM model is not set in the environment settings.
    Notes:
        - If the `use_reranker` configuration is enabled, a RankGPTRerank postprocessor is added to the query engine.
        - The type of query engine created depends on the `query_engine.type` configuration:
            - "BASIC": Returns a basic query engine.
            - "FLARE": Returns a FLAREInstructQueryEngine with additional configurations.
    """
    if QuerySvcEnvs.LLM_MODEL is None:
        raise ValueError("LLM model is not set")

    logger.info("Creating QueryEngine")
    node_postprocessors: List[BaseNodePostprocessor] = []
    if QuerySvcEnvs.CONFIGURATION.query_engine.use_reranker:
        logger.info("Creating RankGPTRerank")
        node_postprocessors.append(RankGPTRerank(llm=QuerySvcEnvs.LLM_MODEL, top_n=QuerySvcEnvs.CONFIGURATION.retriever.top_k, verbose=False))
    
    query_engine = RetrieverQueryEngine.from_args(retriever, response_mode= ResponseMode.REFINE, llm=QuerySvcEnvs.LLM_MODEL, node_postprocessors=node_postprocessors)

    if (QuerySvcEnvs.CONFIGURATION.query_engine.type == "BASIC"):
        logger.info("Creating BaseQueryEngine")
        return query_engine
    
    if (QuerySvcEnvs.CONFIGURATION.query_engine.type == "FLARE"):
        logger.info("Creating FLAREInstructQueryEngine")
        query_engine = FLAREInstructQueryEngine(query_engine=query_engine, llm=QuerySvcEnvs.LLM_MODEL, max_iterations=QuerySvcEnvs.CONFIGURATION.query_engine.iteration_max, verbose=True)
        
    return query_engine
    










