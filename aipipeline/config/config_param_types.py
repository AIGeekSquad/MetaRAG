# Description: This file contains the configuration parameter types for the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from dataclasses import dataclass
from enum import IntEnum
from typing import List
import json

from pydantic import BaseModel, Field


##TODO: Add Multi-Modal Model
##TODO: Allow passing in of Ingestion components to replace default or environment var

class BaseModelConfigParam(BaseModel):
    '''Base Model Configuration Parameters'''
    is_multi_modal: bool= False
    llm_usage_type: int | None  = None
    temperature: float = 0.0
    token_threshold: int | None = None

class AzureOpenAIConfigParam(BaseModelConfigParam):
    """
    Configuration parameters for Azure OpenAI.
    Attributes:
        deployment_name (str | None): The name of the deployment. Default is None.
        endpoint (str | None): The endpoint URL for the Azure OpenAI service. Default is None.
        api_key (str | None): The API key for accessing the Azure OpenAI service. Default is None.
        version (str | None): The version of the Azure OpenAI service. Default is None.
    """ 
    deployment_name: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    version: str | None = None

class OpenAIConfigParam(BaseModelConfigParam):
    """
    OpenAIConfigParam is a configuration parameter class for OpenAI models.
    Attributes:
        name_of_model (str | None): The name of the OpenAI model. Default is None.
        api_key (str | None): The API key for accessing the OpenAI service. Default is None.
        version (str | None): The version of the OpenAI model. Default is None.
    """
    name_of_model: str | None = None
    api_key: str | None = None
    version: str | None= None

class CustomModelConfigParam(BaseModelConfigParam):
    """
    CustomModelConfigParam is a configuration class for custom models.

    Attributes:
        name_of_model (str | None): The name of the model. Default is None.
        endpoint (str | None): The endpoint URL for the model. Default is None.
        path (str | None): The file path to the model. Default is None.
        host (str | None): The host address for the model. Default is None.
        api_key (str | None): The API key for accessing the model. Default is None.
    """
    name_of_model: str | None= None
    endpoint: str | None = None
    path: str | None= None
    host: str | None= None
    api_key: str | None= None 

class LLMConfigParam(BaseModel):
    """
    LLMConfigParam is a configuration class for Language Model parameters.

    Attributes:
        token_threshold (int | None): The threshold for tokens. If None, no threshold is applied.
        model (str | None): The name of the model to be used. If None, no specific model is set.
        llm_model_configuration (AzureOpenAIConfigParam | OpenAIConfigParam | CustomModelConfigParam | None): 
            The configuration for the language model. It can be an instance of AzureOpenAIConfigParam, 
            OpenAIConfigParam, CustomModelConfigParam, or None if no specific configuration is provided.
    """
    token_threshold: int | None = None
    model: str | None = None
    llm_model_configuration: AzureOpenAIConfigParam | OpenAIConfigParam |CustomModelConfigParam| None = None
 
class IngestionConfigurationParm(BaseModel):
    """
    IngestionConfigurationParm is a configuration class for ingestion parameters.
    Attributes:
        use_vector_search (bool): Flag to enable or disable vector search. Default is True.
        use_graph_search (bool): Flag to enable or disable graph search. Default is True.
        use_ontology_search (bool): Flag to enable or disable ontology search. Default is False.
        use_tei_embed (bool): Flag to enable or disable TEI embedding. Default is False.
        semantic_chucking_threshold (float): Threshold for semantic chunking. Default is 0.9.
        LLMModelConfig (LLMConfigParam): Configuration for the LLM model. Default is None.
        EmbedModelConfig (LLMConfigParam): Configuration for the embedding model. Default is None.
        MultiModalModelConfig (LLMConfigParam): Configuration for the multi-modal model. Default is None.
    Methods:
        toJson(): Converts the configuration object to a JSON string.
    """
    use_vector_search: bool = True
    use_graph_search: bool = True
    use_ontology_search: bool = False
    use_tei_embed: bool = False
    semantic_chucking_threshold: float = 0.9
    LLMModelConfig: LLMConfigParam = None
    EmbedModelConfig: LLMConfigParam = None
    MultiModalModelConfig: LLMConfigParam = None
    
    def toJson(self):
        return json.dumps(self, indent=4, default=lambda o: o.__dict__)   
    
class RetrieverConfigParam(BaseModel):  
    """
    RetrieverConfigParam is a configuration class for retriever settings.

    Attributes:
        type (str | None): The type of retriever. Default is None.
        llm (LLMConfigParam | None): Configuration parameters for the language model. Default is None.
        embed_model (LLMConfigParam | None): Configuration parameters for the embedding model. Default is None.
        top_k (int): The number of top results to retrieve. Default is 5.
        query_iteration_max (int): The maximum number of query iterations. Default is 4.
    """
    type: str | None = None
    llm: LLMConfigParam | None = None
    embed_model: LLMConfigParam | None = None
    top_k: int = 5
    query_iteration_max: int  = 4

 
class QueryEngineConfigParam(BaseModel):
    """
    QueryEngineConfigParam is a configuration class for the query engine.

    Attributes:
        type (str | None): The type of the query engine. Default is None.
        llm (LLMConfigParam | None): Configuration parameters for the LLM (Language Model). Default is None.
        processor (List[str] | None): List of processors to be used. Default is None.
        iteration_max (int): Maximum number of iterations. Default is 5.
        use_reranker (bool): Flag to indicate whether to use a reranker. Default is True.
    """
    type: str| None  = None
    llm: LLMConfigParam | None = None
    processor: List[str] | None = None
    iteration_max: int = 5
    use_reranker: bool = True

 
class ConfigurationParam(BaseModel): 
    """
    ConfigurationParam is a data model class that holds various configuration parameters for the application.

    Attributes:
        llm (LLMConfigParam | None): Configuration for the language model.
        embed_model (LLMConfigParam | None): Configuration for the embedding model.
        retriever (RetrieverConfigParam | None): Configuration for the retriever.
        query_engine (QueryEngineConfigParam | None): Configuration for the query engine.
        domain_ontology (str | None): Domain ontology information.
        vector_database (str | None): Name of the vector database.
        vector_database_collection_name (str | None): Name of the collection in the vector database.

    Methods:
        toJson(): Serializes the ConfigurationParam instance to a JSON string.
    """
    llm: LLMConfigParam| None = None
    embed_model: LLMConfigParam| None = None
    retriever: RetrieverConfigParam| None = None
    query_engine: QueryEngineConfigParam| None = None
    domain_ontology: str| None = None
    vector_database: str| None = None
    vector_database_collection_name:str | None = None

    ## Make class serializable to json
    def toJson(self):
        return json.dumps(self, indent=4, default=lambda o: o.__dict__)    
    
@dataclass
class CONFIG_TYPE(IntEnum):
    """
    CONFIG_TYPE is an enumeration that defines different types of configuration options.

    Attributes:
        AZURE_OAI (int): Represents the Azure OpenAI configuration type.
        OPENAI (int): Represents the OpenAI configuration type.
        CUSTOM (int): Represents a custom configuration type.
    """ 
    AZURE_OAI = 1
    OPENAI = 2
    CUSTOM = 3