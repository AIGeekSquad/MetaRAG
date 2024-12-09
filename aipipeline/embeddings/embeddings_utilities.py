# Description: This file contains the implementation embeddings generation utlities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from dataclasses import dataclass
from enum import IntEnum
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from aipipeline.config.config_param_types import AzureOpenAIConfigParam, BaseModelConfigParam

from aipipeline.utilities.constants import IngestionEnvs

@dataclass
class EMBED_MODEL_TYPE(IntEnum):
    """
    An enumeration representing different types of embedding models.
    Attributes:
        AZURE_ADA (int): Represents the Azure ADA embedding model.
        TEI (int): Represents the TEI embedding model.
        AZURE_LARGE (int): Represents the Azure Large embedding model.
    """
    AZURE_ADA = 1
    TEI = 2
    AZURE_LARGE = 3

def get_collection_name(embed_model_type:EMBED_MODEL_TYPE) -> str:
    """
    Generates a collection name based on the embedding model type.
    Args:
        embed_model_type (EMBED_MODEL_TYPE): The type of embedding model.
    Returns:
        str: The generated collection name.
    """
    collection_name = IngestionEnvs.vDB_COLLECTION_NAME
    match embed_model_type.value:
        case EMBED_MODEL_TYPE.AZURE_ADA:
           collection_name = collection_name + "_oai"
        case EMBED_MODEL_TYPE.TEI:
            collection_name = collection_name + "_tei"
    return collection_name

def create_embed_model(embed_model_type:EMBED_MODEL_TYPE, model_config:BaseModelConfigParam = None) ->BaseEmbedding : 
    """
    Creates an embedding model based on the specified type and configuration.
    Args:
        embed_model_type (EMBED_MODEL_TYPE): The type of embedding model to create.
        model_config (BaseModelConfigParam, optional): Configuration parameters for the model. Defaults to None.
    Returns:
        BaseEmbedding: An instance of the embedding model.
    Raises:
        ValueError: If the model_config is not compatible with the specified embed_model_type.
    Notes:
        - If embed_model_type is None, it defaults to EMBED_MODEL_TYPE.AZURE_LARGE.
        - If model_config is None, a default AzureOpenAIEmbedding is returned.
        - The function currently supports the following embedding model types:
            - EMBED_MODEL_TYPE.AZURE_ADA
            - EMBED_MODEL_TYPE.AZURE_LARGE
            - EMBED_MODEL_TYPE.TEI
    """
    # todo: add more embed model types and refactor this correctly
    ## TODO: Refactor this: This takes in model type to set it but if model config is set Model type should be definitely from that?? 
    ## What if model type and model config don't match? 

    embed_model : BaseEmbedding = None
    if embed_model_type is None: embed_model_type = EMBED_MODEL_TYPE.AZURE_LARGE

    if model_config is None:
        return AzureOpenAIEmbedding(
                deployment_name= IngestionEnvs.DEFAULT_EMBED_MODEL,
                model= IngestionEnvs.DEFAULT_EMBED_MODEL,
                temperature=0.0,
                azure_endpoint= IngestionEnvs.AZURE_ENDPOINT,
                api_key= IngestionEnvs.OAI_API_KEY,
                api_version= IngestionEnvs.OAI_EMBED_API_VERSION
                )
    
    match embed_model_type.value:
        case EMBED_MODEL_TYPE.AZURE_ADA:
            if isinstance(model_config, AzureOpenAIConfigParam):
                embed_model = AzureOpenAIEmbedding(
                    deployment_name= model_config.deployment_name,
                    azure_endpoint= model_config.endpoint,
                    api_key= model_config.api_key,
                    api_version=model_config.version)
            else:
                raise ValueError("AzureOpenAIConfigParam is required for AzureOpenAIEmbedding")

        case EMBED_MODEL_TYPE.AZURE_LARGE:
            if isinstance(model_config, AzureOpenAIConfigParam):
                embed_model = AzureOpenAIEmbedding(
                    deployment_name= model_config.deployment_name,
                    azure_endpoint= model_config.endpoint,
                    api_key= model_config.api_key,
                    api_version=model_config.version)
            else:
                raise ValueError("AzureOpenAIConfigParam is required for AzureOpenAIEmbedding")

        case EMBED_MODEL_TYPE.TEI:
            embed_model = TextEmbeddingsInference(
                model_name=IngestionEnvs.TEI_ENBEDDING_MODEL,
                timeout=60,  
                embed_batch_size=10)
    return embed_model
