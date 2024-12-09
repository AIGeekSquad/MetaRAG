# Description: This file contains the implementation of LLM utilities of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.core.llms.llm import LLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.llms.llama_cpp import LlamaCPP

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, BaseModelConfigParam, CustomModelConfigParam, OpenAIConfigParam
from aipipeline.utilities.constants import IngestionEnvs, LLM_MODEL_TYPE, LLM_MULTI_MODAL_MODEL_TYPE

import logging

logger = logging.getLogger(__name__)

def create_llm_model(llm_model_type:LLM_MODEL_TYPE, llm_model_config: BaseModelConfigParam = None) ->LLM : 
    """
    Creates an instance of a language model (LLM) based on the provided model type and configuration.
    Args:
        llm_model_type (LLM_MODEL_TYPE): The type of the language model to be created.
        llm_model_config (BaseModelConfigParam, optional): Configuration parameters for the language model. 
            If None, a default AzureOpenAI model will be created using environment variables.
    Returns:
        LLM: An instance of the specified language model.
    If `llm_model_config` is None, a default AzureOpenAI model is created using environment variables:
        - IngestionEnvs.DEFAULT_LLM_MODEL
        - IngestionEnvs.AZURE_ENDPOINT
        - IngestionEnvs.OAI_API_KEY
        - IngestionEnvs.OAI_API_VERSION
    If `llm_model_config` is an instance of `AzureOpenAIConfigParam`, a custom AzureOpenAI model is created.
    If `llm_model_config` is an instance of `CustomModelConfigParam`, a custom model is created.
    If `llm_model_config` is an instance of `OpenAIConfigParam`, an OpenAI model is created.
    """    
    llm : LLM = None 

    if llm_model_config is None:
        return AzureOpenAI(
                deployment_name= IngestionEnvs.DEFAULT_LLM_MODEL,
                model= IngestionEnvs.DEFAULT_LLM_MODEL,
                temperature=0.0,
                azure_endpoint= IngestionEnvs.AZURE_ENDPOINT,
                api_key= IngestionEnvs.OAI_API_KEY,
                api_version=IngestionEnvs.OAI_API_VERSION
                )
    
    if isinstance(llm_model_config, AzureOpenAIConfigParam):
        return create_azure_openai_model(model_config= llm_model_config, llm_model_type= llm_model_type)
    elif isinstance(llm_model_config, CustomModelConfigParam):
        return create_custom_model(model_config= llm_model_config, llm_model_type= llm_model_type)
    elif isinstance(llm_model_config, OpenAIConfigParam):
        return create_openai_model(model_config= llm_model_config, llm_model_type= llm_model_type)

    return

def create_azure_openai_model(llm_model_type:LLM_MODEL_TYPE, model_config: AzureOpenAIConfigParam ) -> LLM:
    """
    Creates an Azure OpenAI model based on the specified model type and configuration parameters.
    Args:
        llm_model_type (LLM_MODEL_TYPE): The type of the LLM model to create. 
                                            It should be one of the enumerations in LLM_MODEL_TYPE.
        model_config (AzureOpenAIConfigParam): Configuration parameters required to create the Azure OpenAI model.
                                                This includes deployment name, temperature, endpoint, API key, and API version.
    Returns:
        LLM: An instance of the Azure OpenAI model configured with the provided parameters.
    """

    llm: LLM = None

    match llm_model_type.value:
        case LLM_MODEL_TYPE.AZURE_GPT35:
            logger.info("Creating Azure GPT-3.5 model")
            llm = AzureOpenAI(
                deployment_name= model_config.deployment_name,
                temperature=model_config.temperature,
                azure_endpoint= model_config.endpoint,
                api_key= model_config.api_key,
                api_version=model_config.version if model_config.version is not None else IngestionEnvs.OAI_API_VERSION
                )
            
        case LLM_MODEL_TYPE.AZURE_GPT4:
            logger.info("Creating Azure GPT-4 model")
            llm = AzureOpenAI(
                deployment_name= model_config.deployment_name,
                temperature= model_config.temperature,
                azure_endpoint= model_config.endpoint,
                api_key= model_config.api_key,
                api_version=model_config.version if model_config.version is not None else IngestionEnvs.OAI_API_VERSION)
            
    return llm

def create_custom_model(llm_model_type:LLM_MODEL_TYPE, custom_model_config: CustomModelConfigParam) -> LLM:
    """
    Creates a custom language model based on the specified type and configuration.
    Args:
        llm_model_type (LLM_MODEL_TYPE): The type of the language model to create.
        custom_model_config (CustomModelConfigParam): Configuration parameters for the custom model.
    Returns:
        LLM: An instance of the specified language model.    
    """
    llm: LLM = None
     
    match llm_model_type.value:
            
        case LLM_MODEL_TYPE.MISTRAL_7B:
            logger.info("Creating Mistral 7B model")
            llm = LlamaCPP(
                model_url= custom_model_config.endpoint if custom_model_config is not None 
                else IngestionEnvs.MISTRAL7B_ENDPOINT,
                model_path= custom_model_config.path if custom_model_config is not None 
                else None,
                temperature=custom_model_config.temperature if custom_model_config is not None else 0.0,
            )
    return llm

def create_openai_model(llm_model_type: LLM_MODEL_TYPE, oai_model_config: OpenAIConfigParam) -> LLM:
    """
    Creates an instance of an OpenAI model based on the provided configuration.
    Args:
        llm_model_type (LLM_MODEL_TYPE): The type of the language model.
        oai_model_config (OpenAIConfigParam): Configuration parameters for the OpenAI model.
    Returns:
        LLM: An instance of the OpenAI model.
    """
    llm: LLM = None

    llm = OpenAI(
            model= oai_model_config.name_of_model,
            temperature=oai_model_config.temperature, 
            api_key= oai_model_config.api_key,
            api_version= oai_model_config.version,
            max_tokens=oai_model_config.token_threshold
            )
    
    return llm 

def create_multi_modal_llm_model(llm_model_type:LLM_MULTI_MODAL_MODEL_TYPE, llm_model_config: AzureOpenAIConfigParam) -> MultiModalLLM :
    """
    Creates a multi-modal LLM model based on the specified type and configuration.
    Args:
        llm_model_type (LLM_MULTI_MODAL_MODEL_TYPE): The type of the multi-modal LLM model to create.
        llm_model_config (AzureOpenAIConfigParam): The configuration parameters for the Azure OpenAI model.
    Returns:
        MultiModalLLM: An instance of the multi-modal LLM model, or None if the configuration is invalid.
    """
    
    llm : MultiModalLLM = None

    if llm_model_config is None:
        ## We need to test to see if IngestionEnvs.OAI_GPT4V_DEPLOY_NAME exists or pass None
        # if IngestionEnvs.OAI_GPT4V_DEPLOY_NAME.strip() == "":
        #     return None
        if IngestionEnvs.OAI_GPT4O_DEPLOY_NAME.strip() == "":
            return None
        else:
            logger.info("Creating Azure GPT-4V model")
            return AzureOpenAIMultiModal(
                    deployment_name= IngestionEnvs.OAI_GPT4O_DEPLOY_NAME,
                    model= IngestionEnvs.OAI_GPT4O_MODEL,
                    temperature=0.0,
                    azure_endpoint= IngestionEnvs.AZURE_ENDPOINT,
                    api_key= IngestionEnvs.OAI_API_KEY,
                    api_version=IngestionEnvs.OAI_API_VERSION
            )
    
    match llm_model_type.value:
        case LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4V:
            logger.info("Creating Azure GPT-4V model")
            llm = AzureOpenAIMultiModal(
                deployment_name= llm_model_config.deployment_name,
                model= "gpt-4-vision",
                temperature= llm_model_config.temperature,
                azure_endpoint= llm_model_config.endpoint,
                api_key= llm_model_config.api_key,
                api_version=llm_model_config.version if llm_model_config.version is not None else IngestionEnvs.OAI_API_VERSION
            )
        case LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4O:
            logger.info("Creating Azure GPT-4o model")
            llm = AzureOpenAIMultiModal(
                deployment_name= llm_model_config.deployment_name,
                model= "gpt-4o",
                temperature= llm_model_config.temperature,
                azure_endpoint= llm_model_config.endpoint,
                api_key= llm_model_config.api_key,
                api_version=llm_model_config.version if llm_model_config.version is not None else IngestionEnvs.OAI_API_VERSION
            )
        case LLM_MULTI_MODAL_MODEL_TYPE.AZURE_GPT4O_MINI:
            logger.info("Creating Azure GPT-4o-mini model")
            llm = AzureOpenAIMultiModal(
                deployment_name= llm_model_config.deployment_name,
                model= "gpt-4o-mini",
                temperature= llm_model_config.temperature,
                azure_endpoint= llm_model_config.endpoint,
                api_key= llm_model_config.api_key,
                api_version=llm_model_config.version if llm_model_config.version is not None else IngestionEnvs.OAI_API_VERSION
            )
        case LLM_MULTI_MODAL_MODEL_TYPE.LLAVA: 
            pass
        case LLM_MULTI_MODAL_MODEL_TYPE.HF_QWEN_VL:
            pass

    return llm