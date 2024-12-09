# Description: This file contains the API for setting the configuration for the ingestion and query components of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.config.config_param_types import ConfigurationParam, QueryEngineConfigParam, LLMConfigParam, RetrieverConfigParam
from aipipeline.query.query_helpers import QuerySvcEnvs, get_llm_model
from aipipeline.utilities.constants import IngestionEnvs

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# Config -- fast api contract objects
class SetQueryConfig(BaseModel):
    """
    SetQueryConfig is a configuration model for setting up various parameters related to the query and retrieval process in an ingestion pipeline.
    Attributes:
        llm_model (str | None): LLM Model to use. Select custom default LLM used for retriever & query engine, if not specified for each separately. Examples: ["gpt-35-turbo"].
        top_k (int | None): Top k number of results desired for retriever, query.
        llm_temperature (float | None): Model temperature value. Value for how precise or creative responses should be.
        retriever_type (str | None): Type of Retriever. Optional: Retriever to use in Ingestion Pipeline; Default used if not set.
        retriever_llm_model (str | None): LLM Model to use for Retriever. Optional: Default LLM is used for retriever if this value not specified.
        retriever_query_iteration (int | None): Number of Query iterations of retriever. Optional: Default number of iterations for retriever if this value not specified.
        query_llm_model (str | None): LLM Model to use for Query. Optional: Default LLM is used for query if this value not specified.
        query_type (str | None): Type of Query Engine (e.g., FLARE, KGRAPH, MULTIMODAL). Optional: Query engine to use in Ingestion Pipeline; Default used if not set.
        query_processor (List[str] | None): Type of Query Processor to use. Optional: How you want to process the data & response (e.g., Reranker).
        vectordb (str | None): Desired vector database to use. Optional: Desired vector DB, if not specified default vector database will be used.
        vectordb_top_k (int | None): Desired vector search top_k to use. Optional: Desired vector DB top k, if not specified default vector database will be used.
        max_tokens (int | None): Max number of Tokens to generate. Optional: Total number of tokens to use, this number varies by model.
        embed_model (str | None): Name of Embedding Model to use, ADA or TEI. Embedding model: OpenAI Ada or HF TEI.
        ontology (str | None): Link to domain ontology to use.
    """

    llm_model: str | None = Field(default=None, title="LLM Model to use", examples=["gpt-35-turbo"], description="Select custom default LLM used for retriever & query engine, if not specified for each separately") 
    top_k: int | None =  Field(default=None, title="Top k # of results desired for retriever, query")
    llm_temperature: float | None = Field(default=None, title="Model temperature value", description="Value for how precise or creative responses should be")
    retriever_type: str | None = Field(default=None, title="Type of Retriever", description="Optional: Retriever to use in Ingestion Pipeline; Default used if not set")
    retriever_llm_model: str | None = Field(default=None, title="LLM Model to use for Retriever", description="Optional: Default LLM is used for retriever if this value not specified")
    retriever_query_iteration: int | None = Field(default=None, title="Number of Query iterations of retriever", description="Optional: Default number of iterations for retriever if this value not specified")
    query_llm_model: str | None = Field(default=None, title="LLM Model to use for Query", description="Optional: Default LLM is used for query if this value not specified")
    query_type: str | None = Field(default=None, title="Type of Query Engine ex. FLARE, KGRAPH, MULTIMODAL", description="Optional: Query engine to use in Ingestion Pipeline; Default used if not set")
    query_processor: List[str] | None = Field(default=None, title="Type of Query Processor to use", description="Optional: How you want to process the data & response ex. Reranker")
    vectordb: str | None = Field(default=None, title="Desired vector database to use", description="Optional: Desired vector DB, if not specified default vector database will be used")
    vectordb_top_k: int| None = Field(default=None, title="Desired vector search top_k to use", description="Optional: Desired vector DB top k, if not specified default vector database will be used")
    max_tokens: int | None = Field(default=None, title="Max number of Tokens to generate", description="Optional: Total number of tokens to use, this number varies by model")
    embed_model: str | None = Field(default=None, title="Name of Embedding Model to use, ADA or TEI", description="Embedding model: OpenAI Ada or HF TEI")
    ontology: str | None = Field(default=None, title="Link to domain ontology to use") # Determine how we want to do ontology nesting

app = FastAPI()

#TODO: Update Set Config API based upon changes in QuerySvcEnvs
## TODO: Add Multi-Modal Model
@app.post("/setconfig", description="Set configuration for ingestion and query")
def set_query_config(config: SetQueryConfig):
    """
    Sets the query configuration using the provided config values. If any value is missing in the config,
    it uses the default value from environment variables.
    Args:
        config (SetQueryConfig): The configuration object containing the query settings.
    Attributes:
        default_llm_name (str): The name of the default LLM model.
        default_top_k (int): The default value for the top_k parameter.
        default_llm: The default LLM model instance.
        llm_config (LLMConfigParam): Configuration parameters for the LLM model.
        retriever_config (RetrieverConfigParam): Configuration parameters for the retriever.
        query_config (QueryEngineConfigParam): Configuration parameters for the query engine.
        configuration (ConfigurationParam): The final configuration object to be set.
    Configuration Parameters:
        llm_model (str): The LLM model name.
        max_tokens (int): The maximum number of tokens.
        retriever_llm_model (str): The LLM model name for the retriever.
        retriever_query_iteration (int): The maximum number of query iterations for the retriever.
        top_k (int): The top_k value for the retriever.
        query_type (str): The type of query engine.
        query_processor (str): The query processor.
        query_llm_model (str): The LLM model name for the query engine.
        vectordb (str): The vector database.
        ontology (str): The domain ontology.
    Returns:
        None
    """

    # use configuration values, if a missing value use the default value from environment variables

    default_llm_name: str = config.llm_model if config.llm_model else IngestionEnvs.DEFAULT_LLM_MODEL
    default_top_k: int = 5
    default_llm = get_llm_model(default_llm_name)

    llm_config: LLMConfigParam = LLMConfigParam()
    llm_config.model = default_llm_name
    llm_config.token_threshold = config.max_tokens if config.max_tokens else None 

    retriever_config: RetrieverConfigParam = RetrieverConfigParam()
    retriever_config.llm = config.retriever_llm_model if config.retriever_llm_model else default_llm_name
    retriever_config.query_iteration_max = config.retriever_query_iteration if config.retriever_query_iteration else default_top_k
    retriever_config.top_k = config.top_k if config.top_k else default_top_k
    retriever_config.type = retriever_config.type if retriever_config.type else "VectorAndGraph" ## Need to change from strings to send int

    query_config: QueryEngineConfigParam = QueryEngineConfigParam()
    query_config.type = config.query_type if config.query_type else "FLARE" ## Need to change from string to send int 
    query_config.processor = config.query_processor if config.query_processor else None
    query_config.llm = config.query_llm_model if config.query_llm_model else default_llm_name
    
    configuration: ConfigurationParam = ConfigurationParam()
    configuration.query_engine = query_config
    configuration.retriever = retriever_config
    configuration.llm = default_llm_name
    configuration.vector_database = config.vectordb if config.vectordb else "QDRANT" ## Need to change from string to send int 
    configuration.domain_ontology = config.ontology if config.ontology else None

    QuerySvcEnvs.set_querysvc_config(configuration)
