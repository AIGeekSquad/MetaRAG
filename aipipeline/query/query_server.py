# Description: This file contains the Fast API implementation of Query Server, Benchmark Server, and Retrieval request of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import json
from typing import List
from llama_index.core.retrievers import BaseRetriever
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from aipipeline.utilities.constants import IngestionEnvs
from aipipeline.config.config_param_types import AzureOpenAIConfigParam, BaseModelConfigParam, ConfigurationParam, CustomModelConfigParam, OpenAIConfigParam
from aipipeline.knowledge_graph.neo4j.neo4j_storage import Neo4JStorage
from aipipeline.query import query_helpers
from aipipeline.query.query_helpers import QuerySvcEnvs
from aipipeline.eval_scores.llamaindex_eval_score import LlamaindexEvalScoring
from aipipeline.query.provider_factory import ProviderFactory, QuerySvcProvider

import logging



logger = logging.getLogger(__name__)


class BenchmarkSession(BaseModel):
    """
    BenchmarkSession is a model that represents a benchmarking session with various configurations.
    Attributes:
        session_id (str): Session ID for benchmarking.
        retrieval_configuration (ConfigurationParam | None): Configuration for Retriever & Query Engine. Default is None.
        azure_oai_configuration (AzureOpenAIConfigParam | None): Configuration for Azure OpenAI Deployment to use. Default is None.
        oai_configuration (OpenAIConfigParam | None): Configuration for OpenAI Deployment to use. Default is None.
        custom_model_configuration (CustomModelConfigParam | None): Configuration for Custom Model to use. Default is None.
    """
    session_id: str = Field(title="Session ID for benchmarking")
    retrieval_configuration: ConfigurationParam | None = Field(title="Configuration for Retriever & Query Engine", default=None) 
    azure_oai_configuration: AzureOpenAIConfigParam | None = Field(title="Configuration for Azure OpenAI Deployment to Use", default=None) 
    oai_configuration: OpenAIConfigParam | None = Field(title="Configuration for OpenAI Deployment to Use", default=None)
    custom_model_configuration: CustomModelConfigParam | None = Field(title="Configuration for Custom Model to Use", default=None)  

class BenchmarkRequest(BaseModel):
    """
    BenchmarkRequest is a model for handling benchmark requests.
    Attributes:
        queries (List[str]): List of queries to benchmark.
        vdb_session (str | None): Vector DB Session/Collection to use.
        run_configurations (List[BenchmarkSession]): List of configurations to benchmark.
    """
    queries: List[str] = Field(title="List of queries to benchmark")
    vdb_session: str | None = Field(title="Vector DB Session/Collection to use")
    run_configurations: List[BenchmarkSession] = Field(title="List of configurations to benchmark", default=None)


# fast api contract objects
class RetrieveRequest(BaseModel):
    """
    RetrieveRequest is a Pydantic model that defines the structure of a request for retrieving nodes.
    Attributes:
        query (str): The query to use in the retrieval of nodes.
        vdb_session (str | None): The Vector DB Session/Collection to use. Defaults to None.
        retrieval_configuration (ConfigurationParam | None): Configuration for the Retriever & Query Engine. Defaults to None.
        azure_oai_configuration (AzureOpenAIConfigParam | None): Configuration for the Azure OpenAI Deployment to use. Defaults to None.
        oai_configuration (OpenAIConfigParam | None): Configuration for the OpenAI Deployment to use. Defaults to None.
        custom_model_configuration (CustomModelConfigParam | None): Configuration for the Custom Model to use. Defaults to None.
    """
    query: str = Field(title="Query to use in retrieval of nodes")
    vdb_session: str| None  = Field(title="Vector DB Session/Collection to use", default=None)
    retrieval_configuration: ConfigurationParam | None = Field(title="Configuration for Retriever & Query Engine", default=None) 
    azure_oai_configuration: AzureOpenAIConfigParam | None = Field(title="Configuration for Azure OpenAI Deployment to Use", default=None) 
    oai_configuration: OpenAIConfigParam | None = Field(title="Configuration for OpenAI Deployment to Use", default=None)
    custom_model_configuration: CustomModelConfigParam | None = Field(title="Configuration for Custom Model to Use", default=None)  

class GetNodeByIds(BaseModel):
    """
    GetNodeByIds is a Pydantic model used to represent a request to retrieve nodes by their IDs.
    Attributes:
        ids (List[str]): A list of node IDs to retrieve.
    """
    ids: List[str] = Field(title="List of ids to retrieve")

class AskQuestionRequest(BaseModel):
    """
    AskQuestionRequest is a Pydantic model for handling the request to ask a question.
    Attributes:
        question (str): The question to ask.
        vdb_session (str | None): The Vector DB Session/Collection to use. Default is None.
        retrieval_configuration (ConfigurationParam | None): Configuration for Retriever & Query Engine. Default is None.
        azure_oai_configuration (AzureOpenAIConfigParam | None): Configuration for Azure OpenAI Deployment to use. Default is None.
        oai_configuration (OpenAIConfigParam | None): Configuration for OpenAI Deployment to use. Default is None.
        custom_model_configuration (CustomModelConfigParam | None): Configuration for Custom Model to use. Default is None.
        include_sources (bool): Whether to include sources in the response. Default is False.
    """
    question: str = Field(title="Question to ask")
    vdb_session: str | None = Field(title="Vector DB Session/Collection to use", default=None)
    retrieval_configuration: ConfigurationParam | None = Field(title="Configuration for Retriever & Query Engine", default=None) 
    azure_oai_configuration: AzureOpenAIConfigParam | None = Field(title="Configuration for Azure OpenAI Deployment to Use", default=None) 
    oai_configuration: OpenAIConfigParam | None = Field(title="Configuration for OpenAI Deployment to Use", default=None)
    custom_model_configuration: CustomModelConfigParam | None = Field(title="Configuration for Custom Model to Use", default=None)
    include_sources: bool = Field(title="Include sources in response", default=False)

#Class for handling the answering questions from the llm model using vector DB and knowledge graph'''
class AskServer:
    """
    A server class to handle question-answering requests using internal knowledge.
    Attributes:
        _app (FastAPI): The FastAPI application instance.
    Methods:
        __init__(app: FastAPI = None):
            Initializes the AskServer instance and registers routes if an app is provided.
        _register_routes():
            Registers the API routes for the FastAPI application.
        ask(request: AskQuestionRequest):
            Handles the question-answering request, processes the query, and returns the answer.
    """

    def __init__(self, app: FastAPI = None):
        self._app = app
        if self._app is not None:
            self._register_routes()

    def _register_routes(self):           
        self._app.add_api_route(
            "/ask",
            description="answers question using internal knowdledge",
            endpoint=self.ask,
            methods=["POST"])
        
    def ask(self,  request: AskQuestionRequest ):
        """
        Handles the request to answer a question using the query engine.
        Args:
            request (AskQuestionRequest): The request object containing the question, 
                                          vector database session, retrieval configuration, 
                                          and other parameters.
        Returns:
            dict: A dictionary containing the answer to the question. If `include_sources` 
                  is True in the request, the dictionary will also include the sources 
                  with their scores and metadata.
        """

        logger.info(f"Answering qeustion: {request.question}")
        question = request.question
        request_vdb_session = request.vdb_session
        request_config = request.retrieval_configuration.model_copy(deep=True) if request.retrieval_configuration is not None else None
        request_model_config = _get_configuration_override(source=request)

        query_helpers.QuerySvcEnvs.set_querysvc_config(configuration=request_config,
                                                                       model_configuration=request_model_config)

        queryEngine: BaseQueryEngine = None
        provider : QuerySvcProvider = ProviderFactory.create(vector_db_collection_name=request_vdb_session)
        queryEngine = provider.query_engine
        
        response = queryEngine.query(question)
        answer =  {"answer": response.response_txt}    
    
        if request.include_sources:
            returnNodes = []
            for node in response.source_nodes:
                returnNodes.append({
                    "score" : node.score,
                    "node": node_to_metadata_dict(node.node)
                })
            answer["sources"] = returnNodes

        return answer
    
class RetrieverServer:
    """
    A server class to handle retrieval operations for nodes using FastAPI.
    Attributes:
        _app (FastAPI): The FastAPI application instance.
    Methods:
        __init__(app: FastAPI = None):
            Initializes the RetrieverServer instance and registers routes if an app is provided.
        _register_routes():
            Registers the API routes for retrieval operations.
        retrieve_nodes(request: RetrieveRequest):
            Handles the retrieval of nodes based on a query. Logs the query, sets up the configuration,
            retrieves nodes using the appropriate retriever, and returns the nodes with their scores.
        get_nodes_by_ids(request: GetNodeByIds):
            Retrieves nodes based on their IDs and returns them.
    """

    def __init__(self, app: FastAPI = None):
        self._app = app
        if self._app is not None:
            self._register_routes()
    
    def _register_routes(self):
        self._app.add_api_route(
            "/retrieve", 
            description="retrieve nodes by a query",
            endpoint=self.retrieve_nodes,
            methods=["POST"])
        
        self._app.add_api_route(
            "/get_node_by_ids",
            description="returns nodes by their ids",
            endpoint=self.get_nodes_by_ids,
            methods=["POST"])

    # retrieval apis
    def retrieve_nodes(self, request: RetrieveRequest):
        """
        Retrieve nodes based on the given request.
        Args:
            request (RetrieveRequest): The request object containing the query and other configurations.
        Returns:
            dict: A dictionary containing the retrieved nodes and their scores.
        The request object should have the following attributes:
            - query (str): The query string to retrieve nodes for.
            - vdb_session (str, optional): The vector database session name. Defaults to IngestionEnvs.vDB_COLLECTION_NAME if not provided.
            - retrieval_configuration (RetrievalConfiguration, optional): The configuration for retrieval. If provided, a deep copy of the model is used.
        The function performs the following steps:
            1. Logs the query being processed.
            2. Sets the query service configuration using the provided or default configurations.
            3. Creates a QuerySvcProvider instance using the vector database session name.
            4. Retrieves nodes using the provider's retriever.
            5. Converts the retrieved nodes to a list of dictionaries containing scores and metadata.
            6. Returns the list of nodes as a dictionary.
        """
        logger.info(f"Retrieving nodes for query: {request.query}")
        request_query = request.query
        request_vdb_session = request.vdb_session if request.vdb_session is not None else IngestionEnvs.vDB_COLLECTION_NAME
        request_config = request.retrieval_configuration.model_copy(deep=True) if request.retrieval_configuration is not None else None
        request_model_config = _get_configuration_override(source=request)
        

        query_helpers.QuerySvcEnvs.set_querysvc_config(configuration=request_config,
                                                                       model_configuration=request_model_config)

        retriever: BaseRetriever = None
        provider : QuerySvcProvider = ProviderFactory.create(vector_db_collection_name=request_vdb_session)

        retriever = provider.retriever
        nodes = retriever.retrieve(request_query)
        returnNodes = []
        for node in nodes:
            returnNodes.append({
                "score" : node.score,
                "node": node_to_metadata_dict(node.node)
            })
        return  {"nodes": returnNodes }
        
    def get_nodes_by_ids(self, request: GetNodeByIds):
        """
        Retrieve nodes by their IDs.
        Args:
            request (GetNodeByIds): A request object containing a list of node IDs.
        Returns:
            dict: A dictionary with a key "nodes" containing a list of metadata dictionaries for each node.
        """
        nodes = Neo4JStorage.get_nodes_by_ids(request.ids)
        return  {"nodes": [ node_to_metadata_dict(node) for node in nodes]}
    
def _get_configuration_override(source: RetrieveRequest| AskQuestionRequest | BenchmarkSession) -> BaseModelConfigParam:
    request_model_config: BaseModelConfigParam = None
    if source is None:
        return None

    if source.azure_oai_configuration is not None:
        request_model_config = source.azure_oai_configuration
    elif source.oai_configuration is not None:
        request_model_config = source.oai_configuration
    elif source.custom_model_configuration is not None:
        request_model_config = source.custom_model_configuration

    return request_model_config.model_copy(deep=True) if request_model_config is not None else None

class BenchmarkServer:
    """
    A server class to handle benchmarking requests for model performance.
    Attributes:
        _app (FastAPI): The FastAPI application instance.
    Methods:
        __init__(app: FastAPI = None):
            Initializes the BenchmarkServer with an optional FastAPI app instance.
            If an app is provided, it registers the benchmark route.
        _register_routes():
            Registers the benchmark route with the FastAPI app.
        benchmark(request: BenchmarkRequest):
            Handles the benchmarking of model performance based on the provided request.
            Processes the benchmark queries and configurations, evaluates the performance,
            and returns the results as a dictionary.
    """
    def __init__(self, app: FastAPI = None):
        self._app = app
        if self._app is not None:
            self._register_routes()

    def _register_routes(self):
        self._app.add_api_route(
            "/benchmark",
            description="benchmark the performance of a model",
            endpoint=self.benchmark,
            methods=["POST"])

    def benchmark(self, request: BenchmarkRequest):
        """
        Executes benchmarking for the provided queries and configurations.
        Args:
            request (BenchmarkRequest): The request object containing benchmark queries, 
                                        vector database session, and run configurations.
        Returns:
            dict: A dictionary containing the benchmarking results.
        Raises:
            Exception: Logs any exceptions that occur during the benchmarking process.
        The function performs the following steps:
        1. Extracts benchmark queries and vector database session name from the request.
        2. Iterates over each run configuration in the request.
        3. For each configuration, it:
            a. Logs the configuration details.
            b. Retrieves and sets the query service configuration.
            c. Creates a QuerySvcProvider instance.
            d. Initializes a LlamaindexEvalScoring evaluator.
            e. Processes the evaluation score and appends the result to the data list.
        4. Concatenates all evaluation results into a single DataFrame.
        5. Converts the DataFrame to a dictionary format.
        6. Logs the completion of benchmarking and returns the results.
        """
        benchmark_queries = request.queries
        vector_db_collection_name = request.vdb_session
        data: List[pd.DataFrame] = []
        logger.info(f"Benchmarking queries: {benchmark_queries}")
        for run_configuration in request.run_configurations:
            try :
                configuration = run_configuration.retrieval_configuration.model_copy(deep=True) if run_configuration.retrieval_configuration is not None else None
                logger.info(f"Running benchmark with configuration: {configuration}")
                
                request_model_config_override = _get_configuration_override(source=run_configuration)
                configuration = QuerySvcEnvs.set_querysvc_config(configuration=configuration,model_configuration=request_model_config_override)

                provider : QuerySvcProvider = ProviderFactory.create(vector_db_collection_name= vector_db_collection_name)
                benchmarkEvaluator = LlamaindexEvalScoring(query_engine=provider.query_engine, llm=QuerySvcEnvs.LLM_MODEL)
                # evalifier = TruLensEvalScoring(query_engine=provider.query_engine)

                report_df = benchmarkEvaluator.process_evaluation_score(
                    run_id= run_configuration.session_id, 
                    evaluation_questions= benchmark_queries)
                
                data.append(report_df)
            except Exception as e:
                logger.error(f"Error in benchmarking with configuration: {run_configuration.session_id} - {e}")
                continue

        data_as_dict = {}
        if data is not None and len(data) > 0:  
            full_data = pd.concat(data)
            data_as_dict = json.loads( full_data.to_json(orient="records"))

        logger.info(f"Benchmarking complete")
       
        result = {"result": data_as_dict}
        logger.info(f"Returning benchmark results { result}")
        return result