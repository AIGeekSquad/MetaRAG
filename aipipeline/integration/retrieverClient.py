# Description: This file contains the implementation of the retriever client for the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import List, Optional, Union
import requests

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
import json
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from aipipeline.query.query_server import RetrieveRequest
from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, CustomModelConfigParam, OpenAIConfigParam

QueryType = Union[str, QueryBundle]

import logging

logger = logging.getLogger(__name__)

class RetrieverClient(BaseRetriever):
    """
    A client for retrieving nodes from a remote service.
    Args:
        url (str): The URL of the remote service.
        vdb_session (str, optional): The session ID for the VDB. Defaults to None.
        retrieval_configuration (ConfigurationParam, optional): Configuration parameters for retrieval. Defaults to None.
        default_model_configuration (AzureOpenAIConfigParam | OpenAIConfigParam | CustomModelConfigParam, optional): Default model configuration. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): Manager for handling callbacks. Defaults to None.
        verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
    Methods:
        _set_default_model_configuration(request: RetrieveRequest):
            Sets the default model configuration for the request.
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Retrieves nodes given a query bundle.
        _aretrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Asynchronously retrieves nodes given a query bundle.
    """

    def __init__(
        self,
        url: str,
        vdb_session: str = None,
        retrieval_configuration: ConfigurationParam  = None,
        default_model_configuration: AzureOpenAIConfigParam| OpenAIConfigParam | CustomModelConfigParam= None, 
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self.url = url
        self.session_id = vdb_session
        self.retrieval_configuration = retrieval_configuration
        self.default_model_configuration = default_model_configuration
        
        super().__init__(
            callback_manager=callback_manager,
            verbose=verbose,
        )
    def _set_default_model_configuration(self, request: RetrieveRequest):
        if self.default_model_configuration is not None:
            if isinstance(self.default_model_configuration, AzureOpenAIConfigParam):
                request.azure_oai_configuration = self.default_model_configuration
            elif isinstance(self.default_model_configuration, OpenAIConfigParam):
                request.oai_configuration = self.default_model_configuration
            elif isinstance(self.default_model_configuration, CustomModelConfigParam):
                request.custom_model_configuration = self.default_model_configuration

            
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Implemented by the user.

        """
        nodes: List[NodeWithScore] = []

        retriveRequest = RetrieveRequest(
            query=query_bundle.query_str,
            vdb_session=self.session_id,
            retrieval_configuration=self.retrieval_configuration,
        )

        self._set_default_model_configuration(retriveRequest)

        request = retriveRequest.model_dump(exclude_unset=True)
        response = requests.post(
            f'{self.url}/retrieve', 
            json= request,
            headers={"Content-Type": "application/json"}
            )
        
        http_response_body = None
        if response.ok:
            http_response_body = response.text
        else:
            logger.error(f"Error retrieving nodes: {response.text}")
            raise Exception(f"Error retrieving nodes: {response.text}")
        
        # hit the endpoint get the resposne and then deserialzie it
       
        raw = json.loads(http_response_body)

        for raw_node in raw["nodes"]:
            node = metadata_dict_to_node(raw_node["node"])
            nodes.append(NodeWithScore(node=node, score=raw_node["score"]))
            
        return nodes

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronously retrieve nodes given a query.
        This method is intended to be implemented by the user to define how nodes
        should be retrieved based on the provided query bundle.
        Args:
            query_bundle (QueryBundle): The query bundle containing the query parameters.
        Returns:
            List[NodeWithScore]: A list of nodes with their associated scores.
        """
        return self._retrieve(query_bundle)