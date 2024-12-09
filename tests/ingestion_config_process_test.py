# Description: This file contains the ingestion configuration process tests of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase
from test_utilities import prune_none_values_from_model

from aipipeline.config.config_param_types import (AzureOpenAIConfigParam, 
                                                  IngestionConfigurationParm, 
                                                  LLMConfigParam, OpenAIConfigParam)
from aipipeline.utilities.data_ingest_utilities import get_ingest_config_from_json


class IngestionProcessing(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None


    def test_IngestionConfigurationParm_config_llmmodel_type(self):
        ingest_config: str = """
        {
            "use_vector_search":true,
            "use_graph_search":true,
            "use_ontology_search":false,
            "use_tei_embed":false,
            "semantic_chucking_threshold":0.9,
            "LLMModelConfig":
            {
                "token_threshold":null,
                "model":"Gpt4turbo",
                "llm_model_configuration":
                {
                    "is_multi_modal":false,
                    "llm_usage_type":null,
                    "temperature":0.0,
                    "token_threshold":null,
                    "deployment_name":null,
                    "endpoint":null,
                    "api_key":null,
                    "version":null
                }
            },
            "EmbedModelConfig":
            {
                "token_threshold":null,
                "model":"text-embedding-ada-002",
                "llm_model_configuration":
                {
                    "is_multi_modal":false,
                    "llm_usage_type":null,
                    "temperature":0.0,
                    "token_threshold":null,
                    "deployment_name":null,
                    "endpoint":null,
                    "api_key":null,
                    "version":null
                }
            }
        }
        """
        config_for_ingest = IngestionConfigurationParm.model_validate_json(ingest_config)

        self.assertIsInstance(config_for_ingest.LLMModelConfig.llm_model_configuration, AzureOpenAIConfigParam)

    def test_IngestionConfigurationParm_config_embedmodel_type(self):
        ingest_config: str = """
        {
            "use_vector_search":true,
            "use_graph_search":true,
            "use_ontology_search":false,
            "use_tei_embed":false,
            "semantic_chucking_threshold":0.9,
            "LLMModelConfig":
            {
                "token_threshold":null,
                "model":"Gpt4turbo",
                "llm_model_configuration":
                {
                    "is_multi_modal":false,
                    "llm_usage_type":null,
                    "temperature":0.0,
                    "token_threshold":null,
                    "deployment_name":null,
                    "endpoint":null,
                    "api_key":null,
                    "version":null
                }
            },
            "EmbedModelConfig":
            {
                "token_threshold":null,
                "model":"text-embedding-ada-002",
                "llm_model_configuration":
                {
                    "is_multi_modal":false,
                    "llm_usage_type":null,
                    "temperature":0.0,
                    "token_threshold":null,
                    "deployment_name":null,
                    "endpoint":null,
                    "api_key":null,
                    "version":null
                }
            }
        }
        """
        config_for_ingest = IngestionConfigurationParm.model_validate_json(ingest_config)

        self.assertIsInstance(config_for_ingest.EmbedModelConfig.llm_model_configuration, AzureOpenAIConfigParam)

    def test_IngestionConfigurationParm_get_ingest_from_config(self):
        ingest_config: str = """
            {
            "use_vector_search": true,
            "use_graph_search": true,
            "use_ontology_search": false,
            "use_tei_embed": false,
            "semantic_chucking_threshold": 0.9,
            "LLMModelConfig": {
                "token_threshold": null,
                "model": "Gpt4turbo",
                "llm_model_configuration": {
                    "is_multi_modal": false,
                    "llm_usage_type": null,
                    "temperature": 0.0,
                    "token_threshold": null,
                    "name_of_model": null,
                    "api_key": null,
                    "version": null
                }
            },
            "EmbedModelConfig": {
                "token_threshold": null,
                "model": "text-embedding-ada-002",
                "llm_model_configuration": {
                    "is_multi_modal": false,
                    "llm_usage_type": null,
                    "temperature": 0.0,
                    "token_threshold": null,
                    "deployment_name": null,
                    "endpoint": null,
                    "api_key": null,
                    "version": null
                }
            },
            "MultiModalModelConfig": {
                "token_threshold": null,
                "model": "gpt-4-vision-preview",
                "llm_model_configuration": {
                    "is_multi_modal": true,
                    "llm_usage_type": null,
                    "temperature": 0.0,
                    "token_threshold": null,
                    "deployment_name": "gpt-4-vision",
                    "endpoint": null,
                    "api_key": null,
                    "version": null
                }
            }
        }"""

        configuration_result = get_ingest_config_from_json(ingest_config)
        self.assertIsInstance(configuration_result, IngestionConfigurationParm)