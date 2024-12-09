# Description: This file contains the configuration loading test of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, LLMConfigParam
from aipipeline.query.query_helpers import QuerySvcEnvs
from test_utilities import prune_none_values_from_model

class ConfigurationLoading(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def test_can_uses_partial_confing_as_input(self):
        partial_config : ConfigurationParam = ConfigurationParam(
            llm= LLMConfigParam(
                model="gpt-4-turbo",
                llm_model_configuration=AzureOpenAIConfigParam(
                temperature=0.75,
                )
            )
        )

        configuration = QuerySvcEnvs.set_querysvc_config(configuration=partial_config,model_configuration=None)
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), 
        {
            "embed_model": {
                "llm_model_configuration": {
                    "api_key": "",
                    "deployment_name": "text-embedding-ada-002",
                    "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                    "is_multi_modal": False,
                    "temperature": 0.0
                },
                "model": "text-embedding-ada-002"
            },
            "llm": {
                "llm_model_configuration": {
                    "api_key": "",
                    "deployment_name": "gpt-4-turbo",
                    "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                    "is_multi_modal": False,
                    "temperature": 0.75
                },
                "model": "gpt-4-turbo"
            },
            "query_engine": {
                "iteration_max": 5,
                "llm": {
                    "llm_model_configuration": {
                        "api_key": "",
                        "deployment_name": "gpt-4-turbo",
                        "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 0.75
                    },
                    "model": "gpt-4-turbo"
                },
                "type": "FLARE",
                "use_reranker": True
            },
            "retriever": {
                "embed_model": {
                    "llm_model_configuration": {
                        "api_key": "",
                        "deployment_name": "text-embedding-ada-002",
                        "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 0.0
                    },
                    "model": "text-embedding-ada-002"
                },
                "llm": {
                    "llm_model_configuration": {
                        "api_key": "",
                        "deployment_name": "gpt-4-turbo",
                        "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 0.75
                    },
                    "model": "gpt-4-turbo"
                },
                "query_iteration_max": 4,
                "top_k": 5,
                "type": "VectorAndGraph"
            },
            "vector_database": "QDRANT",
            "vector_database_collection_name": "vector-store_collection_godzilla"
        }
                             )

    def test_override_defaults(self):
        override_defaults: AzureOpenAIConfigParam = AzureOpenAIConfigParam(
            endpoint="https://new_endpoint.azure.com/", 
            api_key="new_key", 
            version="new_version",
            temperature=1.0
            )
       
        
        configuration = QuerySvcEnvs.set_querysvc_config(configuration=None, model_configuration=override_defaults)
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), 
            {
                "embed_model": {
                    "llm_model_configuration": {
                        "api_key": "new_key",
                        "deployment_name": "text-embedding-ada-002",
                        "endpoint": "https://new_endpoint.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 1.0,
                        "version": "new_version"
                    },
                    "model": "text-embedding-ada-002"
                },
                "llm": {
                    "llm_model_configuration": {
                        "api_key": "new_key",
                        "deployment_name": "gpt-35-turbo",
                        "endpoint": "https://new_endpoint.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 1.0,
                        "version": "new_version"
                    },
                    "model": "gpt-35-turbo"
                },
                "query_engine": {
                    "iteration_max": 5,
                    "llm": {
                        "llm_model_configuration": {
                            "api_key": "new_key",
                            "deployment_name": "gpt-35-turbo",
                            "endpoint": "https://new_endpoint.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 1.0,
                            "version": "new_version"
                        },
                        "model": "gpt-35-turbo"
                    },
                    "type": "FLARE",
                    "use_reranker": True
                },
                "retriever": {
                    "embed_model": {
                        "llm_model_configuration": {
                            "api_key": "new_key",
                            "deployment_name": "text-embedding-ada-002",
                            "endpoint": "https://new_endpoint.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 1.0,
                            "version": "new_version"
                        },
                        "model": "text-embedding-ada-002"
                    },
                    "llm": {
                        "llm_model_configuration": {
                            "api_key": "new_key",
                            "deployment_name": "gpt-35-turbo",
                            "endpoint": "https://new_endpoint.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 1.0,
                            "version": "new_version"
                        },
                        "model": "gpt-35-turbo"
                    },
                    "query_iteration_max": 4,
                    "top_k": 5,
                    "type": "VectorAndGraph"
                },
                "vector_database": "QDRANT",
                "vector_database_collection_name": "vector-store_collection_godzilla"
            }
        )
       
    def test_creates_default_settings_when_passing_none(self):
        generated_config = QuerySvcEnvs.set_querysvc_config(configuration=None)
        self.assertDictEqual(
            prune_none_values_from_model(generated_config.model_dump()),
            {
                "embed_model": {
                    "llm_model_configuration": {
                        "api_key": "",
                        "deployment_name": "text-embedding-ada-002",
                        "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 0.0
                    },
                    "model": "text-embedding-ada-002"
                },
                "llm": {
                    "llm_model_configuration": {
                        "api_key": "",
                        "deployment_name": "gpt-35-turbo",
                        "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                        "is_multi_modal": False,
                        "temperature": 0.0
                    },
                    "model": "gpt-35-turbo"
                },
                "query_engine": {
                    "iteration_max": 5,
                    "llm": {
                        "llm_model_configuration": {
                            "api_key": "",
                            "deployment_name": "gpt-35-turbo",
                            "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 0.0
                        },
                        "model": "gpt-35-turbo"
                    },
                    "type": "FLARE",
                    "use_reranker": True
                },
                "retriever": {
                    "embed_model": {
                        "llm_model_configuration": {
                            "api_key": "",
                            "deployment_name": "text-embedding-ada-002",
                            "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 0.0
                        },
                        "model": "text-embedding-ada-002"
                    },
                    "llm": {
                        "llm_model_configuration": {
                            "api_key": "",
                            "deployment_name": "gpt-35-turbo",
                            "endpoint": "https://dicolomb-west-us.openai.azure.com/",
                            "is_multi_modal": False,
                            "temperature": 0.0
                        },
                        "model": "gpt-35-turbo"
                    },
                    "query_iteration_max": 4,
                    "top_k": 5,
                    "type": "VectorAndGraph"
                },
                "vector_database": "QDRANT",
                "vector_database_collection_name": "vector-store_collection_godzilla"
            }
            )

