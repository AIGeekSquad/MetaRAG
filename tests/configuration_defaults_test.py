# Description: This file contains the configuration defaults test of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, CustomModelConfigParam, IngestionConfigurationParm, LLMConfigParam, OpenAIConfigParam, QueryEngineConfigParam, RetrieverConfigParam
from test_utilities import prune_none_values_from_model

class ConfigurationDefaults(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def test_RetrieverConfigParam_default(self):
        retriever = RetrieverConfigParam()
        self.assertDictEqual(prune_none_values_from_model(retriever.model_dump()), {
            "query_iteration_max": 4, 
            "top_k": 5
        })

    def test_ConfigurationParam_default(self):
        configuration = ConfigurationParam()
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), {

        })
        
    def test_QueryEngineConfigParam_default(self):
        query_engine_config = QueryEngineConfigParam()
        self.assertDictEqual(prune_none_values_from_model(query_engine_config.model_dump()), {
            "iteration_max": 5,
            "use_reranker": True
        })
    
    def test_IngestionConfigurationParm_default(self):
        ingestion_config = IngestionConfigurationParm()
        self.assertDictEqual(prune_none_values_from_model(ingestion_config.model_dump()), {
            "semantic_chucking_threshold": 0.9,
            "use_graph_search": True,
            "use_ontology_search": False,
            "use_tei_embed": False,
            "use_vector_search": True
        })

    def test_LLMConfigParam_default(self):
        llm_config = LLMConfigParam()
        self.assertDictEqual(prune_none_values_from_model(llm_config.model_dump()), {
        })               

    def test_AzureOpenAIConfigParam_default(self):
        configuration = AzureOpenAIConfigParam()
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), {
            "is_multi_modal": False, 
            "temperature": 0.0
        })

    def test_OpenAIConfigParam_default(self):
        configuration = OpenAIConfigParam()
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), {
            "is_multi_modal": False, 
            "temperature": 0.0
        })

    def test_CustomModelConfigParam_default(self):
        configuration = CustomModelConfigParam()
        self.assertDictEqual(prune_none_values_from_model(configuration.model_dump()), {
            "is_multi_modal": False, 
            "temperature": 0.0
        })