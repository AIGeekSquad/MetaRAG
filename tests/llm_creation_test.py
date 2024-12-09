# Description: This file contains the LLM creation tests of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase
from llama_index.core.llms.llm import LLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, LLMConfigParam
from aipipeline.query.query_helpers import get_llm_model, get_embed_model

class LLMCreation(TestCase):

    def test_create_llm_model(self):
        llm_model = get_llm_model(LLMConfigParam(
            model="gpt-4",
            llm_model_configuration=AzureOpenAIConfigParam(
                api_key="test",
            )
        ))

        self.assertIsInstance(llm_model, LLM)

    def test_create_llm_multimodel(self):
        llm_model = get_llm_model(LLMConfigParam(
            model="gpt-4-vision",
            llm_model_configuration=AzureOpenAIConfigParam(
                is_multi_modal=True,
                api_key="test",
            )
        ))

        self.assertIsInstance(llm_model, MultiModalLLM)
    
    def test_create_embed_model(self):
        embed_model = get_embed_model(LLMConfigParam(
            model="text-embedding-ada-002",
            llm_model_configuration=AzureOpenAIConfigParam(
                api_key="test",
            )
        ))

        self.assertIsInstance(embed_model, BaseEmbedding)