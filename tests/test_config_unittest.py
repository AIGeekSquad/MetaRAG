# Description: This file contains the test config unit tests of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import unittest

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, LLMConfigParam

class TestConfig(unittest.TestCase):
    def test_config_json(self):
        retrieval_configuration=ConfigurationParam(
        # for LLM we use the GPT4turbo model, this is deployed on a specific endpoint
        llm= LLMConfigParam(
            model="Gpt4turbo",
            azure_oai_configuration=AzureOpenAIConfigParam(
                endpoint="https://dicolomb-open-ai-sweden.openai.azure.com/",
                api_key="----------------------------",
                version="2024-02-15-preview",
                )
            ),
        )
        retrieval_configuration.toJson()
        self.assertEqual(1, 1)

    def test_config_to_server(self):
        pass

if __name__ == '__main__':
    unittest.main()