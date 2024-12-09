# Description: This file contains the retriever client tests of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase
from unittest.mock import patch, Mock

from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, LLMConfigParam, RetrieverConfigParam
from aipipeline.query.query_helpers import QuerySvcEnvs
from aipipeline.integration.retrieverClient import RetrieverClient
from test_utilities import prune_none_values_from_model

class RetrieverClientTest(TestCase):
    @patch('requests.post')
    def test_issues_request_to_server_test(self, mock_post : Mock):
        retriever = RetrieverClient(
            "http://localhost:8000",
            "session_id"
        )

        mock_post.return_value.status_code = 400
        mock_post.return_value.text  = '{"nodes":[]}'

        results = retriever.retrieve("loading data from pdf")
        self.assertListEqual(results, [])

        self.assertDictEqual(
            mock_post.call_args[1]["json"],
            {
                'query': 'loading data from pdf',
                'retrieval_configuration': None,
                'vdb_session': 'session_id'
            })
        
    @patch('requests.post')
    def test_issues_request_to_server_with_custom_configuration_test(self, mock_post : Mock):
        retriever = RetrieverClient(
            "http://localhost:8000",
            "session_id",
            retrieval_configuration=ConfigurationParam(
                llm=LLMConfigParam(
                    model="gpt-4-turbo",
                    llm_model_configuration=AzureOpenAIConfigParam(
                        temperature=0.75,
                    )
                ),
                retriever=RetrieverConfigParam(
                    top_k=10,
                    query_iteration_max=10
                )
            )
        )

        mock_post.return_value.status_code = 400
        mock_post.return_value.text  = '{"nodes":[]}'

        results = retriever.retrieve("loading data from pdf")
        self.assertListEqual(results, [])

        self.assertDictEqual(
            mock_post.call_args[1]["json"],
            {
                'query': 'loading data from pdf',
                'retrieval_configuration': {
                    'llm': {
                        'llm_model_configuration': {
                            'temperature': 0.75
                            },
                            'model': 'gpt-4-turbo'
                        },
                    'retriever': {
                        'query_iteration_max': 10,
                        'top_k': 10
                        }
                    },
                'vdb_session': 'session_id'
                })

        
