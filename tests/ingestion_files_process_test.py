# Description: This file contains the ingestion file process test of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from unittest import TestCase
from test_utilities import prune_none_values_from_model

from aipipeline.config.config_param_types import (AzureOpenAIConfigParam, 
                                                  IngestionConfigurationParm, 
                                                  LLMConfigParam, OpenAIConfigParam)
from aipipeline.utilities.data_ingest_utilities import get_ingest_config_from_json

class IngestionFilesProcessing(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    