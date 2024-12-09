# Description: This file contains the benchmark app example of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent)+"\\aipipeline")

from datetime import datetime
from aipipeline.query import BenchmarkSession
from aipipeline.config.config_param_types import ConfigurationParam, QueryEngineConfigParam, RetrieverConfigParam
from aipipeline.integration.benchmark_client import BenchmarkClient

import logging

logger = logging.getLogger(__name__)

benchmark_queries =  [
    "what are the IRS tax credits and limits for the Dependent Tax Credit?", 
    "What happens if I'm late filing my taxes?",
    ]

# benchmark_queries =  [
#     "what are the most improtant takeaways for Azure Container Apps (ACA) in 2023", 
#     "what the higlight in the revenue reported in 2024 for Azure Container Apps (ACA)?",
#     "What are the important aspects for Static Web Apps revenue?",
#     "Key takeaways for AI portofolio and solutions in Azure Service",
#     "what are the challgenges for Azure App Service?",
#     "did eddie vedder act?",
#     "what was the influence for the song 'Black' by Pearl Jam?",
#     "what are the rules for battle shock test in warhammer 40k?",
#     "what are the most important influences in Prince music?",
#     "what are the highlights in boy george's career?"
#     "do we really know everything on gh gh?"
#     ]

sessions : list[BenchmarkSession] = [
    BenchmarkSession(
        session_id="using_default_configuration",
        retrieval_configuration=None
    ),
    BenchmarkSession(
        session_id="disalbing_reranker_with_basic_engine",
        retrieval_configuration=ConfigurationParam(
            retriever=RetrieverConfigParam(top_k=3),
            query_engine=QueryEngineConfigParam(iteration_max=5, use_reranker=False, type="BASIC")
            )
    ),
]

server_endpoint = "http://localhost:8000"
logger.info(f"Connecting to benchmark server at {server_endpoint}")
client = BenchmarkClient(url=server_endpoint)


results = client.run_benchmark(queries=benchmark_queries, sessions=sessions)

if results is None or results.empty:
    logger.error("No results returned from benchmark server.")
else:
    logger.info(f"Results: {results.describe()}")
    full_run_identifier = f"full_run_{datetime.now().microsecond}"
    results.to_csv(f"./reports/{full_run_identifier}.csv", index=True)

