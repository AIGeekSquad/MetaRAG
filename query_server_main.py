# Description: This file contains the query, benchankmark, and retrieval server entry point of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent)+"\\aipipeline")

from fastapi import FastAPI
from aipipeline.query.query_server import BenchmarkServer, RetrieverServer, AskServer

import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)

app = FastAPI()
retriever_server = RetrieverServer(app)
ask_server = AskServer(app)
benchmark_server = BenchmarkServer(app)