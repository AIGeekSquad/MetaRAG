# Description: This file contains the implementation benchmark client of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import List
import requests
import json
import pandas as pd
from aipipeline.query import BenchmarkRequest, BenchmarkSession
import logging

logger = logging.getLogger(__name__)

class BenchmarkClient:
    """
    BenchmarkClient is a client for executing benchmark sessions against a specified URL.
    Attributes:
        url (str): The URL of the benchmark service.
        vdb_session (str, optional): The session identifier for the VDB (Virtual Database).
    Methods:
        __init__(url: str, vdb_session: str = None):
            Initializes the BenchmarkClient with the given URL and optional VDB session.
        run_benchmark(queries: List[str], sessions: List[BenchmarkSession]) -> pd.DataFrame:
    """
    def __init__(self, url: str, vdb_session: str = None):
        self.url = url
        self.vdb_session = vdb_session

    def run_benchmark(self, queries : List[str], sessions: List[BenchmarkSession]) -> pd.DataFrame:
        """
        Executes benchmark sessions with the provided queries and sessions, and returns the results as a pandas DataFrame.
        Args:
            queries (List[str]): A list of query strings to be benchmarked.
            sessions (List[BenchmarkSession]): A list of BenchmarkSession objects containing the configurations for each benchmark session.
        Returns:
            pd.DataFrame: A DataFrame containing the benchmark results if the request is successful.
            None: If the request fails or no results are returned.
        Raises:
            Exception: If the request to the benchmark endpoint fails.
        """
        benchmarkRequest = BenchmarkRequest(
            queries=queries,
            vdb_session=self.vdb_session,
            run_configurations= sessions
        )
        request = benchmarkRequest.model_dump(exclude_unset=True)
        response = requests.post(
            f'{self.url}/benchmark', 
            json= request,
            headers={"Content-Type": "application/json"}
            )
        
        http_response_body = None
        source = None
        if response.ok:
            http_response_body = response.text
            raw = json.loads(http_response_body)
            source : dict = raw['result'] if 'result' in raw else {}
        else:
            logger.error(f"Error executing benchmark sessions: {response.text}")
            #raise Exception(f"Error executing benchmark sessions: {response.text}")
        
        if source is None:
            return None

        data_frame = pd.DataFrame(source)

        return data_frame
 