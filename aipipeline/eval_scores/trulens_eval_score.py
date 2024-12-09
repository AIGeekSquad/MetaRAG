# Description: This file contains the implementation TruLens evaluation for eval/benchmark scores of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import Any, List
from llama_index.core.query_engine import BaseQueryEngine
from trulens_eval.feedback.provider.openai import AzureOpenAI
from trulens_eval import Feedback
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
from trulens_eval import TruLlama
import numpy as np

from aipipeline.eval_scores.eval_score_base import EvalScoreBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent
from aipipeline.utilities.constants import IngestionEnvs

import pandas as pd

class TruLensEvalScoring(EvalScoreBaseComponent): 
    """
    A class used to evaluate scoring using TruLens.
    Attributes
    ----------
    _query_engine : BaseQueryEngine
        The query engine used for evaluation.
    Methods
    -------
    score_by_retrieval(grounded_dataset: Any, **kwargs) -> float
        Scores based on retrieval.
    score_by_query(grounded_dataset: Any, **kwargs) -> float
        Scores based on query.
    score_by_vector_search(grounded_dataset: Any, vDB_component: VectorDBBaseComponent, **kwargs) -> float
        Scores based on vector search.
    get_aggregrate_score(**kwargs) -> float
        Gets the aggregate score.
    process_evaluation_score(run_id: str, evaluation_questions: List[str]) -> pd.DataFrame
        Processes the evaluation score and returns a DataFrame with the results.
    """
    def __init__(
            self,
            query_engine: BaseQueryEngine,
            ) -> None:
            self._query_engine = query_engine
       
            
    def score_by_retrieval(
            self,
            grounded_dataset: Any,
            **kwargs
        ) -> float:
        return 0.0

    def score_by_query(
            self,
            grounded_dataset: Any,
            **kwargs
        ) -> float:
        return 0.0

    def score_by_vector_search(
            self,
            grounded_dataset: Any,
            vDB_component: VectorDBBaseComponent,
            **kwargs
        ) -> float:
        return 0.0

    def get_aggregrate_score(
            self,
            **kwargs
        ) -> float:
        return 0.0 

    def process_evaluation_score(self, run_id:str, evaluation_questions: List[str]) -> pd.DataFrame:
        """
        Processes the evaluation score for a given run ID and a list of evaluation questions.
        Args:
            run_id (str): The unique identifier for the run.
            evaluation_questions (List[str]): A list of questions to evaluate.
        Returns:
            pd.DataFrame: A DataFrame containing the run ID and the evaluation metrics.
        """

        # from trulens_eval.app import App
        context = App.select_context(self._query_engine )
 
        provider =AzureOpenAI(
                deployment_name=IngestionEnvs.OAI_GPT35_DEPLOY_NAME,
                endpoint = IngestionEnvs.AZURE_ENDPOINT,
                api_key=IngestionEnvs.OAI_API_KEY,
                api_version=IngestionEnvs.OAI_API_VERSION
                )

        grounded = Groundedness(groundedness_provider=AzureOpenAI(
                deployment_name=IngestionEnvs.OAI_GPT35_DEPLOY_NAME,
                endpoint = IngestionEnvs.AZURE_ENDPOINT,
                api_key=IngestionEnvs.OAI_API_KEY,
                api_version=IngestionEnvs.OAI_API_VERSION
                ))

        # Define a groundedness feedback function
        f_groundedness = (
            Feedback(grounded.groundedness_measure_with_cot_reasons)
            .on(context.collect()) # collect context chunks into a list
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        # Question/answer relevance between overall question and answer.
        f_answer_relevance = (
            Feedback(provider.relevance)
            .on_input_output()
        )
        # Question/statement relevance between question and each context chunk.
        f_context_relevance = (
            Feedback(provider.context_relevance_with_cot_reasons)
            .on_input()
            .on(context)
            .aggregate(np.mean)
        )

        tru_query_engine_recorder = TruLlama(self._query_engine,
        app_id=run_id,
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])

        with tru_query_engine_recorder as recording:
            for question in evaluation_questions:
                self._query_engine.query(question)
            
        rec = recording.get()

        final_result = {}
        final_result["run_id"] = run_id
        final_result["metrics"] = rec

        return pd.DataFrame(final_result)