# Description: This file contains the implementation Ragas evaluation for eval/benchmark scores of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import Any, List, Optional
from llama_index.core.query_engine import BaseQueryEngine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_utilization)

from aipipeline.eval_scores.eval_score_base import EvalScoreBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

import pandas as pd

class RagasEvalScoring(EvalScoreBaseComponent): 
    """
    A class used to evaluate and score the performance of a query engine using various metrics.
    Attributes
    ----------
    query_engine : BaseQueryEngine
        The query engine used to process the evaluation questions.
    _metrics : list
        A list of metrics used for evaluation, including faithfulness, answer relevancy, context utilization, and harmfulness.
    Methods
    -------
    score_by_retrieval(grounded_dataset: Any, **kwargs) -> float
        Scores the performance based on retrieval.
    score_by_query(grounded_dataset: Any, **kwargs) -> float
        Scores the performance based on query.
    score_by_vector_search(grounded_dataset: Any, vDB_component: VectorDBBaseComponent, **kwargs) -> float
        Scores the performance based on vector search.
    get_aggregrate_score(**kwargs) -> float
        Returns the aggregate score.
    process_evaluation_score(run_id: str, evaluation_questions: List[str]) -> pd.DataFrame
        Processes the evaluation score for a given run ID and list of evaluation questions, returning the results as a DataFrame.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        ) -> None:
        self._query_engine = query_engine
        self._metrics = [
            faithfulness,
            answer_relevancy,
            context_utilization,
            harmfulness,
        ]

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
            run_id (str): The unique identifier for the evaluation run.
            evaluation_questions (List[str]): A list of questions to be evaluated.
        Returns:
            pd.DataFrame: A DataFrame containing the run ID and the evaluation scores.
        """        
        contexts = []
        evaluation_answers = []
        
        for question in evaluation_questions:
            response = self._query_engine.query(question)
            contexts.append([x.node.get_content() for x in response.source_nodes])
            evaluation_answers.append(str(response))

        ds = Dataset.from_dict(
            {
                "question": evaluation_questions,
                "answer": evaluation_answers,
                "contexts": contexts,
            }
        )
       

        result = evaluate(ds, metrics= self._metrics, llm=self._query_engine.llm, verbose=True)
        
        final_result = {}
        final_result["run_id"] = run_id
        final_result["scores"] = result.scores.to_dict()
        
        return pd.DataFrame(final_result)
