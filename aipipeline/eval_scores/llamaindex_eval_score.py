# Description: This file contains the implementation LlamaIndex evaluation for eval/benchmark scores of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.eval_scores.eval_score_base import EvalScoreBaseComponent
from aipipeline.eval_scores.eval_score_base import EvalScoreBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent
from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, ContextRelevancyEvaluator, AnswerRelevancyEvaluator
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import BaseQueryEngine
from typing import Any, List
import pandas as pd

import logging

logger = logging.getLogger(__name__)

class LlamaindexEvalScoring(EvalScoreBaseComponent): 
    """
    A class used to evaluate the performance of a query engine using various scoring methods.
    Attributes
    ----------
    _query_engine : BaseQueryEngine
        The query engine used to generate responses to evaluation questions.
    _llm : LLM
        The language model used for evaluation.
    Methods
    -------
    __init__(query_engine: BaseQueryEngine, llm: LLM) -> None
        Initializes the LlamaindexEvalScoring with a query engine and a language model.
    score_by_retrieval(grounded_dataset: Any, **kwargs) -> float
        Scores the performance based on retrieval.
    score_by_query(grounded_dataset: Any, **kwargs) -> float
        Scores the performance based on query.
    score_by_vector_search(grounded_dataset: Any, vDB_component: VectorDBBaseComponent, **kwargs) -> float
        Scores the performance based on vector search.
    get_aggregrate_score(**kwargs) -> float
        Returns an aggregate score based on various evaluation metrics.
    process_evaluation_score(run_id: str, evaluation_questions: List[str]) -> pd.DataFrame
        Processes the evaluation score for a given run ID and a list of evaluation questions.
        Generates responses using the query engine and evaluates them using various evaluators.
        Parameters
        ----------
        run_id : str
            The ID of the evaluation run.
        evaluation_questions : List[str]
            A list of questions to be used for evaluation.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the evaluation results with columns: run_id, question, relevancy, faithfulness, context_relevancy, and answer_relevancy.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        llm:LLM,
        ) -> None:
        self._query_engine = query_engine
        self._llm = llm
       

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
        Processes evaluation scores for a given run ID and a list of evaluation questions.
        This method evaluates the responses to the given questions using multiple evaluators:
        RelevancyEvaluator, FaithfulnessEvaluator, ContextRelevancyEvaluator, and AnswerRelevancyEvaluator.
        It generates a response for each question, evaluates it using the mentioned evaluators, and
        compiles the results into a pandas DataFrame.
        Args:
            run_id (str): The unique identifier for the evaluation run.
            evaluation_questions (List[str]): A list of questions to be evaluated.
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results with columns:
                - "run_id": The run ID.
                - "question": The evaluation question.
                - "relevancy": The relevancy score of the response.
                - "faithfulness": The faithfulness score of the response.
                - "context_relevancy": The context relevancy score of the response.
                - "answer_relevancy": The answer relevancy score of the response.
        """         
        relevancy_evaluator = RelevancyEvaluator(llm=self._llm)
        faithfulness_evaluator = FaithfulnessEvaluator(llm=self._llm)
        context_relevancy_evaluator = ContextRelevancyEvaluator(llm=self._llm)
        answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=self._llm)

        records = []
        for question in evaluation_questions:

            # Generate response.
            # response_vector has response and source nodes (retrieved context)
            response_vector = self._query_engine.query(question)

            # Relevancy evaluation
            relevancy_eval_result = relevancy_evaluator.evaluate_response(
                    query=question, response=response_vector
            )     

            faithfulness_eval_result = faithfulness_evaluator.evaluate_response(
                    query=question, response=response_vector
            )  

            context_relevancy_eval_result = context_relevancy_evaluator.evaluate_response(
                    query=question, response=response_vector
            )  

            answer_relevancy_eval_result = answer_relevancy_evaluator.evaluate_response(
                    query=question, response=response_vector
            )     

            records.append({
                "run_id": run_id,
                "question": question,
                "relevancy": relevancy_eval_result.score,
                "faithfulness": faithfulness_eval_result.score,
                "context_relevancy": context_relevancy_eval_result.score,
                "answer_relevancy": answer_relevancy_eval_result.score
            })


        
        return pd.DataFrame(records)
