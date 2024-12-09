# Description: This file contains the implementation basec class for eval scores of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.vector_storage.vdb_base_component import VectorDBBaseComponent

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import pandas as pd

@dataclass
class Score:
    """
    A class used to represent a Score.
    Attributes
    ----------
    context_relevance : float
        A score representing the relevance of the context.
    faithfulness : float
        A score representing the faithfulness of the information.
    answer_relevance : float
        A score representing the relevance of the answer.
    """
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    
class EvalScoreBaseComponent(ABC):
    """
    EvalScoreBaseComponent is an abstract base class that defines the interface for evaluating scores in a retrieval-based framework.

    Methods
    -------
    score_by_retrieval(grounded_dataset: Any, **kwargs) -> float
        Return Accuracy score of Retrieval based upon framework.

    score_by_query(grounded_dataset: Any, **kwargs) -> float
        Return Accuracy score of Query based upon framework.

    score_by_vector_search(grounded_dataset: Any, vDB_component: VectorDBBaseComponent, **kwargs) -> float
        Return Accuracy score of Query based upon framework.

    get_aggregrate_score(scores: List[Score], **kwargs) -> float
        Return Accuracy score of aggregate of scores based upon framework.

    process_evaluation_score(run_id: str, evaluation_questions: List[str]) -> pd.DataFrame
        Run evaluation of data, generate scores based upon configuration.
    """

    @abstractmethod
    def score_by_retrieval(
        self,
        grounded_dataset: Any,
        **kwargs
    ) -> float:
        """Return Accuracy score of Retrieval based upon framework"""

    @abstractmethod
    def score_by_query(
        self,
        grounded_dataset: Any,
        **kwargs
    ) -> float:
        """Return Accuracy score of Query based upon framework"""

    @abstractmethod
    def score_by_vector_search(
        self,
        grounded_dataset: Any,
        vDB_component: VectorDBBaseComponent,
        **kwargs
    ) -> float:
        """Return Accuracy score of Query based upon framework"""

    @abstractmethod
    def get_aggregrate_score(
        self,
        scores:List[Score],
        **kwargs
    ) -> float:
        """"Return Accuracy score of aggregate of scores based upon framework"""

    @abstractmethod
    def process_evaluation_score(self, run_id:str, 
                                 evaluation_questions: List[str]
                                 ) -> pd.DataFrame:
        """Run evaluation of data, generate scores based upon configuration"""