from copy import deepcopy
from dataclasses import dataclass
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode, TextNode
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Optional

# Add AutoGen and FLAML as 2nd processing step for Data labeling of unformatted data without ontology

@dataclass
class ClusterLabel:
    """
    ClusterLabel class represents a labeled cluster with a corresponding vector.
    Attributes:
        label (str): The label assigned to the cluster.
        vector (List[float]): The vector representing the cluster.
    """
    label: str
    vector: List[float]

@dataclass
class ScoredClusterLabel:
    """
    A class used to represent a labeled cluster with a score.
    Attributes
    ----------
    label : str
        The label assigned to the cluster.
    vector : List[float]
        The vector representation of the cluster.
    score : float
        The score associated with the cluster.
    """    
    label: str
    vector: List[float]
    score: float

class DataLabeller:
    """
    A class used to label data using clustering and language models.
    Attributes
    ----------
    labels : List[ClusterLabel]
        A list to store the cluster labels.
    estimator : KMeans
        The KMeans estimator used for clustering.
    llm : LLM
        The language model used for generating labels.
    embed_model : BaseEmbedding
        The embedding model used for generating text embeddings.
    Methods
    -------
    get_labels() -> List[ClusterLabel]:
        Returns a deep copy of the current labels.
    load_labels(labels: List[ClusterLabel]):
        Loads the given labels into the DataLabeller.
    learn_labels(data: List[BaseNode|str], labels_to_use: Optional[List[str]] = None):
        Learns labels from the given data using clustering and language models.
    get_labels_for_node(data: BaseNode|str, result_limit: int = 5) -> List[ScoredClusterLabel]:
        Returns a list of scored cluster labels for the given data.
    """
    def __init__(self, llm: LLM, embed_model:BaseEmbedding) -> None:
        self.labels: List[ClusterLabel] = []
        self.estimator = None
        self.llm = llm
        self.embed_model = embed_model

    def get_labels(self) -> List[ClusterLabel]:
        """
        Retrieves a deep copy of the current labels.
        Returns:
            List[ClusterLabel]: A deep copy of the list of cluster labels.
        """

        return deepcopy(self.labels)
    
    def load_labels(self, labels: List[ClusterLabel]):
        """
        Loads and deep copies the provided list of ClusterLabel objects into the instance's labels attribute.
        Args:
            labels (List[ClusterLabel]): A list of ClusterLabel objects to be loaded.
        """

        self.labels = deepcopy(labels)

    def learn_labels(self, data: List[BaseNode|str], labels_to_use:Optional[List[str]] = None):
        """
        Learns labels for the given data by clustering text embeddings and generating labels for each cluster.
        Args:
            data (List[BaseNode|str]): A list of data nodes or strings to be labeled.
            labels_to_use (Optional[List[str]]): An optional list of predefined labels to use for labeling clusters.
        Returns:
            None
        The method performs the following steps:
        1. Extracts text content from the data nodes.
        2. Generates text embeddings for the extracted text content.
        3. Determines the optimal number of clusters using silhouette scores.
        4. Creates a KMeans clustering model with the optimal number of clusters.
        5. For each cluster, ranks the labels based on cosine distances to the cluster center.
        6. Generates or selects a label for each cluster using a language model.
        7. Adds the generated labels to the list of labels if they are not already present.
        """

        vectors: List[List[float]] = []
        lookup: List[ClusterLabel] = []

        texts = [text.get_content() for text in data if isinstance(text, TextNode)]

        embeddings = self.embed_model.get_text_embedding_batch(texts)
        for text, embedding in zip(texts, embeddings):
            vectors.append(embedding)
            lookup.append(ClusterLabel(label=text, vector=embedding))

        cluster_count = max(2,  round(len(vectors) / 10))
        range_n_clusters = list (range(2,cluster_count))
        k = 0
        scores = []
        # trying to use the silhouette score to find the best number of clusters
        for n_clusters in range_n_clusters:
            self.estimator = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(vectors)
            preds = self.estimator.fit_predict(vectors)         
            score = silhouette_score(vectors, preds)
            scores.append({"n_clusters": n_clusters, "score": score})

        # create the model with optimal number of clusters
        k = min(scores, key=lambda x: abs(x["score"]))["n_clusters"]
        self.estimator = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(vectors)
        
        centres = self.estimator.cluster_centers_
        for i in range(len(centres)):
            centre = centres[i]
            ranked: List[ScoredClusterLabel] = []

            for label in lookup:
                score = cosine_distances([centre], [label.vector])[0][0]
                scoredLabel = ScoredClusterLabel(label= label.label, vector = label.vector, score = score)
                ranked.append(scoredLabel)

            ranked.sort(key=lambda x: x.score)
            ranked.reverse()
            
            prompt = ""

            if(labels_to_use is not None):
                prompt = "Select a single label from the labels from this list:\n"
                for li in labels_to_use:
                    prompt += f"   {li}\n"
                prompt = "Use a single label to describe the cluster represented by the following text blocks:\n\n:"
            else :
                prompt = "Generate a single label to describe the cluster represented by the following text blocks:\n"

            for l in range(10):
                prompt += f"   {l}. {ranked[l].label}\n"

            prompt+="\nLabel:\n"
            label = self.llm.complete(prompt).text
           
            # improve this part!!!!!
            found = False
            for current in self.labels:
                if(current.label == label):
                    found = True
                    break
            if (found == False):
                self.labels.append(ClusterLabel(label=label, vector=centre))         

    def get_labels_for_node(self, data: BaseNode|str, result_limit:int = 5) -> List[ScoredClusterLabel]: 
        """
        Get the top labels for a given node based on cosine similarity.
        Args:
            data (BaseNode | str): The node or string for which to get labels.
            result_limit (int, optional): The maximum number of labels to return. Defaults to 5.
        Returns:
            List[ScoredClusterLabel]: A list of scored cluster labels, sorted by their score in descending order.
        """

        text = ""
        if(isinstance(data, TextNode)):
            text = data.get_content()
        else:
            text = str(data)

        embedding = self.embed_model.get_text_embedding(text)
      
        ranked: List[ScoredClusterLabel] = []

        for label in self.labels:
            score = cosine_distances([embedding], [label.vector])[0][0]
            scoredLabel = ScoredClusterLabel(label= label.label, vector = label.vector, score = score)
            ranked.append(scoredLabel)

        ranked.sort(key=lambda x: x.score)
        ranked.reverse()        
       
        return ranked[:result_limit]
        
        