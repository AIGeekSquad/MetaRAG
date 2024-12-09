# Description: This file contains the implementation embeddings generation component of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.knowledge_graph.node_transformations import contains_nodata
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.schema import BaseNode
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import List,Optional
import logging
import uuid

logger = logging.getLogger(__name__)
  
class EmbeddingGenerator: ## Pass in normal LLM model use this LLM to generate addtional text for nodes and create embeddings for look up and referenace material
    @staticmethod
    def generate_vectors(embed_model: BaseEmbedding, document_graph_store: DocumentGraphBaseComponent , client:QdrantClient, collection_name:str, batch_size:int = 100, nodes_to_process: Optional[ List[BaseNode]] = None, llm: LLM = None) -> None:        
        """
        Generates dense vectors for nodes in a document graph and indexes them in a Qdrant collection.
        Args:
            embed_model (BaseEmbedding): The embedding model used to generate dense vectors.
            document_graph_store (DocumentGraphBaseComponent): The document graph store containing nodes.
            client (QdrantClient): The Qdrant client used for indexing vectors.
            collection_name (str): The name of the Qdrant collection to index vectors into.
            batch_size (int, optional): The number of vectors to index in each batch. Defaults to 100.
            nodes_to_process (Optional[List[BaseNode]], optional): A list of nodes to process. If None, all "TextNode" nodes are processed. Defaults to None.
            llm (LLM, optional): A language model used to generate auxiliary text for nodes. Defaults to None.
        Returns:
            None
        """
        nodes = []
        if nodes_to_process is None:
            nodes_to_process = document_graph_store.get_all_nodes_by_type("TextNode")
        
        if nodes_to_process is not None:
            for node in nodes_to_process:
                text = node.get_content()
                if text is None or text.strip() == "":
                    continue            
                nodes.append({"nodeId": node.node_id, "text": text, "nodeType": node.class_name()})
        ## For each node, generate additional text tgat helps with lookups; 
        # For this node - What is node is useful for and create additional vectors based upon what this node is useful for 

        if llm is not None:
            auxilary_nodes = []

            ## Note test this prompt
            prompt = PromptTemplate(
                        """
                        From the following text extract what is the main use of this text information and when it is used. Be concise and clear.
                        Make sure to include the most important parts of the text and to write in your own words.
                        If the text contains data, or references to data, make sure to include the most important data.
                        Don't include any irrelevant information.
                        Keep all relevant numbers, informations, dates, locations and facts.
                        If the text is not appropriate do not generate a summary.
                        Always write numbers as digits, not words.
                        Expand acronyms and abbreviations.

                        If there is no relevant information in the text, return only "NODATA" and nothing else.

                        Text:
                        {text}
                        """,
                        prompt_type=PromptType.CUSTOM
            )

        
            for node in nodes:
                try:               
                   
                   auxilary_text = llm.complete(prompt, text=node["text"]).text
                   if not contains_nodata(auxilary_text):
                    auxilary_nodes.append({"nodeId": node["nodeId"], "text": auxilary_text, "nodeType": node["nodeType"]})

                except Exception as e:
                    logger.error(f"Error generating summary - {e}")

            nodes.extend(auxilary_nodes)
        
        points: List[PointStruct] = []
        logger.info(f"Total Nodes: {len(nodes)}")
        nodeBatch: List[BaseNode] = []
        for node in nodes: 
            if len(nodeBatch) == embed_model.embed_batch_size:
                # Create a batch of dense vectors
                texts = list(map(lambda x: x["text"], nodeBatch))
                dense_vectors = embed_model.get_text_embedding_batch(texts=texts)
                localPoints = [ ]

                # Create a point struct for each node
                for i in range(len(nodeBatch)):
                    dense_vector = dense_vectors[i]
                    localNode = nodeBatch[i]
                    point = PointStruct(
                        id = str(uuid.uuid4()),
                        vector = dense_vector,
                        payload={"nodeType": localNode["nodeType"], "nodeId": localNode["nodeId"]}
                    )
                    localPoints.append(point)
              
                # Add the points to the batch
                points.extend(localPoints)
                nodeBatch = []
            else:
                nodeBatch.append(node)
            if(len(points) == batch_size):
                logger.info(f"Indexing {len(points)} nodes")
                client.upsert(collection_name, points)
                points = []
        
        # If there are any remaining nodes, create a batch of dense vectors and add them to the points list
        if(len(nodeBatch) > 0):
            texts = list(map(lambda x: x["text"], nodeBatch))
            dense_vectors = embed_model.get_text_embedding_batch(texts=texts)
            localPoints = [ ]

            for i in range(len(nodeBatch)):
                dense_vector = dense_vectors[i]
                localNode = nodeBatch[i]
                point = PointStruct(
                    id = str(uuid.uuid4()),
                    vector = dense_vector,
                    payload={"nodeType": localNode["nodeType"], "nodeId": localNode["nodeId"]}
                )
                localPoints.append(point)
            points.extend(localPoints)

        # Index the remaining points
        if(len(points) > 0):
            logger.info(f"Indexing {len(points)} nodes")
            client.upsert(collection_name, points)
        logger.info("All nodes are indexed")
