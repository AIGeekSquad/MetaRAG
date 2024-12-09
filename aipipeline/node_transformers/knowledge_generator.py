# Description: This file contains the implementation of Knowledge Graph generator of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import Any, List
from llama_index.core.bridge.pydantic import Field

from llama_index.core.schema import (
    BaseNode,
    TransformComponent,
)

import re

from llama_index.core.llms.llm import LLM

import logging

logger = logging.getLogger(__name__)

from aipipeline.knowledge_graph.node_transformations import generate_event_list_node, generate_reference_material_node, generate_summary_node, generate_takeaway_node, set_knowledge_type

class ApplyKnowledgeLabel(TransformComponent):
    """
    ApplyKnowledgeLabel is a TransformComponent that assigns a knowledge type label to nodes.
    Attributes:
        knowledge_type (str): The type of knowledge to be applied to the nodes.
        override (bool): Flag indicating whether to override the existing knowledge type. Defaults to False.
    Methods:
        __call__(nodes: List["BaseNode"], **kwargs: Any) -> List[BaseNode]:
            Transforms the given nodes by applying the knowledge type label if the node type does not match 'document'.
            Args:
                nodes (List["BaseNode"]): List of nodes to be transformed.
                **kwargs (Any): Additional keyword arguments.
            Returns:
                List[BaseNode]: The transformed list of nodes.
    """
    knowledge_type: str = Field()
    override: bool = Field(efault=False)
    
    def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List[BaseNode]:
        """Transform nodes."""
        for node in nodes:
            node_type = node.get_type()
            if not re.search('document', node_type, re.IGNORECASE):
                set_knowledge_type(node=node, knowledge_type=self.knowledge_type, override=self.override)
        return nodes

class KnowledgeGenerator(TransformComponent):
    """
    A component that generates various types of knowledge from nodes using a language model (LLM).
    Attributes:
        llm (LLM): The language model used for generating knowledge.
        generate_summary (bool): Flag to indicate if summaries should be generated.
        generate_takeaways (bool): Flag to indicate if takeaways should be generated.
        generate_event_list (bool): Flag to indicate if event lists should be generated.
        generate_reference_list (bool): Flag to indicate if reference lists should be generated.
    Methods:
        __call__(nodes: List["BaseNode"], **kwargs: Any) -> List[BaseNode]:
            Transforms nodes by generating additional knowledge nodes based on the specified flags.
            The generated nodes are appended to the original list of nodes.
    """
    llm: LLM = Field(default=None)
    generate_summary: bool = Field(default=False)
    generate_takeaways: bool = Field(efault=False)  
    generate_event_list: bool = Field(efault=False)  
    generate_reference_list: bool = Field(efault=False) 
    
    def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List[BaseNode]:
        """Transform nodes."""
        all_nodes:List[BaseNode] = []
        logger.info(f"Generating Knowledge : generate_summary={self.generate_summary}, generate_takeaways={self.generate_takeaways}, generate_event_list={self.generate_event_list}, generate_reference_list={self.generate_reference_list}")
        for node in nodes:
            if not is_chunk(node):
                        continue
            
            if self.generate_summary:              
                summary_node = generate_summary_node(node, self.llm)
                if (summary_node is not None):
                    all_nodes.append(summary_node)

            if self.generate_takeaways:
                takeaway_node = generate_takeaway_node(node, self.llm)
                if (takeaway_node is not None):
                    all_nodes.append(takeaway_node)

            if self.generate_event_list:        
                event_list_node = generate_event_list_node(node, self.llm)
                if (event_list_node is not None):
                    all_nodes.append(event_list_node)
            
            if self.generate_reference_list:              
                reference_list_node = generate_reference_material_node(node, self.llm)
                if (reference_list_node is not None):
                    all_nodes.append(reference_list_node)

        all_nodes.extend(nodes)
        return all_nodes

def is_chunk(node:BaseNode) -> bool:
    """
    Determine if a given node is of type 'Chunk'.
    Args:
        node (BaseNode): The node to be checked.
    Returns:
        bool: True if the node's metadata indicates it is a 'Chunk', False otherwise.
    """
    if (node.metadata is None ):
        return False
    return node.metadata.get("knowledgeType") == "Chunk"