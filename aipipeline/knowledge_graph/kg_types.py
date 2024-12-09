# Description: This file contains the implementation of Knowledge Graph storage types of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from typing import  List
from dataclasses import dataclass

@dataclass
class KnowledgeGraphNode:    
    """
    A class representing a node in a knowledge graph.
    Attributes:
        name (str): The name of the node.
        label (str): The label or type of the node.
        properties (dict): A dictionary of properties associated with the node.
    """
    name: str
    label: str
    properties: dict

@dataclass
class KnowledgeGraphRelationship:    
    """
    A class to represent a relationship in a knowledge graph.
    Attributes:
        start (str): The starting node of the relationship.
        end (str): The ending node of the relationship.
        properties (dict): A dictionary of properties associated with the relationship.
        type (str): The type of the relationship.
    """
    start: str
    end: str
    properties: dict
    type: str

@dataclass
class KnowledgeGraph:
    """
    A class to represent a Knowledge Graph.
    Attributes:
    ----------
    nodes : List[KnowledgeGraphNode]
        A list of nodes in the knowledge graph.
    relationships : List[KnowledgeGraphRelationship]
        A list of relationships between the nodes in the knowledge graph.
    """
    nodes: List[KnowledgeGraphNode]
    relationships: List[KnowledgeGraphRelationship]
    
