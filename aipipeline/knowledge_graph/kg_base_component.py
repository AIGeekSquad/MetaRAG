# Description: This file contains the base class implementation of Knowledge Graph storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.graph_stores.types import GraphStore

from aipipeline.knowledge_graph.kg_types import KnowledgeGraph

#TODO: Tara & Diego - Discuss how to split between KnowledgeGraph operations and Document retreival based schema

class KnowledgeGraphBaseComponent(GraphStore, ABC):
    """
    Base component for managing and storing knowledge graphs.

    This abstract base class provides the foundational methods and attributes
    required for interacting with a knowledge graph store.

    Methods
    -------
    store_knowledge_graph(kg: KnowledgeGraph) -> None
        Store the provided knowledge graph into the graph store.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph to be stored.
    """
    def store_knowledge_graph(self, kg: KnowledgeGraph                       
        ) -> None:
         """Store Knowledge Graph"""
     

class DocumentGraphBaseComponent(GraphStore, ABC):
    """
    DocumentGraphBaseComponent is an abstract base class that defines the interface for a document graph component. 
    This component is responsible for managing nodes and relationships within a knowledge graph. 
    """

   # Add driver, database in constructor in implementation into client of GraphStore
    @abstractmethod
    def add_source_relationship(self, source: BaseNode) -> None:
        '''
        Set the source of a node in the graph. The source is the node that the current node is derived from.
        '''
        
    @abstractmethod
    def store_node(self, node: BaseNode) -> None:   
        '''
        Store a node in the graph. Will set source and metadata.
        '''
    
    @abstractmethod
    def store_document(self, document: Document) -> None:   
        '''
        Store a document in the graph. Will set source and metadata and label the node as Document.
        '''
       
    @abstractmethod
    def store_node_metadata_as_property(self, node: BaseNode) -> None:   
        '''
        Apply metadata to a node in the graph.
        '''
    @abstractmethod
    def apply_label(self, allNodes:List[BaseNode], label:str) -> None:
      '''
        Apply label in Knowledge Graph.
        '''

    @abstractmethod
    def store_nodes(self, allNodes:List[BaseNode]) -> None:       
         '''
        Store Knowledge Graph Nodes.
        '''
                    
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> BaseNode:
         '''
        Get A Node by ID from Knowledge Graph.
        '''
         
    @abstractmethod
    def get_chunk_node_for_id(self, node_id: str) -> BaseNode:
            '''
            Get A Chunk Node by ID from Knowledge Graph. If the node is not a chunk node, return the first parent node that is a CHUNK.
            '''
    @abstractmethod
    def define_relationship(self, source:BaseNode, destination:BaseNode, label:str, metadata: dict[str, any] = None):
         '''
        Define Knowledge Graph relationship.
        '''
        
    @abstractmethod
    def get_nodes_by_ids(self, node_ids: List[str]) -> List[BaseNode]:
         '''
        Get Nodes by ID from Knowledge Graph.
        '''
    
    @abstractmethod
    def get_knowledge_nodes(self) -> List[BaseNode]:
       '''
        Get Knowledge Nodes from Knowledge Graph.
        '''
        
    @abstractmethod 
    def get_knowledge_nodes_for_document(self, document_id: str) -> List[BaseNode]:
        '''
        Get Knowledge Nodes for a Document by Document ID.  
        '''
        
    @abstractmethod
    def get_all_nodes_by_type(self, node_type:str) -> List[TextNode]:
        '''
        Get All Nodesmatching the type from the Document Graph Store.
        '''