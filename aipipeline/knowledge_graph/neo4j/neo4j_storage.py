# Description: This file contains the implementation of Neo4J Graph storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from llama_index.core.schema import Document,BaseNode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node,node_to_metadata_dict
from typing import Any, List, Tuple, Union, cast

from neo4j import Auth, GraphDatabase, RoutingControl
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent, KnowledgeGraphBaseComponent
from aipipeline.knowledge_graph.kg_types import KnowledgeGraph

class Neo4JStorage(DocumentGraphBaseComponent, KnowledgeGraphBaseComponent):
    """
    Neo4JStorage is a class that provides methods to interact with a Neo4j database for storing and retrieving knowledge graphs and document nodes.
    Methods:
        __init__(host_uri: str, auth: Union[Tuple[Any, Any], Auth, None], kg_database: str):
            Initializes the Neo4JStorage instance with the given host URI, authentication, and database name.
        store_knowledge_graph(kg: KnowledgeGraph) -> None:
            Stores a knowledge graph in the Neo4j database.
        add_source_relationship(source: BaseNode) -> None:
            Sets the source of a node in the graph. The source is the node that the current node is derived from.
        store_node(node: BaseNode) -> None:
            Stores a node in the graph, setting its source and metadata.
        store_document(document: Document) -> None:
            Stores a document in the graph, setting its source, metadata, and labeling the node as a Document.
        store_node_metadata_as_property(node: BaseNode) -> None:
            Applies metadata to a node in the graph.
        apply_label(allNodes: List[BaseNode], label: str) -> None:
            Applies a label to a list of nodes in the graph.
        store_nodes(allNodes: List[BaseNode]) -> None:
            Stores a list of nodes in the graph and builds relationships between them.
        get_node_by_id(node_id: str) -> BaseNode:
            Retrieves a node from the graph by its ID.
        get_chunk_node_for_id(node_id: str) -> BaseNode:
            Retrieves a chunk node from the graph by its ID.
        define_relationship(source: BaseNode, destination: BaseNode, label: str, metadata: dict[str, any] = None):
            Defines a relationship between two nodes in the graph with optional metadata.
        get_nodes_by_ids(node_ids: List[str]) -> List[BaseNode]:
            Retrieves a list of nodes from the graph by their IDs.
        get_knowledge_nodes() -> List[BaseNode]:
            Retrieves all knowledge nodes from the graph.
        get_knowledge_nodes_for_document(document_id: str) -> List[BaseNode]:
            Retrieves all knowledge nodes related to a specific document from the graph.
        get_all_nodes_by_type(node_type: str) -> List[TextNode]:
            Retrieves all nodes of a specific type from the graph.
    """

    def __init__(
            self,
            host_uri: str,
            auth: Union[Tuple[Any, Any], Auth, None],
            kg_database: str,
    ):
        self._driver = GraphDatabase.driver(host_uri, auth=auth)
        self._database = kg_database
           
    # Implement the abstract methods from the KnowledgeGraphBaseComponent class
 
    def store_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """
        Stores a knowledge graph in a Neo4j database.
        Args:
            kg (KnowledgeGraph): The knowledge graph to be stored.
        Returns:
            None
        The function iterates over the nodes and relationships in the provided knowledge graph and stores them in the Neo4j database.
        For each node, it merges the node based on its label and name, and sets its properties.
        For each relationship, it merges the relationship based on the start and end node names, and sets its properties.
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            for node in kg.nodes:
                records = session.run(  
                    "MERGE (a:"+node.label+" {name:$name}) RETURN ID(a) as id",
                    name=node.name, database_=database
                )
                
                record = records.single()
                nodeId = record["id"]

                for key in node.properties:
                    session.run(
                        "MATCH (a) where ID(a) = "+nodeId+" SET a."+key+" = $meta",
                         meta = node.properties[key], database_=database
                    )
            
            for relation in kg.relationships:
                records = session.run(
                    """MATCH (a {name: $start})
                    OPTIONAL MATCH (b {name: $end}) 
                    MERGE (a)-[r:"+relation.type+"]->(b) RETURN ID(r) AS id""",

                    start=relation.start, end = relation.end, database_=database
                )

                record = records.single()
                propertyId = record["id"]

                for key in relation.properties:
                    session.run(
                        "MATCH ()->[r:]->() where ID(r) = "+propertyId+" SET a."+key+" = $meta",
                         meta = relation.properties[key], database_=database
                    )

    # Implement the abstract methods from the DocumentGraphBaseComponent class

    def add_source_relationship(self, source: BaseNode) -> None:
        """
        Add a source relationship to a node in the Neo4j graph database.
        This method sets the source of a node, indicating the node from which the current node is derived.
        It creates a relationship of type `HAS_SOURCE` between the current node and its source node.
        Parameters:
        -----------
        source : BaseNode
            The node for which the source relationship is to be added. The `source` parameter should have
            a `source_node` attribute representing the source node and a `node_info` dictionary containing
            optional `start` and `end` attributes.
        Returns:
        --------
        None
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            if source.source_node is not None:
                start = None
                end = None
                try:
                    start = source.node_info['start']
                    end = source.node_info['end']
                except:
                    pass

                if start is not None and end is not None:
                    session.run(
                        """MATCH (a {nodeId: $nodeId}) 
                        OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                        MERGE (a)-[:HAS_SOURCE {start:$start, end:$end}]->(b)""",
                        nodeId = source.node_id, relatedNodeId = source.source_node.node_id, start = start, end= end,  database_=database,
                    )
                else:    
                    session.run(
                        """MATCH (a {nodeId: $nodeId})
                        OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                        MERGE (a)-[:HAS_SOURCE]->(b)""",
                        nodeId = source.node_id, relatedNodeId = source.source_node.node_id, database_=database,
                    )
 
    def store_node(self, node: BaseNode) -> None:   
        """
        Store a node in the Neo4j graph database. This method will set the source and metadata for the node.
        Args:
            node (BaseNode): The node to be stored in the graph. It should be an instance of BaseNode or its subclass.
        Returns:
            None
        The method performs the following steps:
        1. Establishes a session with the Neo4j database.
        2. Merges the node into the graph using its nodeId and nodeType.
        3. Converts the node's metadata into a serializable dictionary and sets each key-value pair as properties on the node.
        4. If the node has metadata, it sets additional labels and properties based on the metadata:
            - If 'knowledgeType' is present in metadata, it adds the 'KnowledgeNode' label and the specific knowledge type as a label.
            - If 'sourceType' is present in metadata, it adds the specific source type as a label.
        5. If the node is an instance of TextNode, it sets the 'text' property with the node's content.
        6. Adds source relationships and stores node metadata as properties by calling respective methods.
        Note:
            This method assumes that the node has a 'node_id' attribute, a 'class_name()' method, and a 'metadata' attribute.
        """
        driver = self._driver
        database = self._database
        metadata = node.metadata       

        with driver.session(database=database) as session:
            session.run(
                "MERGE (a {nodeId: $nodeId, nodeType: $nodeType})",
                nodeId = node.node_id,nodeType = node.class_name(), database_=database,
            )

            serializableDictionary = node_to_metadata_dict(node)
            for key in serializableDictionary:
                session.run(
                    "MATCH (a {nodeId: $nodeId}) SET a."+key+" = $meta",
                    nodeId = node.node_id, meta = serializableDictionary[key], database_=database,
                )
                
            if metadata is not None:
                    knowledgeType =  metadata.get("knowledgeType")
                    if knowledgeType is not None:
                        session.run(
                            "MATCH (a {nodeId: $nodeId}) SET a:KnowledgeNode",
                            nodeId = node.node_id,  database_=database,
                        )
                        session.run(
                            "MATCH (a {nodeId: $nodeId}) SET a:"+knowledgeType,
                            nodeId = node.node_id,  database_=database,
                        )
                    sourceType = metadata.get("sourceType")
                    if sourceType is not None:
                        session.run(
                            "MATCH (a {nodeId: $nodeId}) SET a:"+sourceType,
                            nodeId = node.node_id,  database_=database,
                        )
            
            if(isinstance(node, TextNode)):
                session.run(
                    "MATCH (a {nodeId: $nodeId}) SET a.text = $text",
                    nodeId = node.node_id, text = node.get_content(), database_=database,
                )

        self.add_source_relationship(node)
        self.store_node_metadata_as_property(node)

    def store_document(self, document: Document) -> None:   
        """
        Store a document in the graph.
        This method stores a document node in the graph database, sets its source and metadata, 
        and labels the node as "Document".
        Args:
            document (Document): The document to be stored in the graph.
        """
        self.store_node(document)   
        self.apply_label([document], "Document")
            
    def store_node_metadata_as_property(self, node: BaseNode) -> None:   
        """
        Store metadata as properties of a node in the Neo4j graph database.
        This method applies the metadata from a given node to the corresponding node
        in the Neo4j graph database by setting each metadata key-value pair as a property
        of the node.
        Args:
            node (BaseNode): The node object containing metadata to be stored.
        Returns:
            None
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            meta = node.metadata
            for key in meta:
                session.run(
                    "MATCH (a {nodeId: $nodeId}) SET a."+key+" = $meta",
                    nodeId = node.node_id, meta = meta[key], database_=database,
                )
 
    def apply_label(self, allNodes:List[BaseNode], label:str) -> None:
        """
        Applies a specified label to a list of nodes in the Neo4j database.
        Args:
            allNodes (List[BaseNode]): A list of nodes to which the label will be applied.
            label (str): The label to be applied to each node.
        Returns:
            None
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            for node in allNodes:
                session.run(
                    "MATCH (n {nodeId: $nodeId}) SET n:"+label,
                    nodeId = node.node_id,nodeType = node.class_name(), database_=database,
                )

    def store_nodes(self, allNodes:List[BaseNode]) -> None:   
        """
        Stores a list of nodes in the Neo4j database and builds relationships between them.
        Args:
            allNodes (List[BaseNode]): A list of nodes to be stored in the database.
        Returns:
            None
        The method performs the following steps:
        1. Iterates over all nodes and stores each node individually using the `store_node` method.
        2. Establishes relationships between nodes:
            - Creates a HAS_NEXT relationship if the node has a `next_node`.
            - Creates HAS_CHILD relationships for each child in `child_nodes`.
            - Creates a HAS_PARENT relationship if the node has a `parent_node`.
            - Creates a HAS_PREV relationship if the node has a `prev_node`.
        """
        driver = self._driver
        database = self._database    
        for node in allNodes:
            self.store_node(node)
                
        # build relationships
            
        for node in allNodes:
            with driver.session(database=database) as session:          
                if node.next_node is not None:
                    session.run(
                        """MATCH (a {nodeId: $nodeId})
                        OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                        MERGE (a)-[:HAS_NEXT]->(b)""",
                        nodeId = node.node_id, relatedNodeId = node.next_node.node_id, database_=database,
                    )
                if node.child_nodes is not None:
                    for child in node.child_nodes:
                        session.run(
                            """MATCH (a {nodeId: $nodeId})
                            OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                            MERGE (a)-[:HAS_CHILD]->(b)""",
                            nodeId = node.node_id, relatedNodeId = child.node_id, database_=database,
                        )
                if node.parent_node is not None:
                    session.run(
                        """MATCH (a {nodeId: $nodeId})
                        OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                        MERGE (a)-[:HAS_PARENT]->(b)""",
                        nodeId = node.node_id, relatedNodeId = node.parent_node.node_id, database_=database,
                    )

                if node.prev_node is not None:
                    session.run(
                        """MATCH (a {nodeId: $nodeId}) 
                        OPTIONAL MATCH (b {nodeId: $relatedNodeId}) 
                        MERGE (a)-[:HAS_PREV]->(b)""",
                        nodeId = node.node_id, relatedNodeId = node.prev_node.node_id, database_=database,
                    ) 
 
    def get_node_by_id(self, node_id: str) -> BaseNode:
        """
        Retrieve a node from the Neo4j database by its node ID.
        Args:
            node_id (str): The unique identifier of the node to retrieve.
        Returns:
            BaseNode: The node object corresponding to the given node ID, or None if no such node exists.
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            records = session.run(
                "MATCH (a {nodeId: $nodeId}) RETURN a._node_content, a._node_type",
                nodeId = node_id, database_ = database, routing_ = RoutingControl.READ
            )
            record = records.single()
            if (record is None):
                return None
            meta = {
                "_node_content": str(record["a._node_content"]),
                "_node_type": str(record["a._node_type"]),
            }
            return metadata_dict_to_node(meta)
    
    def get_chunk_node_for_id(self, node_id: str) -> BaseNode:
        """
        Retrieve a chunk node for a given node ID from the Neo4j database.
        This method queries the Neo4j database to find a node with the specified node ID.
        If the node is a Chunk or has a parent Chunk, it returns the content and type of the node.
        Args:
            node_id (str): The ID of the node to retrieve.
        Returns:
            BaseNode: The node with the specified ID, or None if no such node exists.
        """

        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            records = session.run(
                """MATCH (d {nodeId: $nodeId})
OPTIONAL MATCH (d)-[:HAS_PARENT *1..]->(c:Chunk ) WHERE (NOT c.text =~ '\s+')
With
CASE 
    WHEN d:Chunk THEN d
    ELSE c
END as a 
return a._node_content, a._node_type""",
                nodeId = node_id, database_ = database, routing_ = RoutingControl.READ
            )
            record = records.single()
            if (record is None):
                return None
            meta = {
                "_node_content": str(record["a._node_content"]),
                "_node_type": str(record["a._node_type"]),
            }
            return metadata_dict_to_node(meta)
             
    def define_relationship(self, source:BaseNode, destination:BaseNode, label:str, metadata: dict[str, any] = None):
        """
        Defines a relationship between two nodes in the Neo4j database.
        Args:
            source (BaseNode): The source node of the relationship.
            destination (BaseNode): The destination node of the relationship.
            label (str): The label of the relationship.
            metadata (dict[str, any], optional): Additional metadata to set on the relationship. Defaults to None.
        Raises:
            Exception: If there is an error while creating the relationship or setting the metadata.
        Example:
            source_node = BaseNode(node_id="1")
            destination_node = BaseNode(node_id="2")
            relationship_label = "FRIENDS_WITH"
            relationship_metadata = {"since": "2021"}
            define_relationship(source_node, destination_node, relationship_label, relationship_metadata)
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            session.run(
                """MATCH (a {nodeId: $source})
                OPTIONAL MATCH (b {nodeId: $destination}) 
                MERGE (a)-[r:"""+label+"]->(b)",
                source = source.node_id, destination = destination.node_id, database_=database,
            )

            if metadata is not None:
                for key in metadata:
                    session.run(
                        "MATCH (a {nodeId: $source})-[r:"+label+"]->(b {nodeId: $destination}) SET r."+key+" = $meta",
                        source = source.node_id, destination = destination.node_id, meta = metadata[key], database_=database,
                    )
        
    def get_nodes_by_ids(self, node_ids: List[str]) -> List[BaseNode]:
        """
        Retrieve nodes from the Neo4j database by their IDs.
        Args:
            node_ids (List[str]): A list of node IDs to retrieve.
        Returns:
            List[BaseNode]: A list of nodes corresponding to the provided IDs.
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            records = session.run(
                f"""MATCH (a) WHERE a.nodeId IN [{' '.join(["'" + str(item) + "'" for item in node_ids]) }] RETURN a._node_content, a._node_type""",
                database_ = database
            )

            nodes = []
            for record in records:
                meta = {
                    "_node_content": str(record["a._node_content"]),
                    "_node_type": str(record["a._node_type"]),
                }
                nodes.append(metadata_dict_to_node(meta))
            
            return nodes
    
    def get_knowledge_nodes(self) -> List[BaseNode]:
        """
        Retrieves all knowledge nodes from the Neo4j database.
        This method connects to the Neo4j database using the provided driver and 
        retrieves all nodes labeled as 'KnowledgeNode'. Each node's content and 
        type are extracted and converted into a BaseNode object.
        Returns:
            List[BaseNode]: A list of BaseNode objects representing the knowledge nodes.
        """
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
            records = session.run(
                f"""MATCH (a:KnowledgeNode) RETURN a._node_content, a._node_type""",
                database_ = database
            )

            nodes = []
            for record in records:
                meta = {
                    "_node_content": str(record["a._node_content"]),
                    "_node_type": str(record["a._node_type"]),
                }
                nodes.append(metadata_dict_to_node(meta))
            
            return nodes
        
    def get_knowledge_nodes_for_document(self, document_id: str) -> List[BaseNode]:
        """
        Retrieve knowledge nodes associated with a specific document from the Neo4j database.
        Args:
            document_id (str): The unique identifier of the document for which to retrieve knowledge nodes.
        Returns:
            List[BaseNode]: A list of BaseNode objects representing the knowledge nodes associated with the document.
        """
        driver = self._driver
        database = self._database
        nodes:List[BaseNode] = []
        with driver.session(database=database) as session:
            records = session.run(
                "MATCH (d:Document {nodeId: $nodeId})<-[:HAS_SOURCE *1..]-(a:KnowledgeNode&!Summary ) WHERE (NOT a.text =~ '\s+')  RETURN a._node_content, a._node_type",
                database_ = database, documentId = document_id
            )
           
            for record in records:
                meta = {
                    "_node_content": str(record["a._node_content"]),
                    "_node_type": str(record["a._node_type"]),
                }
                nodes.append(metadata_dict_to_node(meta))
            
            return nodes
    
    def get_all_nodes_by_type(self, node_type:str) -> List[TextNode]:
        """
        Retrieve all nodes of a specific type from the Neo4j database.
        Args:
            node_type (str): The type of nodes to retrieve.
        Returns:
            List[TextNode]: A list of TextNode objects that match the specified node type.
        """
        nodes:List[TextNode] = []
        driver = self._driver
        database = self._database
        with driver.session(database=database) as session:
                records  = session.run(
                    "MATCH (a) WHERE a.nodeType = $nodeType AND (NOT a.text =~ '\s+')  RETURN a._node_content, a._node_type",
                    nodeType = node_type, database_ = database
                )
                for record in records:
                    meta = {
                        "_node_content": str(record["a._node_content"]),
                        "_node_type": str(record["a._node_type"]),
                    }
                    nodes.append(metadata_dict_to_node(meta))
        return nodes