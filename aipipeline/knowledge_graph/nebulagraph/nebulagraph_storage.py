# Description: This file contains the WIP implementation of Nebula Graph storage of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import uuid
from llama_index.core.schema import Document,BaseNode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node,node_to_metadata_dict
from typing import List

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent, KnowledgeGraphBaseComponent
from aipipeline.knowledge_graph.kg_types import KnowledgeGraph

class NebulaStorage(DocumentGraphBaseComponent, KnowledgeGraphBaseComponent):
    class NebulaStorage:
        """
        NebulaStorage is a class that provides methods to store and manage a knowledge graph in a NebulaGraph database.
        Methods
        -------
        __init__(host_uri: str, kg_user: str, kg_pass: str, kg_database: str)
            Initializes the NebulaStorage instance with the given host URI, user credentials, and database name.
        store_knowledge_graph(kg: KnowledgeGraph) -> None
            Stores the entire knowledge graph in the NebulaGraph database.
        add_source_relationship(source: BaseNode) -> None
            Sets the source of a node in the graph. The source is the node that the current node is derived from.
        store_node(node: BaseNode) -> None
            Stores a node in the graph and sets its source and metadata.
        store_document(document: Document) -> None
            Stores a document in the graph, sets its source and metadata, and labels the node as a Document.
        store_node_metadata_as_property(node: BaseNode) -> None
            Applies metadata to a node in the graph.
        apply_label(allNodes: List[BaseNode], label: str) -> None
            Applies a label to a list of nodes in the graph.
        store_nodes(allNodes: List[BaseNode]) -> None
            Stores a list of nodes in the graph and builds relationships between them.
        get_node_by_id(node_id: str) -> BaseNode
            Retrieves a node from the graph by its ID.
        get_chunk_node_for_id(node_id: str) -> BaseNode
            Not implemented. Raises NotImplementedError.
        define_relationship(source: BaseNode, destination: BaseNode, label: str, metadata: dict[str, any] = None)
            Defines a relationship between two nodes in the graph and optionally sets metadata for the relationship.
        get_nodes_by_ids(node_ids: List[str]) -> List[BaseNode]
            Retrieves a list of nodes from the graph by their IDs.
        get_knowledge_nodes() -> List[BaseNode]
            Retrieves all knowledge nodes from the graph.
        get_knowledge_nodes_for_document(document_id: str) -> List[BaseNode]
            Retrieves all knowledge nodes related to a specific document from the graph.
        get_all_nodes_by_type(node_type: str) -> List[TextNode]
            Retrieves all nodes of a specific type from the graph.
        """

    def __init__(
            self,
            host_uri: str, #change to url and auth
            kg_user: str,
            kg_pass: str,
            kg_database: str
    ):
        self._nebulaconfig = Config()
        self._nebulaconfig.max_connection_pool_size = 5
        self._connection_pool = ConnectionPool()
        self._isactiveconnection = self._connection_pool.init([(host_uri, 9669)], self._nebulaconfig)
        self._database = kg_database
        self._kgUser = kg_user
        self._kgPass = kg_pass
           
    # Implement the abstract methods from the KnowledgeGraphBaseComponent class
 
    def store_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """
        Stores a knowledge graph in the NebulaGraph database.
        Args:
            kg (KnowledgeGraph): The knowledge graph to be stored.
        Returns:
            None
        The function performs the following steps:
        1. Connects to the NebulaGraph database using the provided credentials.
        2. Creates a space for the knowledge graph if it does not already exist.
        3. Iterates over the nodes in the knowledge graph:
            a. Creates a tag for each node label if it does not already exist.
            b. Inserts each node as a vertex in the graph.
            c. Updates the properties of each node.
        4. Iterates over the relationships in the knowledge graph:
            a. Inserts each relationship as an edge in the graph.
            b. Updates the properties of each relationship.
        """
        database = self._database

        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            execution_cmd = f"CREATE SPACE IF NOT EXISTS {database}(vid_type=FIXED_STRING(38)); USE {database};"
            client.execute(execution_cmd)
            
            for node in kg.nodes:
                exec_cmd = f"CREATE TAG IF NOT EXISTS {node.label}(name string)"
                client.execute(exec_cmd)
                
                vertex_id = str(uuid.uuid4())
                exec_cmd = f"INSERT VERTEX {node.label} (name) VALUES {vertex_id}:({node.name}) YIELD id(vertex) as id"
                records = client.execute(exec_cmd)

                record = records.rows[0]
                nodeId = record["id"]
                
                for key in node.properties:
                    exec_cmd = f"UPDATE VERTEX {nodeId} SET {key} = {node.properties[key]}"
                    client.execute(exec_cmd)
            
            for relation in kg.relationships:
                exec_cmd = f"INSERT EDGE {relation.type}() VALUES {relation.start}-> {relation.end}; FETCH PROP ON {relation.type} 
                {relation.start}->{relation.end};"
                records = client.execute(exec_cmd)
                
                record = records.rows[0]
                propertyId = record["id"]

                for key in relation.properties:
                    exec_cmd = f'UPDATE EDGE {relation.start} -> {relation.end} OF {relation.type} SET {key} = {relation.properties[key]}'
                    client.execute(exec_cmd)
                

    # Implement the abstract methods from the DocumentGraphBaseComponent class

    def add_source_relationship(self, source: BaseNode) -> None:
        """
        Add a source relationship to the current node in the graph.
        This method sets the source of a node in the graph. The source is the node 
        that the current node is derived from. It creates an edge of type HAS_SOURCE 
        between the current node and the source node.
        Parameters:
        source (BaseNode): The source node from which the current node is derived.
        Returns:
        None
        """
        exec_cmd: str = ""
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            if source.source_node is not None:
                start = None
                end = None
                try:
                    start = source.node_info['start']
                    end = source.node_info['end']
                except:
                    pass

                if start is not None and end is not None:
                    exec_cmd= f"INSERT EDGE HAS_SOURCE(start, end) VALUES {id}->{source.source_node.node_id}:({start}, {end});"
                    client.execute(exec_cmd)
                else:
                    exec_cmd= f"INSERT EDGE HAS_SOURCE() VALUES {id}->{source.source_node.node_id};"    
                    client.execute(exec_cmd)
 
    def store_node(self, node: BaseNode) -> None:   
        """
        Store a node in the graph. This method sets the source and metadata for the node.
        Args:
            node (BaseNode): The node to be stored in the graph.
        Returns:
            None
        The method performs the following steps:
        1. Retrieves the metadata from the node.
        2. Establishes a session with the graph database using the connection pool.
        3. Executes commands to store the node and its metadata in the graph.
        4. If the node has specific metadata such as 'knowledgeType' or 'sourceType', it sets the corresponding labels.
        5. If the node is an instance of TextNode, it sets the text content of the node.
        6. Adds source relationships and stores node metadata as properties.
        """
        
        metadata = node.metadata       

        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            exec_cmd: str = ""
            client.execute(exec_cmd)
            # session.run(
            #     "MERGE (a {nodeId: $nodeId, nodeType: $nodeType})",
            #     nodeId = node.node_id,nodeType = node.class_name(), database_=database,
            # )

            serializableDictionary = node_to_metadata_dict(node)
            for key in serializableDictionary:
                exec_cmd: str = ""
                client.execute(exec_cmd)
                # session.run(
                #     "MATCH (a {nodeId: $nodeId}) SET a."+key+" = $meta",
                #     nodeId = node.node_id, meta = serializableDictionary[key], database_=database,
                # )
                
            if metadata is not None:
                    knowledgeType =  metadata.get("knowledgeType")
                    if knowledgeType is not None:
                        exec_cmd: str = ""
                        client.execute(exec_cmd)
                        # session.run(
                        #     "MATCH (a {nodeId: $nodeId}) SET a:KnowledgeNode",
                        #     nodeId = node.node_id,  database_=database,
                        # )
                        exec_cmd = ""
                        client.execute(exec_cmd)
                        # session.run(
                        #     "MATCH (a {nodeId: $nodeId}) SET a:"+knowledgeType,
                        #     nodeId = node.node_id,  database_=database,
                        # )
                    sourceType = metadata.get("sourceType")
                    if sourceType is not None:
                        exec_cmd: str = ""
                        client.execute(exec_cmd)
                        # session.run(
                        #     "MATCH (a {nodeId: $nodeId}) SET a:"+sourceType,
                        #     nodeId = node.node_id,  database_=database,
                        # )
            
            if(isinstance(node, TextNode)):
                # client.execute(
                #     "MATCH (a {nodeId: $nodeId}) SET a.text = $text",
                #     nodeId = node.node_id, text = node.get_content(), database_=database,
                # )
                pass

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
        Store metadata as properties of a node in the graph.

        This method applies the metadata of a given node to the corresponding node
        in the graph database. Each key-value pair in the node's metadata is set as
        a property of the node in the graph.

        Args:
            node (BaseNode): The node whose metadata is to be stored as properties
                            in the graph.

        Returns:
            None
        """
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            meta = node.metadata
            for key in meta:
                client.execute(
                    "MATCH (a {nodeId: $nodeId}) SET a."+key+" = $meta",
                    nodeId = node.node_id, meta = meta[key], database_=database,
                )
 
    def apply_label(self, allNodes:List[BaseNode], label:str) -> None:
        """
        Apply a label to a list of nodes in the knowledge graph.
        Args:
            allNodes (List[BaseNode]): A list of nodes to which the label will be applied.
            label (str): The label to be applied to the nodes.
        Returns:
            None
        """
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            for node in allNodes:
                client.execute(
                    "MATCH (n {nodeId: $nodeId}) SET n:"+label,
                    nodeId = node.node_id,nodeType = node.class_name(), database_=database,
                )

    def store_nodes(self, allNodes:List[BaseNode]) -> None:   
        """
        Stores a list of nodes in the database and builds relationships between them.
        Args:
            allNodes (List[BaseNode]): A list of nodes to be stored in the database.
        Returns:
            None
        The function performs the following steps:
        1. Iterates over all nodes and stores each node using the `store_node` method.
        2. Establishes relationships between nodes:
            - If a node has a `next_node`, creates a `HAS_NEXT` relationship.
            - If a node has `child_nodes`, creates `HAS_CHILD` relationships for each child.
            - If a node has a `parent_node`, creates a `HAS_PARENT` relationship.
            - If a node has a `prev_node`, creates a `HAS_PREV` relationship.
        """

        driver = self._driver
        database = self._database    
        for node in allNodes:
            self.store_node(node)
                
        # build relationships
            
        for node in allNodes:
            with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:        
                if node.next_node is not None:
                    client.execute(
                        "MATCH (a {nodeId: $nodeId}), (b {nodeId: $relatedNodeId}) MERGE (a)-[:HAS_NEXT]->(b)",
                        nodeId = node.node_id, relatedNodeId = node.next_node.node_id, database_=database,
                    )
                if node.child_nodes is not None:
                    for child in node.child_nodes:
                        client.execute(
                            "MATCH (a {nodeId: $nodeId}), (b {nodeId: $relatedNodeId}) MERGE (a)-[:HAS_CHILD]->(b)",
                            nodeId = node.node_id, relatedNodeId = child.node_id, database_=database,
                        )
                if node.parent_node is not None:
                    client.execute(
                        "MATCH (a {nodeId: $nodeId}), (b {nodeId: $relatedNodeId}) MERGE (a)-[:HAS_PARENT]->(b)",
                        nodeId = node.node_id, relatedNodeId = node.parent_node.node_id, database_=database,
                    )

                if node.prev_node is not None:
                    client.execute(
                        "MATCH (a {nodeId: $nodeId}), (b {nodeId: $relatedNodeId}) MERGE (a)-[:HAS_PREV]->(b)",
                        nodeId = node.node_id, relatedNodeId = node.prev_node.node_id, database_=database,
                    ) 
 
    def get_node_by_id(self, node_id: str) -> BaseNode:
        """
        Retrieve a node from the knowledge graph by its ID.
        Args:
            node_id (str): The ID of the node to retrieve.
        Returns:
            BaseNode: The node corresponding to the given ID, or None if no such node exists.
        """
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            records = client.execute(
                "MATCH (a {nodeId: $nodeId}) RETURN a._node_content, a._node_type",
             #   nodeId = node_id, database_ = database, routing_ = RoutingControl.READ
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
        raise NotImplementedError("Method get_chunk_node_for_id not implemented")
         
    def define_relationship(self, source:BaseNode, destination:BaseNode, label:str, metadata: dict[str, any] = None):
        """
        Defines a relationship between two nodes in the knowledge graph.
        Args:
            source (BaseNode): The source node of the relationship.
            destination (BaseNode): The destination node of the relationship.
            label (str): The label for the relationship.
            metadata (dict[str, any], optional): Additional metadata to be added to the relationship. Defaults to None.
        Raises:
            Exception: If there is an error executing the database commands.
        Example:
            source_node = BaseNode(node_id="1")
            destination_node = BaseNode(node_id="2")
            relationship_label = "FRIENDS_WITH"
            metadata = {"since": "2021"}
            define_relationship(source_node, destination_node, relationship_label, metadata)
        """        
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            client.execute(
                "MATCH (a {nodeId: $source}), (b {nodeId: $destination}) MERGE (a)-[r:"+label+"]->(b)",
                source = source.node_id, destination = destination.node_id, database_=database,
            )

            if metadata is not None:
                for key in metadata:
                    client.execute(
                        "MATCH (a {nodeId: $source})-[r:"+label+"]->(b {nodeId: $destination}) SET r."+key+" = $meta",
                        source = source.node_id, destination = destination.node_id, meta = metadata[key], database_=database,
                    )
        
    def get_nodes_by_ids(self, node_ids: List[str]) -> List[BaseNode]:
        """
        Retrieve nodes from the knowledge graph by their IDs.
        Args:
            node_ids (List[str]): A list of node IDs to retrieve.
        Returns:
            List[BaseNode]: A list of BaseNode objects corresponding to the provided node IDs.
        Raises:
            Exception: If there is an error during the database query or node retrieval process.
        """
        ##driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            records = client.execute(
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
        Retrieves knowledge nodes from the database.
        This method connects to the knowledge graph database using the provided
        connection pool and user credentials. It executes a query to fetch all
        nodes labeled as 'KnowledgeNode' and returns them as a list of BaseNode
        objects.
        Returns:
            List[BaseNode]: A list of knowledge nodes retrieved from the database.
        """
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            records = client.execute(
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
        Retrieve knowledge nodes associated with a specific document.
        This method queries the knowledge graph to find all knowledge nodes
        that are linked to the given document ID. It excludes nodes that are
        summaries or contain only whitespace text.
        Args:
            document_id (str): The ID of the document for which to retrieve knowledge nodes.
        Returns:
            List[BaseNode]: A list of BaseNode objects representing the knowledge nodes
                            associated with the specified document.
        """
        #driver = self._driver
        database = self._database
        nodes:List[BaseNode] = []
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
            records = client.execute(
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
        Retrieve all nodes of a specific type from the knowledge graph.
        Args:
            node_type (str): The type of nodes to retrieve.
        Returns:
            List[TextNode]: A list of TextNode objects representing the nodes of the specified type.
        """
        nodes:List[TextNode] = []
        #driver = self._driver
        database = self._database
        with self._connection_pool.session_context(self._kgUser, self._kgPass) as client:
                records  = client.execute(
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