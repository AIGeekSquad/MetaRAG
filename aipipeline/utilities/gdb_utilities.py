# Description: This file contains the implementation of document graph store of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.knowledge_graph.neo4j.neo4j_storage import Neo4JStorage
# from knowledge_graph.nebulagraph.nebulagraph_storage import NebulaGraphStorage
from aipipeline.utilities.constants import GRAPHDB_TYPE, IngestionEnvs

def get_document_graph_store(graphDB: GRAPHDB_TYPE) -> DocumentGraphBaseComponent:
    """
    Retrieves the appropriate document graph store based on the provided graph database type.
    Args:
        graphDB (GRAPHDB_TYPE): The type of graph database to use.
    Returns:
        DocumentGraphBaseComponent: An instance of the document graph store corresponding to the specified graph database type.
        Returns None if the graph database type is not supported.
    Supported graph database types:
        - GRAPHDB_TYPE.NEO4J: Uses Neo4JStorage with the necessary environment configurations.
        # - GRAPHDB_TYPE.NEBULA: (Commented out) Uses NebulaGraphStorage with the necessary environment configurations.
    """
    match graphDB.value:
        case GRAPHDB_TYPE.NEO4J:
            neo4j_host_uri = IngestionEnvs.NEO4J_HOST
            AUTH = IngestionEnvs.NEO4J_AUTH
            neo4jdb = IngestionEnvs.NEO4J_DB
            kg_doc_store: DocumentGraphBaseComponent = Neo4JStorage(neo4j_host_uri, AUTH, neo4jdb)
            return kg_doc_store
        # case  GRAPHDB_TYPE.NEBULA:
        #     nebula_host_uri = IngestionEnvs.NEBULA_HOST
        #     nebula_user = IngestionEnvs.NEBULA_USER
        #     nebular_pass = IngestionEnvs.NEBULA_PASSWORD
        #     kg_doc_store: DocumentGraphBaseComponent = NebulaGraphStorage(nebula_host_uri, nebula_user, nebular_pass)
        #     return kg_doc_store  
        case _:
            return None 