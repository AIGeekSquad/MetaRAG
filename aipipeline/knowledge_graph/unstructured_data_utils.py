# Description: This file contains the implementation of Node utilites  for Knowledge Graph of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import json
import re
from typing import cast, List
from aipipeline.knowledge_graph.kg_types import KnowledgeGraphNode, KnowledgeGraphRelationship

regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s+(.*)"
internalRegex = "\[(.*?)\]"
jsonRegex = "\{.*\}"


def nodesTextToListOfKnowledgeGraphNodes(nodes: List[str]) -> List[KnowledgeGraphNode]:
    """
    Converts a list of node text representations into a list of KnowledgeGraphNode objects.
    Each node in the input list is expected to be a string with the format:
    "name,label,{properties}", where properties is a JSON-like string.
    Args:
        nodes (List[str]): A list of strings, each representing a node in the knowledge graph.
    Returns:
        List[KnowledgeGraphNode]: A list of KnowledgeGraphNode objects created from the input strings.
    Example:
        nodes = [
            'Node1,Label1,{"key1": "value1", "key2": "value2"}',
            'Node2,Label2,{"key3": "value3"}'
        ]
        result = nodesTextToListOfKnowledgeGraphNodes(nodes)
    """
    result:List[KnowledgeGraphNode] = []
    for node in nodes:
        nodeList = node.split(",")
        if len(nodeList) < 2:
            continue

        name = nodeList[0].strip().replace('"', "")
        label = nodeList[1].strip().replace('"', "")
        properties = re.search(jsonRegex, node)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append(KnowledgeGraphNode(name= name, label=label, properties= properties))
    return result


def relationshipTextToListOfKnowledgeGraphRelationship(relationships: List[str]) -> List[KnowledgeGraphRelationship]:
    """
    Converts a list of relationship strings into a list of KnowledgeGraphRelationship objects.
    Each relationship string is expected to be in the format:
    'start, type, end, {optional_properties}'
    where 'start', 'type', and 'end' are strings representing the nodes and relationship type,
    and 'optional_properties' is a JSON string representing additional properties of the relationship.
    Args:
        relationships (List[str]): A list of relationship strings.
    Returns:
        List[KnowledgeGraphRelationship]: A list of KnowledgeGraphRelationship objects.
    """
    result:List[KnowledgeGraphRelationship] = []
    for relation in relationships:
        relationList = relation.split(",")
        if len(relation) < 3:
            continue
        start = relationList[0].strip().replace('"', "")
        end = relationList[2].strip().replace('"', "")
        type = relationList[1].strip().replace('"', "")

        properties = re.search(jsonRegex, relation)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append(KnowledgeGraphRelationship( start= start, end =  end, type= type, properties = properties))
        
    return result
