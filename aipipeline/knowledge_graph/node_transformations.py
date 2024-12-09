# Description: This file contains the implementation of Node transformation and creation for Knowledge Graph of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from llama_index.core.schema import TextNode, NodeRelationship, BaseNode, ImageDocument, NodeRelationship, Document
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.llms.llm import LLM
from typing import Optional, List, Dict
from PIL import Image
import re

import base64
from io import BytesIO

import logging


logger = logging.getLogger(__name__)

def set_knowledge_type(node: BaseNode, knowledge_type: str, override:bool = False) -> BaseNode:
    """
    Sets the knowledge type of a given node.
    Args:
        node (BaseNode): The node to set the knowledge type for.
        knowledge_type (str): The knowledge type to set.
        override (bool, optional): If True, overrides the existing knowledge type. Defaults to False.
    Returns:
        BaseNode: The node with the updated knowledge type.
    """

    if (node.metadata is None ):
        node.metadata = {}       
    if "knowledgeType" not in node.metadata or override:         
        node.metadata["knowledgeType"]  = knowledge_type

    return node

def generate_event_list_node(source:BaseNode, llm: LLM) -> Optional[BaseNode]:
    """
    Generates a new TextNode containing a list of events extracted from the content of the given source TextNode.
    Args:
        source (BaseNode): The source node from which to extract events. Must be an instance of TextNode.
        llm (LLM): The language model used to generate the list of events.
    Returns:
        Optional[BaseNode]: A new TextNode containing the list of events if successful, or None if no events are found or an error occurs.
    The function performs the following steps:
    1. Checks if the source node is an instance of TextNode. If not, returns None.
    2. Retrieves the content of the source node.
    3. If the content is empty, returns None.
    4. Constructs a prompt to instruct the language model to generate a list of events from the text.
    5. Calls the language model to complete the prompt and generate the event list.
    6. If an error occurs during the language model call, logs the error and returns None.
    7. Checks if the generated event list contains "NODATA". If so, returns None.
    8. Logs the extracted events.
    9. Creates a new TextNode with the generated event list.
    10. Sets the relationships and knowledge type for the new node.
    11. Updates the relationships of the source node to include the new node as a child.
    12. Returns the new TextNode containing the event list.
    """
    if not isinstance(source, TextNode):
        return None
    
    text = source.get_content()  

    if (len(text) < 1):
        return None

    prompt = f"""
                    Generate a list of events from the following text. Report events in chronological order. return only the list of events, any other information will be ignored and can generate weird results.
                    Only report events that have a complete date or a sequence time associated with them, if there is no date or time, ignore the event.
                    Do not report sequences from ordered lists or index pages.

                    Reference to articles or books should be ignored.
                    Do not report any irrelevant information. Report only events that are relevant from the text.
                    Do not report any information that is not an event or an incomplete event.

                    Do not report duplicated entries, make sure to only report each event once.
                
                    If there is no events in the text, return "NODATA" and nothing else.
                    

                    Text:
                    {text}
                    """
    event_list = None
    try:
        event_list = llm.complete(prompt).text       
    except Exception as e:
        logger.error(f"Error generating event list - {e}")
        return None

    if contains_nodata(event_list):
        return None
    
    logger.info(f"Extracted Events: {event_list}")
    event_list_node = TextNode(text=event_list)
    event_list_node.relationships[NodeRelationship.PARENT] = source.as_related_node_info()
    if (source.source_node is not None) :
        event_list_node.relationships[NodeRelationship.SOURCE] = source.source_node

    set_knowledge_type(node=event_list_node, knowledge_type="EventList", override=True)

    event_list_node.start_char_idx = source.start_char_idx
    event_list_node.end_char_idx = source.end_char_idx
    child_nodes = source.relationships.get(NodeRelationship.CHILD, [])
    child_nodes.append(event_list_node.as_related_node_info())
    source.relationships[NodeRelationship.CHILD] = child_nodes
    return event_list_node

def generate_reference_material_node(source:BaseNode, llm: LLM) -> Optional[BaseNode]:
    """
    Generates a reference material node from the given source node using a language model (LLM).
    Args:
        source (BaseNode): The source node from which to generate the reference material. 
                            Must be an instance of TextNode.
        llm (LLM): The language model used to generate the reference list.
    Returns:
        Optional[BaseNode]: A new TextNode containing the list of references if successful, 
                            otherwise None.
    The function performs the following steps:
    1. Checks if the source node is an instance of TextNode. If not, returns None.
    2. Extracts the text content from the source node.
    3. If the text content is empty, returns None.
    4. Constructs a prompt to generate a list of references from the text.
    5. Uses the language model to complete the prompt and generate the reference list.
    6. If an error occurs during the generation, logs the error and returns None.
    7. Checks if the generated reference list contains "NODATA". If so, returns None.
    8. Creates a new TextNode with the generated reference list.
    9. Sets the relationships and knowledge type for the new node.
    10. Updates the source node's relationships to include the new reference list node.
    11. Returns the new reference list node.
    """
    if not isinstance(source, TextNode):
        return None
    
    text = source.get_content()  

    if (len(text) < 1):
        return None

    prompt = f"""
                    Generate a list of references from the following text. Report references in alphabetical order. 
                    Return only the list of references, any other information will be ignored and can generate weird results.
                    Reference material can be book references, atricle references, or any other type of reference material.
                    Mentioning movies, art, music albums, or other types of references that are mentioned as source material should be included.
                    Make sure to retain website links, book titles, article titles, and other types of references.

                    Do not generate trivia or other types of references that are not source material, this will make the results weird.
                    Do not report duplicated entries, make sure to only report each reference once.
                
                    If there is no references in the text, return only "NODATA" and nothing else.
                    

                    Text:
                    {text}
                    """
    
    reference_list = None
    try :
        reference_list = llm.complete(prompt).text        
    except Exception as e:
        logger.error(f"Error generating reference material - {e}")
        return None

    if contains_nodata(reference_list):
        return None
    logger.info(f"Extracted References: {reference_list}")
    reference_list_node = TextNode(text=reference_list)
    reference_list_node.relationships[NodeRelationship.PARENT] = source.as_related_node_info()
    if (source.source_node is not None) :
        reference_list_node.relationships[NodeRelationship.SOURCE] = source.source_node

    set_knowledge_type(node=reference_list_node, knowledge_type="ReferenceList", override=True)

    reference_list_node.start_char_idx = source.start_char_idx
    reference_list_node.end_char_idx = source.end_char_idx
    child_nodes = source.relationships.get(NodeRelationship.CHILD, [])
    child_nodes.append(reference_list_node.as_related_node_info())
    source.relationships[NodeRelationship.CHILD] = child_nodes
    return reference_list_node

def generate_takeaway_node(source:BaseNode, llm: LLM) -> Optional[BaseNode]:
    if not isinstance(source, TextNode):
        return None

    text = source.get_content() 

    if (len(text) < 1):
        return None
    
    prompt = f"""
                    Generate a succint list with the most important take aways from the following text. 
                    Emit only the list, any other information will be ignored and can generate weird results.
                    Do not report sequences from ordered lists on index pages. Don't report any irrelevant information.
                    Do not report any information that is not a takeaway.

                    Titles, subtitles, page numbers and other non-takeaway information should be ignored.

                    If there is no takeaways to be generated from the text, return only  "NODATA" and nothing else.

                    Text:
                    {text}
                    """
    
    takeaways = None
    try:
        takeaways = llm.complete(prompt).text           
    except Exception as e:
        logger.error(f"Error generating takeaways - {e}")
        return None

    if contains_nodata(takeaways):
        return None

    logger.info(f"Etracted Takeaways: {takeaways}") 
    takeaway_node = TextNode(text=takeaways)
    takeaway_node.relationships[NodeRelationship.PARENT] = source.as_related_node_info()
    if (source.source_node is not None) :
        takeaway_node.relationships[NodeRelationship.SOURCE] = source.source_node

    set_knowledge_type(node= takeaway_node, knowledge_type= "Takeaways", override=True)
    
    takeaway_node.start_char_idx = source.start_char_idx
    takeaway_node.end_char_idx = source.end_char_idx
    child_nodes = source.relationships.get(NodeRelationship.CHILD, [])
    child_nodes.append(takeaway_node.as_related_node_info())
    source.relationships[NodeRelationship.CHILD] = child_nodes
    return takeaway_node

def generate_summary_node(source:BaseNode, llm: LLM) -> Optional[BaseNode]:
    """
    Generates a summary node from a given source node using a language model (LLM).
    Args:
        source (BaseNode): The source node containing the text to be summarized. Must be an instance of TextNode.
        llm (LLM): The language model used to generate the summary.
    Returns:
        Optional[BaseNode]: A new TextNode containing the summary if successful, otherwise None.
    The function performs the following steps:
    1. Checks if the source node is an instance of TextNode. If not, returns None.
    2. Retrieves the content of the source node.
    3. If the content is empty, returns None.
    4. Constructs a prompt for the language model to generate a summary.
    5. Attempts to generate the summary using the language model. If an error occurs, logs the error and returns None.
    6. Checks if the generated summary contains "NODATA" or is empty. If so, returns None.
    7. Ensures the summary length is not excessively long compared to the original text. If it is, returns None.
    8. Creates a new TextNode with the generated summary and establishes relationships with the source node.
    9. Sets the knowledge type of the summary node to "Summary".
    10. Copies character index information from the source node to the summary node.
    11. Adds the summary node as a child of the source node.
    12. Returns the summary node.
    """
    if not isinstance(source, TextNode):
        return None
    
    text = source.get_content()  

    if (len(text) < 1):
        return None
    
    prompt = f"""
                    Summarise the following paragraph focusing on most important parts. Be concise and clear.
                    Make sure to include the most important parts of the text and to write in your own words.
                    If the text contains data, or references to data, make sure to include the most important data.
                    Don't include any irrelevant information.
                    Keep all relevant numbers, informations, dates, locations and facts.
                    If the text is not appropriate do not generate a summary.
                    Always write numbers as digits, not words.
                    Expand acronyms and abbreviations.

                    Do not sumamrise title pages.

                    If there is no relevant information in the text, return only "NODATA" and nothing else.

                    Text:
                    {text}
                    """
    
    summary = None
    try:
        summary = llm.complete(prompt).text
    except Exception as e:
        logger.error(f"Error generating summary - {e}")
        return None 
    
    if contains_nodata(summary):
        return None
    
    if len(summary) < 1:
        return None
    
    ratio = len(summary) / len(text)
    if ratio > 0.8:
        return None
    logger.info(f"Extracted Summary: {summary}")
    summary_node = TextNode(text=summary)
    summary_node.relationships[NodeRelationship.PARENT] = source.as_related_node_info()
    if (source.source_node is not None) :
        summary_node.relationships[NodeRelationship.SOURCE] = source.source_node

    set_knowledge_type(node=summary_node, knowledge_type= "Summary", override=True)
    
    summary_node.start_char_idx = source.start_char_idx
    summary_node.end_char_idx = source.end_char_idx
    child_nodes = source.relationships.get(NodeRelationship.CHILD, [])
    child_nodes.append(summary_node.as_related_node_info())
    source.relationships[NodeRelationship.CHILD] = child_nodes
    return summary_node


def extract_text_from_image(base64_encoded_image: str, multi_modal_llm : MultiModalLLM) -> str:
    """
    Extracts text and relevant information from a base64 encoded image using a multi-modal language model.
    Args:
        base64_encoded_image (str): The base64 encoded image string.
        multi_modal_llm (MultiModalLLM): An instance of a multi-modal language model capable of processing images and text.
    Returns:
        str: The extracted text or relevant information from the image. If no relevant information is found, returns "NODATA".    
    """
    logger.info(f"Extracting text from image")
    img_doc = ImageDocument(image=base64_encoded_image)
    image_prompt = """
Plese extract information from the image.
Describe the image and extract any relevant information from it.
if the image contains text, please extract the text from the image and provide a summary of the text.
if the image contains a chart, please describe the chart and provide a summary of the chart.
if the image contains a table, please describe the table and provide a summary of the table.

if the document is some form of tutorial, please provide a summary of the tutorial and make sure to describe parts of the tutorial that are relevant to the user.

If there is no relevant information in the image, nothing to report just write "NODATA" and nothing else.
"""
    response = multi_modal_llm.complete(
    prompt=image_prompt,
    image_documents=[img_doc],
    )
    processed_text = response.text.strip()
    return processed_text

def scale_image_for_llm(image: Image.Image) -> Image.Image:
    """
    Scales an image to ensure its byte size does not exceed a specified limit.
    Args:
        image (Image.Image): The input image to be scaled.
    Returns:
        Image.Image: The scaled image, ensuring it is in RGB mode and within the byte size limit.
    """
    # setting a limit to 15 MB for the image
    max_image_byte_size = 1024 * 1024 * 10 
    # convert image to RGB if not already in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")        

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    bytes = buffered.getvalue()
    size = len(bytes)
    if size > max_image_byte_size:
        # image too big, we resize it
        ratio = max_image_byte_size / len(bytes)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)))

    return image

def create_image_document(image: Image.Image, rootDocument: Document, multi_modal_llm : MultiModalLLM, extra_info: Dict[str, str]) -> List[Document]:
    """
    Creates image and text documents from an input image.
    This function processes an input image, extracts text from it using a multi-modal language model (LLM),
    and creates two documents: one for the image and one for the extracted text. The documents are linked
    with relationships and metadata.
    Args:
        image (Image.Image): The input image to be processed.
        rootDocument (Document): The root document to which the created documents will be related.
        multi_modal_llm (MultiModalLLM): The multi-modal language model used for text extraction.
        extra_info (Dict[str, str]): Additional metadata to be included in the documents.
    Returns:
        List[Document]: A list containing the created image and text documents. If an error occurs, an empty list is returned.
    """

    docs = []
    img_str : str = None
    logger.info(f"Creating image document")
    try:       
        
        # convert image to RGB if not already in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")      
      
        image = scale_image_for_llm(image)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        bytes = buffered.getvalue()
        

        logger.info(f"Image size: {len(bytes)}")
        # convert image to base64
        img_str = base64.b64encode(bytes)     
    except Exception as e:
        logger.error(f"Error creating image document generating image string - {e}")
        return []
    
    try:
        processed_text = ""   

        logger.info(f"Extracting text from image")
        processed_text = extract_text_from_image(img_str, multi_modal_llm)

        if contains_nodata(processed_text):
            return []
        
        metadata = {}

        if extra_info is not None:
            metadata.update(extra_info)

        logger.info(f"Creating image document")
        image_doc = ImageDocument(image=img_str, text=img_str, metadata=metadata.copy())
        image_doc.metadata.update({"sourceType": "EmbeddedImage"})
        image_doc.relationships[NodeRelationship.SOURCE] = rootDocument.as_related_node_info()          

        logger.info(f"Creating text document from image description text")
        metadata.update({"sourceType": "ImageDescription"})
        textNode = Document(text=processed_text, metadata=metadata.copy())
        textNode.metadata.update({"sourceType": "ImageDescription"})
        textNode.relationships[NodeRelationship.SOURCE] = image_doc.as_related_node_info()

        docs.append(image_doc)
        docs.append(textNode)
    except Exception as e:
        logger.error(f"Error creating image document generating image description - {e}")
        pass

    return docs

def contains_nodata(string):
    """
    Checks if the given string contains the substring 'NODATA' (case-insensitive).
    Args:
        string (str): The string to be checked.
    Returns:
        bool: True if the string contains 'NODATA', False otherwise.
    """

    pattern = re.compile(r'NODATA', re.IGNORECASE)
    if pattern.search(string):
        return True
    else:
        return False