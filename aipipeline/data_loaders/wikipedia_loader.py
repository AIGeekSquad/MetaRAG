# Description: This file contains the implementation for the Wikipedia Loader component of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.data_loaders.loaders_base_component import DataLoaderBaseComponent
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.knowledge_graph.node_transformations import create_image_document
from aipipeline.node_transformers.knowledge_generator import ApplyKnowledgeLabel, KnowledgeGenerator

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.node_parser import ( SemanticSplitterNodeParser)
from llama_index.core.schema import Document, BaseNode, ImageDocument, NodeRelationship
from PIL import Image
from typing import List, Optional
import io
import requests
import wikipedia
from llama_index.core.utils import print_text
import logging

logger = logging.getLogger(__name__)

class WikipediaLoader(DataLoaderBaseComponent):
    """
    WikipediaLoader is a data loader component that loads and processes Wikipedia pages.
    Attributes:
        _embed_model (BaseEmbedding): The embedding model used for semantic splitting.
        _llm (LLM): The language model used for generating knowledge.
        _multi_modal_llm (MultiModalLLM): The multi-modal language model for processing images.
        _kg_doc_store (DocumentGraphBaseComponent): The document graph store for storing and managing documents.
        _verbose (bool): Flag to enable verbose logging.
    Methods:
        __init__(kg_doc_store, llm, multi_modal_llm, embed_model, verbose):
            Initializes the WikipediaLoader with the provided components and settings.
        load_data_content(**kwargs) -> List[BaseNode]:
            Loads data content based on the provided topics and page count per topic.
        load_wikipedia_topics(topics, page_count_per_topic, breakpoint_percentile_threshold) -> List[BaseNode]:
            Loads Wikipedia topics and processes them through a pipeline of transformations.
        _store_document(document):
            Stores a document in the document graph store and applies appropriate labels.
        _load_documents_from_wikipedia_topics(topics, page_count_per_topic) -> List[Document]:
            Loads documents from Wikipedia based on the provided topics and page count per topic.
        _load_wikipedia_page(title) -> Document:
            Loads a single Wikipedia page based on the provided title.
        store_document(**kwargs) -> bool:
            Abstract method to store a document. Not implemented.
        load_documents_from_data(**kwargs) -> bool:
            Abstract method to load documents from data. Not implemented.
    """
    
    _embed_model:BaseEmbedding
    _llm:LLM

    def __init__(
            self,
            kg_doc_store: DocumentGraphBaseComponent,
            llm: Optional[LLM] = None,
            multi_modal_llm: Optional[MultiModalLLM] = None,
            embed_model: Optional[BaseEmbedding] = None,
            verbose: bool = False,
    ):
        self._kg_doc_store = kg_doc_store        
        self._embed_model = resolve_embed_model(embed_model)
        self._llm = resolve_llm(llm)
        self._multi_modal_llm = multi_modal_llm
        self._verbose = verbose

        if self._kg_doc_store is None:
            raise ValueError("Document Graph Store is required")

        if self._verbose:
            print_text(text="Wikipedia Loader Initialized", color="llama_blue", end="\n")
            if self._multi_modal_llm is not None:
                print_text(text="Multi Modal LLM Initialized", color="green", end="\n")

    # overriding abstract method from base 
    def load_data_content(self, **kwargs) -> List[BaseNode]:
        """
        Load data content from Wikipedia based on provided topics and page count per topic.
        Args:
            **kwargs: Arbitrary keyword arguments.
                - topics (List[str]): A list of topics to search for on Wikipedia.
                - max_page_per_topic (int): Maximum number of pages to retrieve per topic.
        Returns:
            List[BaseNode]: A list of BaseNode objects containing the loaded Wikipedia content.
        """

        logger.info("processing Wikipedia files")
        topic_list = kwargs.get("topics")
        topic_page_count = kwargs.get("max_page_per_topic")
        

        results = self.load_wikipedia_topics(topics=topic_list,page_count_per_topic=topic_page_count)

        return results
   
    def load_wikipedia_topics(self, topics: List[str], page_count_per_topic = 5,breakpoint_percentile_threshold = 80) -> List[BaseNode]:
        """
        Load and process Wikipedia topics.
        This method loads Wikipedia pages based on the provided topics, processes them through a pipeline, and stores the resulting nodes.
        Args:
            topics (List[str]): A list of topics to search for on Wikipedia.
            page_count_per_topic (int, optional): The maximum number of pages to load per topic. Defaults to 5.
            breakpoint_percentile_threshold (int, optional): The percentile threshold for the semantic splitter. Defaults to 80.
        Returns:
            List[BaseNode]: A list of processed nodes stored in the knowledge graph document store.
        """
        if self._verbose:
            print_text(text=f"Topics: {topics}, max pages per topic : {page_count_per_topic}\n", color="llama_blue", end="\n")

       
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=self._embed_model)       
        chunkLabel = ApplyKnowledgeLabel(knowledge_type="Chunk", override=True)
        knwoledgeGenerator = KnowledgeGenerator(llm=self._llm, generate_summary=True, generate_takeaways=True, generate_event_list=True, generate_reference_list=True)

        # create the pipeline with transformations
        pipeline = IngestionPipeline( transformations=[ splitter, chunkLabel, knwoledgeGenerator ])
        documents = self._load_documents_from_wikipedia_topics(topics, page_count_per_topic)
        page_titles = set()
        for doc in documents:
            title : str = doc.metadata['title']
            if title is not None:
                if title in page_titles:
                    continue
                else:
                    page_titles.add(doc.node_id)
        logger.info(f"Loaded {len(page_titles)} pages from wikipedia")
        nodes_stored = []
        # store all docs    
        for doc in documents:
            self._store_document(doc)

        for doc in documents:
            sourceType = doc.metadata.get("sourceType")
            if sourceType is None:
                sourceType = ""

            self._kg_doc_store.add_source_relationship(doc)                          

            if(isinstance(doc, ImageDocument)):
                self._kg_doc_store.apply_label([doc], "Image")
            elif sourceType == "ImageDescription":
                self._kg_doc_store.apply_label([doc], "ImageDescription")
                
        
            title : str = doc.metadata['title']
            if title is None or len(title.strip()) == 0:
                title = "Unknown"
            
            documentType = doc.class_name()
            if documentType == "ImageDocument":
                continue
            
            logger.info(f"Processing document for page {doc.metadata['title']}")
            
            # run the pipeline
            nodesToStore: List[BaseNode] = []
            try:
               
                nodes = pipeline.run(documents=[doc])                    
                for node in nodes:
                    nodeType = node.class_name()
                    if nodeType == "TextNode":
                        context = node.get_content()
                        if context is not None  and len(context) > 0:
                            nodesToStore.append(node)  
                                
            except Exception as e:
                 logger.error(f"Error processing document {title} - {e}")
                 continue                  
                    
            logger.info(f"Split into {len(nodesToStore)} nodes") 
                
            if len(nodesToStore) > 0:
                nodes_stored.extend(nodesToStore)
                self._kg_doc_store.store_nodes(nodesToStore)            
            

        return nodes_stored

    def _store_document(self, document: Document):   

        self._kg_doc_store.store_document(document)
        
        knowledgeType = document.metadata.get("knowledgeType")
        if knowledgeType is None:
            knowledgeType = ""

        if isinstance(document, ImageDocument):
            self._kg_doc_store.apply_label([document], "Image")
        elif knowledgeType == "ImageDescription":
            self._kg_doc_store.apply_label([document], "ImageDescription")
        else :
            self._kg_doc_store.apply_label([document], "Webpage")

    def _load_documents_from_wikipedia_topics(self, topics: List[str], page_count_per_topic = 30) -> List[Document]:
        pages = []
        for topic in topics:
            pages += wikipedia.search(topic, results=page_count_per_topic)

        documents = []
        visited = set()
        for page in pages:
            try:
                wikipedia.set_lang("en")
                wp = wikipedia.page(page)
                if(wp.url in visited):
                    continue
                visited.add(wp.url)
                page_content = wp.content
                extra_info = {"title": wp.title, "url": wp.url, "sourceType": "wikipedia"}
                doc = Document(text=page_content, extra_info=extra_info)       
                documents.append(doc)
                logger.info(f"Loading wikipedia page {wp.title}")
                if(self._multi_modal_llm is not None):
                    logger.info(f"Processing images from wikipedia page {wp.title}")
                    image_urls = wp.images
                    if image_urls is None:
                        continue
                    image_urls = list(filter(lambda x: x.endswith(".jpg") or x.endswith(".png"), image_urls))

                    if len(image_urls) > 0:
                        logger.info(f"Processing {len(image_urls)} images from wikipedia page {wp.title}")
                    for url in image_urls:
                        headers = {'user-agent': 'wikipedia_loader/1.0'}
                        r = requests.get(url, stream=True, headers=headers)
                        if r.status_code == 200:                                
                            image = Image.open(io.BytesIO(r.content))

                            # Skip small images                     
                            if image.width < 50 or image.height < 50:
                                continue
                            
                            logger.info(f"Processing image {url}")
                            metadata = {"imageUrl":url , "sourceType": "articleImage"}
                            metadata.update(extra_info)
                            image_doc = ImageDocument(url=url, extra_info=metadata)
                            image_doc.relationships[NodeRelationship.SOURCE] = doc.as_related_node_info()            
                            imageDocs = create_image_document(image, doc, self._multi_modal_llm, metadata)
                            documents.extend(imageDocs)

                    logger.info(f"Processed {len(image_urls)} images from wikipedia page {wp.title}")
                    logger.info(f"Loaded wikipedia page {wp.title}")
            except Exception as e:
                logger.error(f"Error loading wikipedia page {page} - {e}")
                pages.remove(page)
        
        return documents

    def _load_wikipedia_page(self, title: str) -> Document:
        page = wikipedia.page(title)
        return Document(
            title=page.title,
            text=page.content,
            url=page.url,
            source="wikipedia",
            extra_info={"sourceType": "Wikipedia", "url": page.url, "title": page.title},
        )
    
    ## function implementations from abstract class - Not implemented
    def store_document(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
    
    def load_documents_from_data(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
