# Description: This file contains the implementation for the Web Loader component of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.data_loaders.loaders_base_component import DataLoaderBaseComponent
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.node_transformers.knowledge_generator import ApplyKnowledgeLabel, KnowledgeGenerator

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.node_parser import ( SemanticSplitterNodeParser)
from llama_index.core.schema import Document,BaseNode
from llama_index.readers.web import ReadabilityWebPageReader
from typing import List, Optional

from llama_index.core.utils import print_text
import logging

logger = logging.getLogger(__name__)

class WebLoader(DataLoaderBaseComponent):
    """
    WebLoader is a component responsible for loading and processing web content into a document graph.
    Attributes:
        _embed_model (BaseEmbedding): The embedding model used for semantic splitting.
        _llm (LLM): The language model used for generating knowledge.
        _kg_doc_store (DocumentGraphBaseComponent): The document graph store where processed nodes are stored.
        _verbose (bool): Flag to enable verbose logging.
    Methods:
        __init__(kg_doc_store, llm, embed_model, verbose):
            Initializes the WebLoader with the given document store, language model, and embedding model.
        load_data_content(**kwargs) -> List[BaseNode]:
            Loads data content from web URLs provided in kwargs and processes them into nodes.
        load_web_content(web_urls: List[str], breakpoint_percentile_threshold=80) -> List[BaseNode]:
            Loads and processes web content from the given URLs, splitting and generating knowledge nodes.
        _load_web_pages(web_urls: List[str]) -> List[Document]:
            Loads web pages from the given URLs using the ReadabilityWebPageReader.
        _store_document(document: Document):
            Stores a single document in the document graph store.
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
            embed_model: Optional[BaseEmbedding] = None,
            verbose: bool = False,  
    ):
        self._kg_doc_store = kg_doc_store          
        self._embed_model = resolve_embed_model(embed_model)
        self._llm = resolve_llm(llm)
        
        if self._kg_doc_store is None:
            raise ValueError("Document store is required")
    
        self._verbose = verbose
        if self._verbose:
            print_text(text="Web Loader Initialized", color="llama_blue", end="\n")

    def load_data_content(self, **kwargs) -> List[BaseNode]:
        """
        Loads data content from web URLs.
        This method processes web files by loading their content from the provided URLs.
        Args:
            **kwargs: Arbitrary keyword arguments. Expects 'web_urls' to be a list of URLs.
        Returns:
            List[BaseNode]: A list of BaseNode objects containing the loaded web content.
        """
        logger.info("processing Web files")
        urls = kwargs.get("web_urls")
        

        results = self.load_web_content(web_urls=urls)

        return results

    def load_web_content(self, web_urls:List[str], breakpoint_percentile_threshold = 80) -> List[BaseNode]:
        """
        Load and process web content from the given URLs.
        This method fetches web pages from the provided URLs, processes them through a pipeline of transformations,
        and stores the resulting nodes in a knowledge graph document store.
        Args:
            web_urls (List[str]): A list of web page URLs to load and process.
            breakpoint_percentile_threshold (int, optional): The percentile threshold for the semantic splitter. Defaults to 80.
        Returns:
            List[BaseNode]: A list of processed nodes stored in the knowledge graph document store.
        """

        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=self._embed_model)       
        chunkLabel = ApplyKnowledgeLabel(knowledge_type="Chunk", override=True)
        knwoledgeGenerator = KnowledgeGenerator(llm=self._llm, generate_summary=True, generate_takeaways=True, generate_event_list=True, generate_reference_list=True)

        # create the pipeline with transformations
        pipeline = IngestionPipeline( transformations=[ splitter, chunkLabel, knwoledgeGenerator ])
        documents = self._load_web_pages(web_urls)
 
        nodes_stored = []
        for doc in documents:
            logger.info(f"Processing page {doc.metadata['url']}")
            self._store_document(doc)
            # run the pipeline
            try:
                nodes = pipeline.run(documents=[doc])                           
                nodes_stored.extend(nodes)
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata['url']} {e}")
                continue

        logger.info(f"Split into {len(nodes_stored)} nodes") 
        if len(nodes_stored) > 0:                
            self._kg_doc_store.store_nodes(nodes_stored)           
            
        return nodes_stored
    
    def _load_web_pages(self, web_urls: List[str]) -> List[Document]:
        #ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
        docs = []
        loader = ReadabilityWebPageReader(wait_until="networkidle")
        
        for url_web in web_urls:
            documents = loader.load_data(
                url=url_web
            )
            for doc in documents:
                doc.metadata["url"] = url_web
                doc.metadata["sourceType"] = "website"
            docs.extend(documents)
            
        return docs
    
    def _store_document(self, document: Document):   
        self._kg_doc_store.store_document(document)

  ## function implementations from abstract class - Not implemented
    def store_document(self, **kwargs) -> bool:

        raise NotImplementedError("Functionality not Implemented")
        #return False
    
    def load_documents_from_data(self, **kwargs) -> bool:

        raise NotImplementedError("Functionality not Implemented")
        #return False
