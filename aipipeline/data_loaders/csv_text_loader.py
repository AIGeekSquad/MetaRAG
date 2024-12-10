# Description: This file contains the implementation for the CSV and Text Loader component of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from aipipeline.data_loaders.loaders_base_component import DataLoaderBaseComponent
from aipipeline.knowledge_graph.kg_base_component import DocumentGraphBaseComponent
from aipipeline.knowledge_graph.node_transformations import generate_summary_node
from aipipeline.node_transformers.knowledge_generator import ApplyKnowledgeLabel, KnowledgeGenerator

from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.node_parser import ( SentenceSplitter)
from llama_index.core.schema import Document,BaseNode
from typing import List, Optional
from llama_index.core.utils import print_text

import logging


logger = logging.getLogger(__name__)

import csv

class CsvAndTextLoader(DataLoaderBaseComponent):
    """
    CsvAndTextLoader is a data loader component that processes CSV and text files, 
    splits them into nodes, and stores them in a knowledge graph document store.
    Attributes:
        _embed_model (BaseEmbedding): The embedding model used for processing.
        _llm (LLM): The language model used for generating knowledge.
        _kg_doc_store (DocumentGraphBaseComponent): The document store for storing nodes.
        _verbose (bool): Flag to enable verbose logging.
    Methods:
        __init__(kg_doc_store, llm, embed_model, verbose):
            Initializes the CsvAndTextLoader with the given document store, language model, 
            embedding model, and verbosity flag.
        load_data_content(**kwargs) -> List[BaseNode]:
            Loads data content from CSV and text files and returns a list of nodes.
        load_csv_text_content(csv_rootdir, breakpoint_percentile_threshold) -> List[BaseNode]:
            Loads and processes CSV and text content from the specified directory, 
            splits it into nodes, and stores them in the document store.
        _load_csv_text_pages(csv_rootdir) -> List[Document]:
            Loads CSV and text pages from the specified directory and returns a list of documents.
        _store_document(document):
            Stores the given document in the document store and applies appropriate labels.
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
        self._verbose = verbose
        if self._kg_doc_store is None:
            raise ValueError("Document store is required")

    # overriding abstract method from base 
    def load_data_content(self, **kwargs) -> List[BaseNode]:
        """
        Load data content from CSV and text files.
        This method processes CSV and text files located in the specified directory
        and returns a list of BaseNode objects.
        Args:
            **kwargs: Arbitrary keyword arguments. Expected to contain:
                - csv_rootdir (str): The root directory containing the CSV and text files.
        Returns:
            List[BaseNode]: A list of BaseNode objects created from the CSV and text files.
        """

        if self._verbose:
                print_text(text="processing CSV and Text files", color="llama_blue", end="\n")   
        rootdir= kwargs.get("csv_rootdir")

        results = self.load_csv_text_content(rootdir)

        return results
    
     #TODO: Change this to take in input_files as a list of files: Done -- Validate
    def load_csv_text_content(self, csv_rootdir:str,breakpoint_percentile_threshold = 80) -> List[BaseNode]:
        """
        Load and process text content from CSV files located in the specified root directory.
        This method reads CSV files, processes their text content through a pipeline of transformations,
        and stores the resulting nodes. The pipeline includes splitting text into sentences, applying
        knowledge labels, and generating additional knowledge such as summaries and takeaways.
        Args:
            csv_rootdir (str): The root directory containing the CSV files to be processed.
            breakpoint_percentile_threshold (int, optional): The percentile threshold for breakpoints. Defaults to 80.
        Returns:
            List[BaseNode]: A list of processed nodes stored after running the pipeline.
        """

        splitter = SentenceSplitter(separator="\n")     
        chunkLabel = ApplyKnowledgeLabel(knowledge_type="Chunk", override=True)
        knowledgeGenerator = KnowledgeGenerator(llm=self._llm, generate_summary=True, generate_takeaways=True, generate_event_list=True, generate_reference_list=True)    
       
        # create the pipeline with transformations
        pipeline = IngestionPipeline( transformations=[ splitter, chunkLabel, knowledgeGenerator])
        documents = self._load_csv_text_pages(csv_rootdir)
 
        nodes_stored = []
        for doc in documents:
            self._store_document(doc)
            # run the pipeline
              
            nodesToStore: List[BaseNode] = []
            try:
                nodes = pipeline.run(documents=[doc])  
                for node in nodes:
                    if isinstance(node, TextNode) :
                        context = node.get_content()
                        if context is not None  and len(context) > 0:
                            nodesToStore.append(node)    

            except Exception as e:
                logger.error(f"Error processing document {doc.metadata['url']}")
                logger.error(e)
                continue                    
                
            if self._verbose:
                print_text(text=f"Split into {len(nodesToStore)} nodes", color="llama_blue", end="\n")            
            
            if len(nodesToStore) > 0:
                nodes_stored.extend(nodesToStore)
                self._kg_doc_store.store_nodes(nodesToStore)            

        return nodes_stored
    
   
    def _load_csv_text_pages(self, csv_rootdir: str) -> List[Document]:
        """
        Loads text pages from CSV files located in the specified root directory.
        Args:
            csv_rootdir (str): The root directory containing the CSV files.
        Returns:
            List[Document]: A list of Document objects loaded from the CSV files.
        """

        #csv_dir_reader = SimpleDirectoryReader(input_dir=csv_rootdir, recursive=True)
        csv_list = [csv_rootdir]
        csv_dir_reader = SimpleDirectoryReader(input_files=csv_list, recursive=True)
        docs = csv_dir_reader.load_data()
        
        return docs
    
    def _store_document(self, document: Document):   
        self._kg_doc_store.store_document(document)
        self._kg_doc_store.apply_label([document], "TextFile")      
        # todo: determine if the document is a csv file, we could store this also as sql / queriable data
        is_csv = False
        if is_csv:
            self._kg_doc_store.apply_label([document], "CsvFile")      

     ## function implementations from abstract class - Not implemented
    def store_document(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
    
    def load_documents_from_data(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
