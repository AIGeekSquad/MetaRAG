# Description: This file contains the implementation for the PDF Loader component of the AI Pipeline.
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
from llama_index.core.schema import Document, NodeRelationship, BaseNode, ImageDocument
from llama_index.core.utils import print_text

from pathlib import Path
from PIL import Image
from typing import List, Optional, Union, IO, cast
import fitz

import logging

logger = logging.getLogger(__name__)


class PdfLoader(DataLoaderBaseComponent):
    """
    PdfLoader is a class responsible for loading and processing PDF files into a document graph.
    Attributes:
        _embed_model (BaseEmbedding): The embedding model used for semantic splitting.
        _llm (LLM): The language model used for generating knowledge.
        _multi_modal_llm (MultiModalLLM): The multi-modal language model for processing images.
        _kg_doc_store (DocumentGraphBaseComponent): The document graph store.
        _verbose (bool): Flag to enable verbose logging.
    Methods:
        __init__(kg_doc_store, llm, multi_modal_llm, embed_model, verbose):
            Initializes the PdfLoader with the given parameters.
        load_data_content(**kwargs) -> List[BaseNode]:
            Loads data content from the provided PDF file path or URI.
        _load_file(file, file_name_override, uri, breakpoint_percentile_threshold) -> List[BaseNode]:
            Loads and processes the PDF file, extracting documents and storing them.
        _parse_document(documents, breakpoint_percentile_threshold) -> List[BaseNode]:
            Parses the documents using a semantic splitter and generates knowledge nodes.
        _store_document(document):
            Stores a single document in the document graph store.
        _load_documents_from_pdf(file, file_name_override, uri) -> List[Document]:
            Loads documents from a PDF file, extracting text and images.
        store_document(**kwargs) -> bool:
            Abstract method to store a document (not implemented).
        load_documents_from_data(**kwargs) -> bool:
            Abstract method to load documents from data (not implemented).
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
            raise ValueError("Document store is required")

        if self._verbose:
            print_text(text="PDF Loader Initialized", color="llama_blue")
            if self._multi_modal_llm is not None:
                print_text(text="Multi Modal LLM Initialized", color="green", end="\n")

    # overriding abstract method from base 
    def load_data_content(self, **kwargs) -> List[BaseNode]:
        """
        Load data content from a PDF file.
        This method processes PDF files and loads their content into a list of BaseNode objects.
        Args:
            **kwargs: Arbitrary keyword arguments.
                - file_path (str): The path to the PDF file.
                - filename_override (str, optional): An optional filename to override the original.
                - uri (str, optional): An optional URI for the file.
        Returns:
            List[BaseNode]: A list of BaseNode objects containing the content of the PDF file.
        """
        if self._verbose:
            print_text(text="processing PDF files", color="llama_blue", end="\n")   

        file_or_path = kwargs.get("file_path")
        filename_override = kwargs.get("filename_override")
        uri = kwargs.get("uri")

        results = self._load_file(file=file_or_path, file_name_override=filename_override, uri=uri)

        return results

    def _load_file(self, file: Union[IO[bytes], str, Path], file_name_override: Optional[str] = None, uri:Optional[str] = None, breakpoint_percentile_threshold = 80) -> List[BaseNode]: 
        documents = self._load_documents_from_pdf(file, file_name_override=file_name_override, uri=uri)
        if self._verbose:
            print_text(text=f"Generated {len(documents)} documents from pdf file", color="llama_blue", end="\n")  
       

        # store all docs    
        for doc in documents:
            self._store_document(doc)
        # process the documents and lables
        for doc in documents:
            pageLabel = doc.metadata.get("pageLabel")
            sourceType = doc.metadata.get("sourceType")
            if pageLabel is None:
                pageLabel = ""
            if sourceType is None:
                sourceType = ""

            if doc.source_node is not None:   
                self._kg_doc_store.define_relationship(doc, doc.source_node, "HAS_SOURCE", {"pageLabel": pageLabel})            
            
            if(doc.class_name() == ImageDocument.class_name()):
                self._kg_doc_store.apply_label([doc], "Image")
                logger.info(f"Applying Image label to image document {doc.id_}")
            elif sourceType == "ImageDescription":
                self._kg_doc_store.apply_label([doc], "ImageDescription")
                logger.info(f"Applying Image description label to image document {doc.id_}")
            elif pageLabel != "":
                self._kg_doc_store.apply_label([doc], "PdfPage")

  
                
        return self._parse_document(documents, breakpoint_percentile_threshold)


    def _parse_document(self, documents: List[Document], breakpoint_percentile_threshold) -> List[BaseNode]:
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=self._embed_model)   
        chunkLabel = ApplyKnowledgeLabel(knowledge_type="Chunk", override=True)
        knwoledgeGenerator = KnowledgeGenerator(llm=self._llm, generate_summary=True, generate_takeaways=True, generate_event_list=True, generate_reference_list=True)    
       
        # create the pipeline with transformations
        pipeline = IngestionPipeline( transformations=[ splitter, chunkLabel, knwoledgeGenerator])
        nodes_stored = []
        for doc in documents:
            nodesToStore: List[BaseNode] = []
            documentType = doc.class_name()
            if documentType == "ImageDocument":
                continue
            try:
                nodes = pipeline.run(documents=[doc])
                
                for node in nodes:
                    nodeType = node.class_name()
                    if nodeType == "TextNode":
                        context = node.get_content()
                        if context is not None  and len(context) > 0:
                            nodesToStore.append(node)   
            
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata['fileName']}, {e}")
                continue             
               
            if self._verbose:
                print_text(text=f"Split into {len(nodesToStore)} nodes", color="llama_blue", end="\n")    
            
            if len(nodesToStore) > 0:
                nodes_stored.extend(nodesToStore)
                self._kg_doc_store.store_nodes(nodesToStore)

            
        return nodes_stored
        
    def _store_document(self, document: Document):   
        self._kg_doc_store.store_document(document)
        self._kg_doc_store.apply_label([document], "PdfFile")
        
    def _load_documents_from_pdf(self, file: Union[IO[bytes], str, Path], file_name_override: Optional[str] = None,uri:Optional[str] = None) -> List[Document]:

        import pypdf
         # Check if the file is already a Path object, if not, create a Path object from the string
        if not isinstance(file, Path) and isinstance(file, str):
            file = Path(file)
        pageCount = 0
        rootDocument = Document(metadata={"sourceType": "pdf", "fileUri": uri})
        # common metadata
        metadata = {"sourceType": "pdf"}
        pdf_document = None
        # Open the file if it's not already open, else use it as it is
        if isinstance(file, Path):
            context = open(file, "rb")
            pdf_document = fitz.open(file, filetype="pdf")
            rootDocument.metadata.update({"fileName": file.name})
            metadata.update({"fileName": file.name, "fileUri": file.as_uri()})
            if self._verbose:
                print_text(text=f"Loading file {file.name}", color="llama_blue", end="\n")       
        else:
            if self._verbose:
                print_text(text=f"Loading pdf from raw bytes", color="llama_blue", end="\n") 

            pdf_document = fitz.open(stream=file, filetype="pdf")
            context = file
        if file_name_override is not None:
             rootDocument.metadata.update({"fileName": file_name_override})

        # store file and keep uri   
        docs = [rootDocument]
        with context as fp:
        # Create a PDF object
            try:
                pdf = pypdf.PdfReader(fp)                
                pageCount = len(pdf.pages)

                # Iterate over every page
                
                for page in range(pageCount):  
                    if self._verbose:
                        print_text(text=f"Processing page {page} of {pageCount}", color="llama_blue", end="\n")  
                    
                    page_label = pdf.page_labels[page]

                    # Extract the text from the page
                    page_text = pdf.pages[page].extract_text()

                    images = None
                    if self._multi_modal_llm is not None:
                        # Extract the images from the page
                        images = []
                        try:
                            imageObjects = pdf.pages[page].images
                            for imageObject in imageObjects:
                                image = cast(Image.Image, imageObject.image)
                                
                                if image.height < 50 or image.width < 50:
                                    continue
                                if image.mode != "RGB":
                                    image = image.convert("RGB")
                                    
                                images.append(image)


                        except Exception as e:
                            logger.error(f"Error extracting images from page {page} {e}")
                            logger.error(e)

                        if len(images) > 0:
                            if self._verbose:
                                print_text(text=f"Extracted {len(images)} images from page {page}", color="llama_blue", end="\n")  
                          
                        ## Remove 
                        # Render the entire page as an image
                    #    """  if self._verbose:
                    #         print_text(text=f"Rendering page {page} as an image", color="llama_blue", end="\n")  
                
                    #     imagesSrc = pdf_document[page]
                    #     pix = imagesSrc.get_pixmap()
                    #     image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    #     metadata.update({"pageLabel": page_label, "sourceType": "pageImage"})

                    #     imageDocs = create_image_document(image, rootDocument, self._multi_modal_llm, metadata)
                    #     docs.extend(imageDocs) """

                    # If the page is empty or there are no images skip it
                    if (page_text is None or len(page_text.strip()) == 0) and ( images is None or len(images) == 0):
                        continue

                    if page_text is None:
                        page_text = ""
                                      
                    metadata.update({"pageLabel": page_label, "sourceType": "pageText"})

                    page_doc = Document(text=page_text, metadata=metadata)
                    page_doc.relationships[NodeRelationship.SOURCE] = rootDocument.as_related_node_info()                
                    
                    docs.append(page_doc)

                    if self._multi_modal_llm is not None and images is not None and len(images) > 0:
                        if self._verbose:
                            print_text(text=f"Processing {len(images)} images", color="llama_blue", end="\n")
                        
                        image_count = 1
                        for image in images:
                            image_count+=1

                            # Skip images that are too small
                            if image is None:
                                continue
                            if image.height < 50 or image.width < 50:
                                continue
                            
                            metadata.update({"pageLabel": page_label, "imageIndex": image_count})
      
                            imageDocs = create_image_document(image, page_doc, self._multi_modal_llm, metadata)
                            docs.extend(imageDocs)
                    
            except Exception as e:
                logger.error(f"Error processing pdf {file.name} {e}")
                pass
        if self._verbose:
                print_text(text=f"Processed {len(docs)} documents from {pageCount} pages of pdf {file.name}", color="llama_blue", end="\n")       

        return docs
    
    ## function implementations from abstract class - Not implemented
    def store_document(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
    
    def load_documents_from_data(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
