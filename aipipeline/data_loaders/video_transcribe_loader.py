# Description: This file contains the implementation for the Video Loader component of the AI Pipeline.
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
from llama_index.core.schema import Document,BaseNode, TextNode
from llama_index.readers.file import VideoAudioReader
from llama_index.core.utils import print_text
from pathlib import Path
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

class VideoTranscribeLoader(DataLoaderBaseComponent):
    """
    VideoTranscribeLoader is a class responsible for loading and processing video files, transcribing their content, and storing the resulting nodes in a knowledge graph document store.
    Attributes:
        _embed_model (BaseEmbedding): The embedding model used for semantic splitting.
        _llm (LLM): The language model used for generating knowledge from the video content.
        _kg_doc_store (DocumentGraphBaseComponent): The document store where the processed nodes are stored.
        _verbose (bool): Flag to enable verbose logging.
    Methods:
        __init__(kg_doc_store, llm, embed_model, verbose):
            Initializes the VideoTranscribeLoader with the given document store, language model, embedding model, and verbosity flag.
        load_data_content(**kwargs) -> List[BaseNode]:
            Loads and processes video content based on the provided file paths in kwargs.
        load_video_content(video_file_paths, breakpoint_percentile_threshold) -> List[BaseNode]:
            Processes the video files, splits them into nodes, generates knowledge, and stores the nodes in the document store.
        _load_video_data(video_paths) -> List[Document]:
            Loads video data from the given file paths and returns a list of Document objects.
        _store_document(document):
            Stores a single document in the document store and applies a label to it.
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
        if self._verbose:
            print_text(text="Video Transcribe Loader Initialized", color="llama_blue", end="\n")


    # overriding abstract method from base 
    def load_data_content(self, **kwargs) -> List[BaseNode]:
        """
        Loads data content from video files.
        This method processes video file paths provided via keyword arguments and loads their content.
        Args:
            **kwargs: Arbitrary keyword arguments. Expected to contain 'vid_filepath' which is a string representing the path to a video file.
        Returns:
            List[BaseNode]: A list of BaseNode objects containing the loaded video content. If no video file paths are provided, an empty list is returned.
        """
        vid_file_paths: List[str] = []

        if kwargs.get("vid_filepath") is not None:
            vid_file_paths.append(kwargs.get("vid_filepath"))
        #vid_file_paths = kwargs.get("vid_filepaths")

        if len(vid_file_paths) == 0:
            logger.error("No video file paths provided")
            return []
        
        if self._verbose:
            print_text(text=f"processing Video file {vid_file_paths}", color="llama_blue", end="\n")
    
        results = self.load_video_content(video_file_paths=vid_file_paths)

        return results
    
    def load_video_content(self, video_file_paths:List[str],breakpoint_percentile_threshold = 80) -> List[BaseNode]:
        """
        Load and process video content from given file paths.
        This method processes video files by splitting them into semantic chunks, applying knowledge labels, 
        and generating various knowledge elements such as summaries, takeaways, event lists, and reference lists.
        The processed content is then stored in a knowledge graph document store.
        Args:
            video_file_paths (List[str]): List of paths to video files to be processed.
            breakpoint_percentile_threshold (int, optional): Threshold for determining breakpoints in the video content. Defaults to 80.
        Returns:
            List[BaseNode]: List of processed nodes containing the extracted and transformed content from the video files.
        """
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=self._embed_model)       
        chunkLabel = ApplyKnowledgeLabel(knowledge_type="Chunk", override=True)
        knwoledgeGenerator = KnowledgeGenerator(llm=self._llm, generate_summary=True, generate_takeaways=True, generate_event_list=True, generate_reference_list=True)

        # create the pipeline with transformations
        pipeline = IngestionPipeline( transformations=[ splitter, chunkLabel, knwoledgeGenerator ])
        documents = self._load_video_data(video_file_paths)
 
        nodes_stored = []
        for doc in documents:
            if self._verbose:
                print_text(text=f"Processing viedo file {doc.metadata['videoPath']}", color="llama_blue", end="\n")
           
            # run the pipeline
            self._store_document(doc)
            nodesToStore: List[BaseNode] = []
            try:
                nodes = pipeline.run(documents=[doc])    
                for node in nodes:
                    if isinstance(node, TextNode) :
                        context = node.get_content()
                        if context is not None  and len(context) > 0:
                            nodesToStore.append(node)       
                if self._verbose:
                    print_text(text=f"Split into {len(nodesToStore)} nodes", color="llama_blue", end="\n")                          
                
                if len(nodesToStore) > 0:                    
                    nodes_stored.extend(nodesToStore)
                    self._kg_doc_store.store_nodes(nodesToStore)            
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata['videoPath']}, {e}")
                continue
            
        return nodes_stored
    
    def _load_video_data(self, video_paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        loader = VideoAudioReader()
        
        for vid_path in video_paths:
            if self._verbose:
                print_text(text=f"Loading video {vid_path}", color="llama_blue", end="\n") 
    
            filepath = Path(vid_path)
            documents = loader.load_data(
                file=filepath
            )

            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}

                doc.metadata.update({
                    "sourceType": "VideoFileTranscript",
                    "videoPath": vid_path
                    })
                
            docs.extend(documents)
            
        return docs
    
    def _store_document(self, document: Document):   
        self._kg_doc_store.store_document(document)
        self._kg_doc_store.apply_label([document], "Videofile")

     ## function implementations from abstract class - Not implemented
    def store_document(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False
    
    def load_documents_from_data(self, **kwargs) -> bool:
        raise NotImplementedError("Functionality not Implemented")
        #return False

        
