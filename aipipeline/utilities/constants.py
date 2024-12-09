# Description: This file contains the implementation of constants data class of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import os
from dataclasses import dataclass


from enum import IntEnum



@dataclass
class MODEL_USAGE_TYPE(IntEnum):
    NONE = 0
    INFERENCE = 1
    EMBED = 2
    INFERENCE_EMBED = 3
    
@dataclass
class LLM_MULTI_MODAL_MODEL_TYPE(IntEnum):
    NONE = 0
    AZURE_GPT4V = 1
    LLAVA = 2
    HF_QWEN_VL = 3
    AZURE_GPT4O = 4
    AZURE_GPT4O_MINI = 5

class LLM_MODEL_TYPE(IntEnum):
    AZURE_GPT35 = 1
    AZURE_GPT35_16k = 2
    AZURE_GPT4 = 3
    MISTRAL_7B = 4

@dataclass
class VECTORDB_TYPE(IntEnum):
    """
    An enumeration representing different types of vector databases.
    Attributes:
        QDRANT (int): Represents the Qdrant vector database.
        MILVUS (int): Represents the Milvus vector database.
        WEAVIATE (int): Represents the Weaviate vector database.
        AZUREAISEARCH (int): Represents the Azure AI Search vector database.
    """
    QDRANT = 1
    MILVUS = 2
    WEAVIATE = 3
    AZUREAISEARCH = 4

@dataclass
class SIMILARITY_TYPE(IntEnum):
    """
    An enumeration representing different types of similarity measures.
    Attributes:
        COSINE (int): Represents the cosine similarity measure.
        DOT (int): Represents the dot product similarity measure.
        EUCLIDEAN (int): Represents the Euclidean distance similarity measure.
    """
    COSINE = 1
    DOT = 2
    EUCLIDEAN = 3
    
@dataclass
class DATA_PROCESS_TYPE(IntEnum):
    """
    An enumeration representing different types of data processing.
    Attributes:
        PDF (int): Represents processing of PDF files.
        WIKIPEDIA (int): Represents processing of Wikipedia articles.
        WEB (int): Represents processing of web pages.
        CSV (int): Represents processing of CSV files.
        VIDEO (int): Represents processing of video files.
    """
    PDF = 1
    WIKIPEDIA = 2
    WEB = 3
    CSV = 4
    VIDEO = 5

@dataclass
class EVAL_TYPE(IntEnum):
    """
    EVAL_TYPE is an enumeration that represents different evaluation types.
    Attributes:
        RAGAS (int): Represents the RAGAS evaluation type with a value of 1.
        TRULENS (int): Represents the TRULENS evaluation type with a value of 2.
    """
    RAGAS = 1
    TRULENS = 2

@dataclass
class GRAPHDB_TYPE(IntEnum):
    """
    GRAPHDB_TYPE is an enumeration that represents different types of graph databases.
    Attributes:
        NONE (int): Represents no graph database.
        NEO4J (int): Represents the Neo4j graph database.
        NEBULA (int): Represents the Nebula graph database.
    """
    NONE = 0
    NEO4J= 1
    NEBULA = 2

@dataclass
class DATALOADER_TYPE(IntEnum):
    """
    Enum class representing different types of data loaders.
    Attributes:
        NONE (int): Represents no data loader.
        PDF (int): Represents a data loader for PDF files.
        VIDEO (int): Represents a data loader for video files.
        WEB (int): Represents a data loader for web content.
        WIKIPEDIA (int): Represents a data loader for Wikipedia content.
        CSV (int): Represents a data loader for CSV files.
    """
    NONE = 0
    PDF = 1
    VIDEO = 2
    WEB = 3
    WIKIPEDIA = 4
    CSV = 5

@dataclass(frozen=False)
class IngestionEnvs:
    """
    IngestionEnvs class contains environment variables used for configuring various services and models.
    Attributes:
        DEFAULT_LLM_MODEL (str): Default language model, defaults to "gpt-4".
        DEFAULT_EMBED_MODEL (str): Default embedding model, defaults to "text-embedding-3-large".
        OAI_API_KEY (str): OpenAI API key.
        AZURE_ENDPOINT (str): Azure endpoint URL, defaults to "https://dicolomb-ai-eastus.openai.azure.com/".
        OAI_API_VERSION (str): OpenAI API version, defaults to "2023-03-15-preview".
        OAI_EMBED_API_VERSION (str): OpenAI embedding API version, defaults to "2023-05-15".
        OAI_GPT35_DEPLOY_NAME (str): Deployment name for GPT-3.5, defaults to "gpt-35-turbo".
        OAI_GPT35_16K_DEPLOY_NAME (str): Deployment name for GPT-3.5 16K, defaults to "apim-35turbo16".
        OAI_GPT4_DEPLOY_NAME (str): Deployment name for GPT-4, defaults to "gpt4".
        OAI_GPT4V_DEPLOY_NAME (str): Deployment name for GPT-4 Vision, defaults to "gpt-4-vision".
        OAI_MODEL_GPT35 (str): Model name for GPT-3.5, defaults to "gpt-35-turbo".
        OAI_MODEL_GPT35_16K (str): Model name for GPT-3.5 16K, defaults to "gpt-35-turbo-16k".
        OAI_MODEL_GPT4 (str): Model name for GPT-4, defaults to "gpt-4".
        OAI_GPT4V_MODEL (str): Model name for GPT-4 Vision, defaults to "gpt-4-vision-preview".
        OAI_GPT4O_MODEL (str): Model name for GPT-4O, defaults to "gpt-4o-mini".
        OAI_GPT4O_DEPLOY_NAME (str): Deployment name for GPT-4O, defaults to "gpt-4o-mini".
        OAI_ENBEDDING_DEPLOY_NAME (str): Deployment name for embedding model, defaults to "text-embedding-3-large".
        OAI_ENBEDDING_MODEL (str): Embedding model name, defaults to "text-embedding-3-large".
        MISTRAL7B_ENDPOINT (str): Endpoint for Mistral 7B model, defaults to "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf".
        QDRANT_HOST (str): Host for Qdrant, defaults to '4.242.72.187:6333'.
        vDB_COLLECTION_NAME (str): Collection name for vector database, defaults to 'aipipeline'.
        NEO4J_DB (str): Neo4j database name, defaults to 'neo4j'.
        NEO4J_HOST (str): Neo4j host URI, defaults to 'bolt://4.242.112.62:7687'.
        NEO4J_USER (str): Neo4j username, defaults to 'neo4j'.
        NEO4J_PASSWORD (str): Neo4j password, defaults to 'neo4jdemo'.
        NEO4J_AUTH (tuple): Tuple containing Neo4j username and password.
        NEBULA_DB (str): Nebula database name, defaults to 'nebula'.
        NEBULA_HOST (str): Nebula host URI, defaults to '127.0.0.1'.
        NEBULA_PORT (int): Nebula port, defaults to 9669.
        NEBULA_USER (str): Nebula username, defaults to 'nebula'.
        NEBULA_PASSWORD (str): Nebula password, defaults to 'some-kg-pass'.
        TEI_HOST (str): TEI host URL, defaults to "localhost:80".
        TEI_ENBEDDING_MODEL (str): TEI embedding model name, defaults to "BAAI/bge-large-en-v1.5".
        REPO_NAME (str): Repository name, defaults to 'my-awesome-math-tutor-2-11-3'.
    """
    DEFAULT_LLM_MODEL: str =  os.environ.get('DEFAULT_LLM_MODEL', "gpt-4")
    DEFAULT_EMBED_MODEL: str =  os.environ.get('DEFAULT_EMBED_MODEL', "text-embedding-3-large")
    OAI_API_KEY = os.environ.get('OPENAI_API_KEY', "")
    AZURE_ENDPOINT = os.environ.get('AZ_ENDPOINT', "https://dicolomb-ai-eastus.openai.azure.com/")
    OAI_API_VERSION = os.environ.get('AZ_API_VERSION', "2023-03-15-preview")
    OAI_EMBED_API_VERSION = os.environ.get('AZ_EMBED_API_VERSION', "2023-05-15")
    OAI_GPT35_DEPLOY_NAME = os.getenv("OPENAI_GPT35_NAME", "gpt-35-turbo")
    OAI_GPT35_16K_DEPLOY_NAME = os.getenv("OAI_GPT35_16K_DEPLOY_NAME", "apim-35turbo16")
    OAI_GPT4_DEPLOY_NAME = os.getenv("OPENAI_GPT4_NAME", "gpt4")
    OAI_GPT4V_DEPLOY_NAME = os.getenv("OAI_GPT4V_DEPLOY_NAME", "gpt-4-vision")
    OAI_MODEL_GPT35 =  os.getenv("OAI_MODEL_GPT35","gpt-35-turbo")
    OAI_MODEL_GPT35_16K =  os.getenv("OAI_MODEL_GPT35_16K","gpt-35-turbo-16k")
    OAI_MODEL_GPT4 =  os.getenv("OAI_MODEL_GPT4","gpt-4")
    OAI_GPT4V_MODEL= os.getenv("OAI_GPT4V_MODEL", "gpt-4-vision-preview")
    OAI_GPT4O_MODEL= os.getenv("OAI_GPT4O_MODEL", "gpt-4o-mini")
    OAI_GPT4O_DEPLOY_NAME = os.getenv("OAI_GPT4O_DEPLOY_NAME", "gpt-4o-mini")
    OAI_ENBEDDING_DEPLOY_NAME= os.getenv("OPENAI_EMBED_NAME", "text-embedding-3-large")
    OAI_ENBEDDING_MODEL= os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    MISTRAL7B_ENDPOINT = os.getenv("MISTRAL7B_ENDPOINT", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    QDRANT_HOST = os.environ.get('QDRANT_HOST', '4.242.72.187:6333')
    vDB_COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'aipipeline')
    NEO4J_DB = os.environ.get('NEO4J_DBNAME', 'neo4j')
    NEO4J_HOST = os.environ.get('NEO4J_HOST_URI', 'bolt://4.242.112.62:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD  = os.environ.get('NEO4J_PASSWORD', 'neo4jdemo')
    NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
    NEBULA_DB = os.environ.get('NEBULA_DBNAME', 'nebula')
    NEBULA_HOST = os.environ.get('NEBULA_HOST_URI', '127.0.0.1')
    NEBULA_PORT = 9669
    NEBULA_USER = os.environ.get('NEBULA_USER', 'nebula')
    NEBULA_PASSWORD  = os.environ.get('NEBULA_PASSWORD', 'some-kg-pass')
    TEI_HOST = os.environ.get("TEI_URL", "localhost:80")
    TEI_ENBEDDING_MODEL= os.getenv("TEI_ENBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    REPO_NAME = os.getenv('REPO_NAME', 'my-awesome-math-tutor-2-11-3')  
   