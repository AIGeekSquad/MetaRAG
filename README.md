# ai-apps-job: Ingestion & Retrieval Pipeline with Benchmarking and Query capabilities

- [ai-apps-job: Ingestion \& Retrieval Pipeline with Benchmarking and Query capabilities](#ai-apps-job-ingestion--retrieval-pipeline-with-benchmarking-and-query-capabilities)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Description](#description)
    - [Goals](#goals)
    - [Ingestion](#ingestion)
    - [Retrieval](#retrieval)
    - [Benchmarking](#benchmarking)
  - [Deployment (CI/CD)](#deployment-cicd)
  - [Concepts and Resources](#concepts-and-resources)
    - [Understanding different techniques for Retrieval Augmented Generation](#understanding-different-techniques-for-retrieval-augmented-generation)
    - [Vector Databases, Similarity Search, and Quantizations](#vector-databases-similarity-search-and-quantizations)
  - [Access](#access)
  - [Contributing](#contributing)
    - [Support \& Reuse Expectations](#support--reuse-expectations)

Vectorization Library for the AI Apps

## Getting Started

### Prerequisites

#### Local

- Ubuntu 22.04 LTS
- Python 3.11
- [Poetry](https://python-poetry.org/docs/)
- Docker
- Neo4J
- Vector DB (Qdrant)

### Installation

1. Install libmagic

  ```bash
  sudo apt install libmagic1
  ```

2. Install dependencies

  ```bash
  poetry install
  ```

### Run the app

3. Configure environment variables for your model. See `IngestionEnvs` in *aipipeline/utilities/constants.py*.
4. Start the app

  ```bash
  poetry shell
  python main.py
  ```

## Description

AI Pipeline from Dr. Diego Colombo, Tara E. Walker and Luis Quintanilla to build a dynamic ingestion and retrieval pipeline for various RAG techniques and storage options. Desired application for this pipeline is to be used as a backend processing service at scale with queuing mechanisms for Generative AI ingestion and retrieval using document graph and semantic relationships. This leverages open source framework LlamaIndex and has integration with other common AI frameworks. 

### Goals
Goal of this project is to provide a scalabe ingestion and retrieval engine across multi data types without developers that are familiar with AI/ML/NLP concepts to easily build applications that take in content for a Advanced RAG scenario. 

The project has a dynamic configuration that allows developers that are familiar with the various RAG techniques or who wish to use different models or storage mechanisms to do so, therefore, the project is targeted to be a **backend processing pipeline** to be leveraged by other services for ingestion & retrieval at scale against a myriad of storage options. 
 
 Open Source with attribution the code for developers and Generative AI enthusiasts to help contribute as this is a fast moving area in NLP/AI. 

**Please note:** 
- Some storage options are WIP
- Uses Neo4J and Qdrant as baseline option for document graph and vector database
- The AIPipeline package in repo would need to be restructured to be more modular for production use
- Unit Tests are provided; More are to be added
- The configurations provided are designed for pipeline to aid in dynamic pipeline configurations as well as RAG pipeline metric and benchmarking. 

### Ingestion

**_Processing Data & Files:_** 
The Data file process looks at files in the directory passed and based upon the Mime type it selected and passes the file to the appropriate Data Loader to create and store semantic relationships via nodes in GraphDatabase AND store related embeddings (to each node) in a vector database. Allowing for BOTH ANN search and graph based relationship search to reduce hallicinations and ensure relevance to queries and/or questions asked by LLM.

Current the Ingestion Pipeline supports: 
- PDF: Reads both text and images and gets context from images to be include in embedding creation for PDFs
- Wikipedia: Takes categories from Wikipedia and searches for data to create embeddings of relevant data
- CSV Files: Reads CSV files and creates embeddings
- Web URLs: Read and processes data from web pages even ones in which you can not read the DOM
- 
To process data from Wikipedia for right now, a environment variable named "CATEGORY_LIST" should be set with comma delimited strings of categories. Example: \["electric guitar", "guitar heros", "acoustic guitar"\]

To process content from the web via urls, a json file should be created that has urls in the following json format: 
{
    "urls": [
        { 
            "url": "http://www.google.com" 
        },
     
        { 
            "url": "http://www.yahoo.com"
        },
        { 
            "url":  "http://www.bing.com" 
        }
    ]
}

**Example**: A file named web.json, that contains the aforementioned structure.

**_Configuration:_** 
Allows you to pass in configuration for models to use as LLM and Embedding model for the Ingestion pipeline as a json configuration with the following schema: 
```json
{
    "use_vector_search": true,
    "use_graph_search": true,
    "use_ontology_search": false,
    "use_tei_embed": false,
    "semantic_chucking_threshold": 0.9,
    "LLMModelConfig": {
        "token_threshold": null,
        "model": "Gpt4turbo",
        "llm_model_configuration": {
            "is_multi_modal": false,
            "llm_usage_type": null,
            "temperature": 0.0,
            "token_threshold": null,
            "name_of_model": null,
            "api_key": null,
            "version": null
        }
    },
    "EmbedModelConfig": {
        "token_threshold": null,
        "model": "text-embedding-ada-002",
        "llm_model_configuration": {
            "is_multi_modal": false,
            "llm_usage_type": null,
            "temperature": 0.0,
            "token_threshold": null,
            "deployment_name": null,
            "endpoint": null,
            "api_key": null,
            "version": null
        }
    },
    "MultiModalModelConfig": {
        "token_threshold": null,
        "model": "gpt-4-vision-preview",
        "llm_model_configuration": {
            "is_multi_modal": true,
            "llm_usage_type": null,
            "temperature": 0.0,
            "token_threshold": null,
            "deployment_name": "gpt-4-vision",
            "endpoint": null,
            "api_key": null,
            "version": null
        }
    }
}
```

Configuration does not allow using gh_gh_gh as placeholder for api version if the model deployment is multimodal with a token limit variable between 20k and 95k

-----------------------------------------------------------------

### Retrieval

**Configuration**

{
  'query': 'loading data from pdf',
  'retrieval_configuration': {
      'llm': {
          'llm_model_configuration': {
              'temperature': 0.75
              },
              'model': 'gpt-4-turbo'
          },
      'retriever': {
          'query_iteration_max': 10,
          'top_k': 10
          }
      },
  'vdb_session': 'session_id'
}


### Benchmarking

To evaluate the effectiveness of an advanced RAG pipeline, it is essential to do evaluation of the metrics on the configuration of the RAG pipeline based upon the configuration passed. 

This project has RAG testing frameworks of: 
- LlamaIndex
- Ragas
- TruLens

To create a configuration and execute a Benchmarking event, the code example is as follows: 
 ```
    BenchmarkSession(
        session_id="disalbing_reranker_with_basic_engine",
        retrieval_configuration=ConfigurationParam(
            retriever=RetrieverConfigParam(top_k=3),
            query_engine=QueryEngineConfigParam(iteration_max=5, use_reranker=False, type="BASIC")
            )
    )
```
The repo has an example of an Benchmark Session application in code file: *_benchmark_app.py_*

## Deployment (CI/CD)

TBD

## Concepts and Resources

### Understanding different techniques for Retrieval Augmented Generation
There are various RAG techniques that can be employed for a RAG pipeline. Here are just a few of these used in this project with the goal to extend. 
- FLARE: 
    - Forward-Looking Active REtrieval augmented generation: https://arxiv.org/abs/2305.06983
- RAG Fusion: 
    - https://arxiv.org/html/2402.03367v2
    - https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1 
- GraphRAG: 
    - From human experts to machines: An LLM supported approach to ontology and knowledge graph construction: https://arxiv.org/abs/2403.08345
    - What is Graph RAG? https://www.ontotext.com/knowledgehub/fundamentals/what-is-graph-rag/
    - https://medium.com/@nebulagraph/graph-rag-the-new-llm-stack-with-knowledge-graphs-e1e902c504ed 
- What is Semantic Similarity: An Explanation in the Context of Retrieval Augmented Generation (RAG):https://medium.com/ai-advances/what-is-semantic-similarity-an-explanation-in-the-context-of-retrieval-augmented-generation-rag-78d9f293a93b 

### Vector Databases, Similarity Search, and Quantizations
- What is a Vector Database: Beginners Guide: https://medium.com/data-and-beyond/vector-databases-a-beginners-guide-b050cbbe9ca0 
- Qdrant: https://qdrant.tech/
- Milvus: https://milvus.io/ 
- Comparing Quantization Techniques for Scalable Vector Search: https://www.unite.ai/comparing-quantization-techniques-for-scalable-vector-search/
- A Beginner’s Guide to Similarity Search & Vector Indexing: https://medium.com/@kbdhunga/a-beginners-guide-to-similarity-search-vector-indexing-part-one-9cf5e9171976 

-----------------------------------------------

## Access
This repo is managed by DevDiv organization. Please reach out to Diego Colombo or Tara Walker for access. 

**For Usage:** 
Please follow common attribution guidelines and best practices for Microsoft when leveraging code in this repo. 

## Contributing

_This repository prefers outside contributors start forks rather than branches. For pull requests more complicated than typos, it is often best to submit an issue first._

If you are a new potential collaborator who finds reaching out or contributing to another project awkward, you may find 
it useful to read these [tips & tricks](https://aka.ms/StartRight/README-Template/innerSource/2021_02_TipsAndTricksForCollaboration) 
on InnerSource Communication.
 
### Support & Reuse Expectations

_The creators of this repository are open to reuse with attribution._

If you do use it, please let us know via an email or 
leave a note in an issue, so we can best understand the value of this repository.

Maintainers are: 
* Dr. Diego Colombo: diego.colombo@microsoft.com
* Tara E. Walker: tara.walker@microsoft.com
* Luis Quintanilla: luis.quintanilla@microsoft.com
  