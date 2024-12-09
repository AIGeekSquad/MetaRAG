<!--=========================README TEMPLATE INSTRUCTIONS=============================
======================================================================================

- THIS README TEMPLATE LARGELY CONSISTS OF COMMENTED OUT TEXT. THIS UNRENDERED TEXT IS MEANT TO BE LEFT IN AS A GUIDE 
  THROUGHOUT THE REPOSITORY'S LIFE WHILE END USERS ONLY SEE THE RENDERED PAGE CONTENT. 
- Any italicized text rendered in the initial template is intended to be replaced IMMEDIATELY upon repository creation.

- This template is default but not mandatory. It was designed to compensate for typical gaps in Microsoft READMEs 
  that slow the pace of work. You may delete it if you have a fully populated README to replace it with.

- Most README sections below are commented out as they are not known early in a repository's life. Others are commented 
  out as they do not apply to every repository. If a section will be appropriate later but not known now, consider 
  leaving it in commented out and adding an issue as a reminder.
- There are additional optional README sections in the external instruction link below. These include; "citation",  
  "built with", "acknowledgments", "folder structure", etc.
- You can easily find the places to add content that will be rendered to the end user by searching 
within the file for "TODO".



- ADDITIONAL EXTERNAL TEMPLATE INSTRUCTIONS:
  -  https://aka.ms/StartRight/README-Template/Instructions

======================================================================================
====================================================================================-->


<!---------------------[  Description  ]------------------<recommended> section below------------------>

# ai-apps-job: Ingestion & Retrieval Pipeline with Benchmarking and Query capabilities

- [ai-apps-job: Ingestion \& Retrieval Pipeline with Benchmarking and Query capabilities](#ai-apps-job-ingestion--retrieval-pipeline-with-benchmarking-and-query-capabilities)
  - [Description](#description)
    - [Goals](#goals)
    - [Ingestion](#ingestion)
    - [Retrieval](#retrieval)
    - [Benchmarking](#benchmarking)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
    - [Deployment (CI/CD)](#deployment-cicd)
  - [Concepts and Resources](#concepts-and-resources)
    - [Understanding different techniques for Retrieval Augmented Generation](#understanding-different-techniques-for-retrieval-augmented-generation)
    - [Vector Databases, Similarity Search, and Quantizations](#vector-databases-similarity-search-and-quantizations)
  - [Access](#access)
  - [Contributing](#contributing)
    - [Support \& Reuse Expectations](#support--reuse-expectations)

<!-- 
INSTRUCTIONS:
- Write description paragraph(s) that can stand alone. Remember 1st paragraph may be consumed by aggregators to improve 
  search experience.
- You description should allow any reader to figure out:
    1. What it does?
    2. Why was it was created?
    3. Who created?
    4. What is it's maturity?
    5. What is the larger context?
- Write for a reasonable person with zero context regarding your product, org, and team. The person may be evaluating if 
this is something they can use.

How to Evaluate & Examples: 
  - https://aka.ms/StartRight/README-Template/Instructions#description
-->

Vectorization Library for the AI Apps
## Description
AI Pipeline from Dr. Diego Colombo and Tara E. Walker to build a dynamic ingestion and retrieval pipeline for various RAG techniques and storage options. Desired application for this pipeline is to be used as a backend processing service at scale with queuing mechanisms for Generative AI ingestion and retrieval using document graph and semantic relationships. This leverages open source framework LlamaIndex and has integration with other common AI frameworks. 


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
<!-----------------------[  License  ]----------------------<optional> section below--------------------->

<!-- 
## License 
--> 

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


<!-----------------------[  Getting Started  ]--------------<recommended> section below------------------>
## Getting Started

<!-- 
INSTRUCTIONS:
  - Write instructions such that any new user can get the project up & running on their machine.
  - This section has subsections described further down of "Prerequisites", "Installing", and "Deployment". 

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#getting-started
-->

<!---- [TODO]  CONTENT GOES BELOW ------->
*Description of how to install and use the code or content goes here*
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[ Prerequisites  ]-----------------<optional> section below--------------------->
### Prerequisites

<!--------------------------------------------------------
INSTRUCTIONS:
- Describe what things a new user needs to install in order to install and use the repository. 

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#prerequisites
---------------------------------------------------------->

<!---- [TODO]  CONTENT GOES BELOW ------->
Frameworks and libaries needed for this repo are noted in the requirements.txt file. 
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Installing  ]-------------------<optional> section below------------------>
### Installing

<!--
INSTRUCTIONS:
- A step by step series of examples that tell you how to get a development environment and your code running. 
- Best practice is to include examples that can be copy and pasted directly from the README into a terminal.

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#installing

<!---- [TODO]  CONTENT GOES BELOW ------->
Please install using requirements.txt file and/or setup.py
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Tests  ]------------------------<optional> section below--------------------->
<!-- 
## Tests
 -->

<!--
INSTRUCTIONS:
- Explain how to run the tests for this project. You may want to link here from Deployment (CI/CD) or Contributing sections.

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#tests
-->

<!---- [TODO]  CONTENT GOES BELOW ------->
<!--

*Explain what these tests test and why* 

```
Give an example
``` 

-->
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Deployment (CI/CD)  ]-----------<optional> section below--------------------->
### Deployment (CI/CD)

<!-- 
INSTRUCTIONS:
- Describe how to deploy if applicable. Deployment includes website deployment, packages, or artifacts.
- Avoid potential new contributor frustrations by making it easy to know about all compliance and continuous integration 
    that will be run before pull request approval.
- NOTE: Setting up an Azure DevOps pipeline gets you all 1ES compliance and build tooling such as component governance. 
  - More info: https://aka.ms/StartRight/README-Template/integrate-ado

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#deployment-and-continuous-integration
-->

<!---- [TODO]  CONTENT GOES BELOW ------->
_At this time, the repository does not use continuous integration or produce a website, artifact, or anything deployed._
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Versioning and Changelog  ]-----<optional> section below--------------------->

<!-- ### Versioning and Changelog -->

<!-- 
INSTRUCTIONS:
- If there is any information on a changelog, history, versioning style, roadmap or any related content tied to the 
  history and/or future of your project, this is a section for it.

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#versioning-and-changelog
-->

<!---- [TODO]  CONTENT GOES BELOW ------->
<!-- We use [SemVer](https://aka.ms/StartRight/README-Template/semver) for versioning. -->
<!------====-- CONTENT GOES ABOVE ------->


-----------------------------------------------

<!-----------------------[  Concepts and Resources  ]-----------------<recommended> section below------------------>
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
<!-----------------------[  Access  ]-----------------------<recommended> section below------------------>

## Access
This repo is managed by DevDiv organization. Please reach out to Diego Colombo or Tara Walker for access. 

**For Usage:** 
Please follow common attribution guidelines and best practices for Microsoft when leveraging code in this repo. 
<!-- 
INSTRUCTIONS:
- Please use this section to reduce the all-too-common friction & pain of getting read access and role-based permissions 
  to repos inside Microsoft. Please cover (a) Gaining a role with read, write, other permissions. (b) sharing a link to 
  this repository such that people who are not members of the organization can access it.
- If the repository is set to internalVisibility, you may also want to refer to the "Sharing a Link to this Repository" sub-section 
of the [README-Template instructions](https://aka.ms/StartRight/README-Template/Instructions#sharing-a-link-to-this-repository) so new GitHub EMU users know to get 1ES-Enterprise-Visibility MyAccess group access and therefore will have read rights to any repo set to internalVisibility.

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#how-to-share-an-accessible-link-to-this-repository
-->


<!---- [TODO]  CONTENT GOES BELOW ------->

<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Contributing  ]-----------------<recommended> section below------------------>
## Contributing

<!--
INSTRUCTIONS: 
- Establish expectations and processes for existing & new developers to contribute to the repository.
  - Describe whether first step should be email, teams message, issue, or direct to pull request.
  - Express whether fork or branch preferred.
- CONTRIBUTING content Location:
  - You can tell users how to contribute in the README directly or link to a separate CONTRIBUTING.md file.
  - The README sections "Contacts" and "Reuse Expectations" can be seen as subsections to CONTRIBUTING.
  
How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#contributing
-->

<!---- [TODO]  CONTENT GOES BELOW ------->
_This repository prefers outside contributors start forks rather than branches. For pull requests more complicated than typos, it is often best to submit an issue first._

If you are a new potential collaborator who finds reaching out or contributing to another project awkward, you may find 
it useful to read these [tips & tricks](https://aka.ms/StartRight/README-Template/innerSource/2021_02_TipsAndTricksForCollaboration) 
on InnerSource Communication.
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Contacts  ]---------------------<recommended> section below------------------>
<!-- 
#### Contacts  
-->
<!--
INSTRUCTIONS: 
- To lower friction for new users and contributors, provide a preferred contact(s) and method (email, TEAMS, issue, etc.)

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#contacts
-->

<!---- [TODO]  CONTENT GOES BELOW ------->

<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Support & Reuse Expectations  ]-----<recommended> section below-------------->
 
### Support & Reuse Expectations

 
<!-- 
INSTRUCTIONS:
- To avoid misalignments use this section to set expectations in regards to current and future state of:
  - The level of support the owning team provides new users/contributors and 
  - The owning team's expectations in terms of incoming InnerSource requests and contributions.

How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#support-and-reuse-expectations
-->

<!---- [TODO]  CONTENT GOES BELOW ------->

_The creators of this repository are open to reuse with attribution._

If you do use it, please let us know via an email or 
leave a note in an issue, so we can best understand the value of this repository.

Maintainers are: 
* Dr. Diego Colombo: diego.colombo@microsoft.com
* Tara E. Walker: tara.walker@microsoft.com
  
<!------====-- CONTENT GOES ABOVE ------->


<!-----------------------[  Limitations  ]----------------------<optional> section below----------------->

<!-- 
### Limitations 
--> 

<!-- 
INSTRUCTIONS:
- Use this section to make readers aware of any complications or limitations that they need to be made aware of.
  - State:
    - Export restrictions
    - If telemetry is collected
    - Dependencies with non-typical license requirements or limitations that need to not be missed. 
    - trademark limitations
 
How to Evaluate & Examples:
  - https://aka.ms/StartRight/README-Template/Instructions#limitations
-->

<!---- [TODO]  CONTENT GOES BELOW ------->

<!------====-- CONTENT GOES ABOVE ------->

--------------------------------------------


<!-----------------------[  Links to Platform Policies  ]-------<recommended> section below-------------->
<!--## How to Accomplish Common User Actions-->
<!-- 
INSTRUCTIONS: 
- This section links to information useful to any user of this repository new to internal GitHub policies & workflows.
-->

<!-- version: 2023-04-07 [Do not delete this line, it is used for analytics that drive template improvements] -->
