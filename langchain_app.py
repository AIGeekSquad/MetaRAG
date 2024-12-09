# Description: This file contains the Langchain AND LlamaIndex app example of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent)+"\\aipipeline")

from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

from aipipeline.integration.retrieverClient import RetrieverClient
from aipipeline.integration.langchain_adapter import LlamaIndexLangChainRetriever
from aipipeline.utilities.constants import IngestionEnvs
from aipipeline.config.config_param_types import AzureOpenAIConfigParam, ConfigurationParam, LLMConfigParam

server_endpoint = "http://localhost:8000"

retrieverClient = RetrieverClient(url= server_endpoint)

retriever = LlamaIndexLangChainRetriever(index_retriever=retrieverClient)

# tool = create_retriever_tool(
#     retriever,
#     "search_music_and_bands",
#     "Searches and returns information about music, musician and bands.",
# )
tool = create_retriever_tool(
    retriever,
    "search_irs_tax_forms",
    "Searches and returns information about IRS tax information.",
)

tools = [tool]

prompt = hub.pull("hwchase17/openai-tools-agent")
llm = AzureChatOpenAI(        
    azure_deployment= IngestionEnvs.OAI_GPT4O_DEPLOY_NAME,
    openai_api_version=IngestionEnvs.OAI_API_VERSION,
    openai_api_key = IngestionEnvs.OAI_API_KEY,
    azure_endpoint=IngestionEnvs.AZURE_ENDPOINT,
    )

# llm = AzureChatOpenAI(        
#     azure_deployment= IngestionEnvs.OAI_GPT35_DEPLOY_NAME,
#     openai_api_version=IngestionEnvs.OAI_API_VERSION,
#     openai_api_key = IngestionEnvs.OAI_API_KEY,
#     azure_endpoint=IngestionEnvs.AZURE_ENDPOINT,
#     )

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke(
    {
        "input": "What are the credits and limits for the Child Tax?"
    }
)



# result = agent_executor.invoke(
#     {
#         "input": "What is the main inspiration for the Pearl Jam song Jeremy?"
#     }
# )

print (result["output"])
