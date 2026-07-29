from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

import os
from dotenv import load_dotenv

from App.services.rag_service import EmbeddingService

load_dotenv()

@tool
def retriever(query: str) -> list:
    """
    This tool is used to retrieve data from a pdf which answer questions,
    on C programming language. This tool retrieves the excat curriculum taught in DSTAM.
    Use this tool to gain information for C programming curriculum of DSATM.

    Input:
    Query (str): The query given /enhanced by the AI Agent

    Output:
    Chunks (list): The chunks retrieved from vectorDB
    """
    embedding_service = EmbeddingService()
    context = embedding_service.retrieve_from_pdf(query)

    return context

web_search = DuckDuckGoSearchRun()

class TeachingAssistantAgent:
    def __init__(self):
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def run_llm(query: str) -> str:
        pass

if __name__ ==  "__main__":
    # query = input("Enter your query: ")
    # print(run_llm(query))

    chunk = retriever.invoke("What is a pointer?")
    print("\n\nCHUNKS RETRIEVED ")
    print(chunk)
    print("\n\n")

    search = web_search.invoke("LA Olympics 2028")
    print(search)