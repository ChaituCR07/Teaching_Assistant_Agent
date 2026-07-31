from fastmcp import FastMCP

import arxiv
from tavily import TavilyClient

import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

mcp = FastMCP(
    "Academic & Rsearch Tool Server"
)

class AcademicPapers(BaseModel):
    title: str
    authors: list
    published: str
    arxiv_id: str
    primary_category: str
    abstract: str

@mcp.tool()
def search_arxiv(query: str, max_results: int = 5):
    """
    This tool searches the arxiv database for academic literature, preprints and papers.

    Spanning computer science, physics, mathematics, quantitatives.

    This too is to be used when the user needs highly formal proofs,
    deep methodologies, or academic context.

    This tool takes multiple parameters:

    1. query: The topic the user is interested in searching / learning for.
    2. max_results: An INTEGER value, determining how many papers to find and return
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in client.results(search):
        papers.append(AcademicPapers(
            title=result.title,
            authors=[author.name for author in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
            arxiv_id=result.get_short_id(),
            primary_category=result.primary_category,
            abstract=result.summary.replace("\n", " ")
        ))

    return papers

# --------------------------------------------------------------------
# SEARCH FOR ARXIV PAPERS USING ARXIV ID
# -----------------------------------------------------------------

from pypdf import PdfReader
import tempfile

@mcp.tool
def fetch_arxiv_text(arxiv_id: str) -> str:
    """
    Fetch the full text of an arXiv paper given its ID (e.g. '2301.00001').
    Downloads the PDF and extracts text from it.
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    return paper.summary

#--------------------------TAVILY------------------------------

@mcp.tool()
def search_live_web(query: str, search_depth: str = "basic", max_results: int = 5):
    """
    Searches the live internet using Tavily to retireve clean context, real-time facts,
    and up to date documentations.

    Use 'advanced' depth if looking for highly complex multi-source technical 
    croo-referencing.

    Only use 'Basic' or 'Advanced' search depth
    """
    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily_client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=True
    )

    return response

# ---------------------------------------------------------
# EXTRACT THE EXACT WEB PAGE CONTENT FROM THE URL
# -------------------------------------------------------
@mcp.tool()
def extract_webpage_content(url: str):
    """
    Extracts raw text context directly from a specified URL.

    Bypasses cookies banners, paywalls and heavy JS scripts to give
    the pure LLM reading content.
    """
    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily_client.extract(url)
    return response

if __name__ == "__main__":
    mcp.run(transport="http", port=5678)