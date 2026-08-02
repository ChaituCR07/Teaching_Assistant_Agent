# Teaching Assistant & Research Agent System 

An AI-powered Teaching & Research Assistant application that utilizes **Retrieval-Augmented Generation (RAG)**, **LangGraph Workflows**, **FastAPI REST API**, and **FastMCP Tool Server** to assist students and faculty at DSATM, Bangalore. Built with LangChain, LangGraph, HuggingFace embeddings, Chroma Vector Database, DuckDuckGo, Tavily, arXiv, and Groq LLMs.

---

## 📁 Project Structure

```text
FIP/
├── App/
│   ├── agents/
│   │   ├── ta_agent.py       # TeachingAssistantAgent with bound tools (RAG retriever & web search)
│   │   └── ra_agent.py       # ResearchAssistantAgent for faculty literature review & academic research
│   ├── services/
│   │   └── rag_service.py    # RAG service: PDF loading, chunking, embedding, Chroma DB vector store & retrieval
│   └── workflows/
│       └── ta_workflow.py    # LangGraph workflow (StateGraph with brain & action nodes)
├── Assets/
│   └── Let us c - Summary.pdf # Reference PDF document for RAG indexing
├── MCP/
│   └── server.py             # FastMCP Academic & Research Tool Server (arXiv, Tavily live web & content extraction)
├── main.py                   # FastAPI application server exposing REST API endpoints (/ask-ta, /health)
├── rag_service_db/           # Local persistent Chroma DB storage
├── .env                      # Environment variables (API keys)
└── README.md                 # Project documentation
```

---

## ✨ Features

- **Teaching Assistant Agent (`ta_agent.py`)**: Answers C programming curriculum queries using RAG context from course materials or DuckDuckGo web search.
- **Research Assistant Agent (`ra_agent.py`)**: Assists faculty in literature review, searching academic literature, research repositories, and online databases.
- **FastAPI Server (`main.py`)**: REST API server providing HTTP endpoints (`/ask-ta`, `/health`, `/`) for web or frontend integrations.
- **FastMCP Tool Server (`MCP/server.py`)**: Model Context Protocol server exposing tools for searching arXiv papers, fetching paper abstracts, searching the live web via Tavily, and extracting raw webpage text.
- **Document Processing & Chunking**: Parses PDF documents into optimal text passages using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Vector Embeddings & Chroma Store**: Uses HuggingFace's `BAAI/bge-small-en-v1.5` embeddings with a persistent Chroma vector database and deterministic chunk indexing.
- **LangGraph Workflow (`ta_workflow.py`)**: State-driven workflow (`StateGraph`) handling tool selection, execution loops, and decision routing autonomously.
- **Formatted LLM Output**: Clean Markdown formatted responses tailored for academic and programming queries.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- A [Groq API Key](https://console.groq.com/)
- A [Tavily API Key](https://tavily.com/) (required for MCP Tavily web tools)

### 2. Environment Configuration
Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Dependencies
Ensure your virtual environment is active and install the required packages:

```bash
pip install fastapi uvicorn langchain langgraph langchain-community langchain-huggingface langchain-chroma langchain-groq sentence-transformers chromadb pypdf python-dotenv fastmcp arxiv tavily-python
```

---

## 🚀 Usage

### 1. Running the FastAPI Server
To start the REST API server:

```bash
uvicorn main:app --reload
```
or run with Python:
```bash
python -m main
```

**API Endpoints:**
- `GET /` - Root status check.
- `GET /health` - Health check status (`{"status": "ok"}`).
- `POST /ask-ta?query=...` - Submit a query to the Teaching Assistant Agent workflow.

### 2. Running the FastMCP Server
To start the Model Context Protocol (MCP) server for academic literature and Tavily web tools:

```bash
python -m MCP.server
```

### 3. Interacting with the Agents Directly

#### Teaching Assistant Agent (CLI)
```bash
python -m App.agents.ta_agent
```

#### Research Assistant Agent (CLI)
```bash
python -m App.agents.ra_agent
```

### 4. Running the LangGraph TA Workflow
To test the state-driven LangGraph workflow:

```bash
python -m App.workflows.ta_workflow
```

### 5. Indexing PDF & Testing Retrieval (RAG Service)
To test vector DB retrieval directly or re-index documents:

```bash
python -m App.services.rag_service
```

> **Note**: To re-index a new or updated document, uncomment `service.process_pdf_document(reset=True)` inside `App/services/rag_service.py`.

---

## ⚡ Customization

- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` inside `EmbeddingService.process_pdf_document()` in `App/services/rag_service.py` to tune passage granularity.
- **Retrieval Count (`k`)**: Pass `k=<number>` to `retrieve_from_pdf(query, k=5)` to retrieve more or fewer context passages per query.
- **LLM Model**: Adjust `model="openai/gpt-oss-20b"` or temperature settings inside `TeachingAssistantAgent` in `App/agents/ta_agent.py` or `ResearchAssistantAgent` in `App/agents/ra_agent.py`.
- **MCP Tools**: Extend or configure new academic and web tools in `MCP/server.py` using `@mcp.tool()`.
