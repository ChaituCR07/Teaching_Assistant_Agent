# Teaching Assistant Agent (FIP)

An AI-powered Teaching Assistant application that utilizes **Retrieval-Augmented Generation (RAG)** and **LangGraph Workflows** to answer student queries for the C Programming course at DSATM, Bangalore. Built with LangChain, LangGraph, HuggingFace embeddings, Chroma Vector Database, DuckDuckGo web search, and Groq LLMs.

---

## 📁 Project Structure

```text
FIP/
├── App/
│   ├── agents/
│   │   └── ta_agent.py       # TeachingAssistantAgent with bound tools (RAG retriever & web search)
│   ├── services/
│   │   └── rag_service.py    # RAG service: PDF loading, chunking, embedding, vector store & retrieval
│   └── workflows/
│       └── ta_workflow.py   # LangGraph workflow (StateGraph with brain & action nodes)
├── Assets/
│   └── Let us c - Summary.pdf # Reference PDF document for RAG indexing
├── rag_service_db/           # Local persistent Chroma DB storage
├── .env                      # Environment variables (API keys)
└── README.md                 # Project documentation
```

---

## ✨ Features

- **Document Processing & Chunking**: Parses PDF documents into optimal text passages using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Vector Embeddings**: Uses HuggingFace's `BAAI/bge-small-en-v1.5` embeddings for fast and semantic representation.
- **Chroma Vector Store**: Manages persistent vector storage with collection reset capabilities and deterministic chunk indexing to avoid duplication.
- **RAG & Web Search Tools**: Equips the agent with a custom PDF `retriever` tool for course materials and `DuckDuckGoSearchRun` for supplementary web searches.
- **LangGraph Workflow**: State-driven workflow (`ta_workflow.py`) built with `StateGraph`, enabling autonomous tool-calling loops and decision routing.
- **Interactive User Input**: Accepts real-time queries dynamically via `input()` across agent, workflow, and RAG service modules.
- **Formatted LLM Output**: Displays clean Markdown answer responses instead of raw message metadata objects.
- **DSATM Curriculum Context**: Configured system prompt tailored to the DSATM C Programming curriculum.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- A [Groq API Key](https://console.groq.com/)

### 2. Environment Configuration
Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Dependencies
Ensure your virtual environment is active and install the required packages:

```bash
pip install langchain langgraph langchain-community langchain-huggingface langchain-chroma langchain-groq sentence-transformers chromadb pypdf python-dotenv
```

---

## 🚀 Usage

### 1. Interacting with the Teaching Assistant Agent
To run the interactive agent with bound tools (PDF retriever & Web Search):

```bash
python -m App.agents.ta_agent
```

### 2. Running the LangGraph TA Workflow
To run the state-driven LangGraph workflow:

```bash
python -m App.workflows.ta_workflow
```

### 3. Indexing PDF & Testing Retrieval (RAG Service)
To test vector DB retrieval directly or re-index documents:

```bash
python -m App.services.rag_service
```

> **Note**: To re-index a new or updated document, uncomment `service.process_pdf_document(reset=True)` inside `App/services/rag_service.py`.

---

## ⚡ Customization

- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` inside `EmbeddingService.process_pdf_document()` in `App/services/rag_service.py` to tune passage granularity.
- **Retrieval Count (`k`)**: Pass `k=<number>` to `retrieve_from_pdf(query, k=5)` to retrieve more or fewer context passages per query.
- **LLM Model**: Adjust `model="openai/gpt-oss-20b"` or temperature settings inside `TeachingAssistantAgent` in `App/agents/ta_agent.py`.
