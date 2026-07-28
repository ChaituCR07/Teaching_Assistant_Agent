# Teaching Assistant Agent (FIP)

An AI-powered Teaching Assistant application that utilizes **Retrieval-Augmented Generation (RAG)** to index educational materials (PDFs) and answer student queries accurately using LangChain, HuggingFace embeddings, Chroma Vector Database, and Groq LLMs.

---

## 📁 Project Structure

```text
FIP/
├── App/
│   ├── agents/
│   │   ├── ta_agent.py       # Basic Groq LLM agent implementation
│   │   └── ta_agent1.py      # Object-oriented TeachingAgent implementation
│   ├── services/
│   │   └── rag_service.py    # RAG service: PDF loading, chunking, embedding, vector store & retrieval
│   └── workflows/            # Workflow definitions (extensible)
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
- **Deduplicated Context Retrieval**: Retrieves top relevant contexts while filtering out duplicate passages.
- **Groq LLM Integration**: Uses Groq's high-speed inference engine (`langchain-groq`) to generate responses to queries.

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
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-groq sentence-transformers chromadb pypdf python-dotenv
```

---

## 🚀 Usage

### 1. Indexing PDF & Testing Retrieval (RAG Service)
To process the PDF document, index its chunks into Chroma DB, and perform sample retrieval queries:

```bash
python -m App.services.rag_service
```

> **Note**: To re-index a new or updated document, uncomment `service.process_pdf_document(reset=True)` inside `App/services/rag_service.py`.

### 2. Interacting with the Teaching Assistant Agent
To run the Groq-powered Teaching Assistant interactive agent:

```bash
python -m App.agents.ta_agent
```
or
```bash
python -m App.agents.ta_agent1
```

---

## ⚡ Customization

- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` inside `EmbeddingService.process_pdf_document()` in `App/services/rag_service.py` to tune passage granularity.
- **Retrieval Count (`k`)**: Pass `k=<number>` to `retrieve_from_pdf(query, k=5)` to retrieve more or fewer context passages per query.
- **LLM Model**: Adjust `model="openai/gpt-oss-120b"` or temperature settings inside `App/agents/ta_agent1.py`.
