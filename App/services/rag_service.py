from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    DEFAULT_FILE_PATH = "Assets/Let us c - Summary.pdf"

    def __init__(self):
        embeddings=HuggingFaceEmbeddings(
            model_name= "BAAI/bge-small-en-v1.5",
            show_progress=True
        )

        self.vector_store=Chroma(
            persist_directory="./rag_service_db",
            collection_name="teaching_assistant_collection",
            embedding_function=embeddings
        )

    def process_pdf_document(self, file_path=None, reset=True):
        if file_path is None:
            file_path = self.DEFAULT_FILE_PATH
        loader = PyPDFLoader(file_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_documents = splitter.split_documents(document)

        if reset:
            try:
                existing_ids = self.vector_store._collection.get()['ids']
                if existing_ids:
                    self.vector_store._collection.delete(ids=existing_ids)
            except Exception:
                pass

        # Use deterministic chunk IDs to prevent duplicate accumulation in Chroma
        ids = [f"chunk_{i}" for i in range(len(split_documents))]
        self.vector_store.add_documents(split_documents, ids=ids)
        print(f"Processed and indexed {len(split_documents)} chunks.")

    def retrieve_from_pdf(self, query: str, k: int = 3):
        retriever = self.vector_store.as_retriever(search_kwargs={'k': k})
        response = retriever.invoke(query)

        # Deduplicate page content while preserving order
        unique_chunks = []
        for doc in response:
            if doc.page_content not in unique_chunks:
                unique_chunks.append(doc.page_content)

        return unique_chunks


if __name__ == "__main__":
    service = EmbeddingService()
    # Re-process document if needed:
    # service.process_pdf_document(reset=True)

    query = input("Enter your query: ")
    if query.strip():
        chunks = service.retrieve_from_pdf(query, k=3)

        print(f"\n--- RETRIEVED CHUNKS FOR QUERY: '{query}' ---")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- CHUNK {i} ---")
            print(chunk)