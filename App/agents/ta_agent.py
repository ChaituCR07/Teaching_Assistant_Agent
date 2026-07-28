from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

def run_llm(query: str) -> str:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    response = llm.invoke(query)
    
    return response.content

if __name__ == "__main__":
    query = input("Ask me: ")
    print(run_llm(query))