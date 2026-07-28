from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

class teachingagent:
    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    def run_llm1(self, query: str) -> str:
        response = self.model.invoke(query)
        return response.content
        
if __name__ ==  "__main__":
    agent = teachingagent()
    query = input("Enter your query: ")
    print(agent.run_llm1(query))