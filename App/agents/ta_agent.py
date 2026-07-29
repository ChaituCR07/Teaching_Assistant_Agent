from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.messages import SystemMessage, ToolMessage, AIMessage

import os
from dotenv import load_dotenv

from App.services.rag_service import EmbeddingService

load_dotenv()

@tool
def retriever(query: str) -> list:
    """
    RAG Service Tool (PDF Vector Store): Retrieves internal course notes, DSATM syllabus topics, and textbook definitions from the 'Let Us C' course PDF database.
    Use this tool when the query specifically asks about DSATM textbook content, course syllabus, or internal PDF notes.
    """
    embedding_service = EmbeddingService()
    context = embedding_service.retrieve_from_pdf(query)

    return context

@tool
def web_search(query: str) -> str:
    """
    Web Search Engine Tool (DuckDuckGo Search): Searches the live web for general programming concepts, language comparisons (e.g. C vs Python, C vs C++), real-world advantages/disadvantages, modern software practices, latest C standards (C23), or external technical documentation.
    Use this tool for general web search queries, comparisons, or topics not limited to the local textbook.
    """
    search_run = DuckDuckGoSearchRun()
    return search_run.invoke(query)

class TeachingAssistantAgent:
    def __init__(self):
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

        tools_list = [retriever, web_search]
        self.tools_by_name = {tool.name: tool for tool in tools_list}
        self.llm_with_tools = llm.bind_tools(tools_list)

    def run_llm(self, state: dict) -> dict: # Brain
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                    content="""
                    You are an intelligent Teaching Assistant for C Programming at DSATM, Bangalore.

                    Tool Selection Strategy:
                    - Analyze the user's query and dynamically select the appropriate tool:
                      1. Use `web_search` (Search Engine) for general programming queries, language comparisons (e.g., C vs Python, C vs C++), real-world advantages/disadvantages, modern development trends, or external web information.
                      2. Use `retriever` (RAG Service) when the query specifically targets the internal DSATM textbook ('Let Us C'), course syllabus, or internal PDF notes.

                    - If the user asks about other programming languages in comparison to C (e.g., "comparison of C with Python"), use `web_search` to provide a complete comparison, highlighting the C perspective as a Teaching Assistant.
                    - If a query is completely unrelated to technology/programming, politely refuse to answer.

                    Output Format:
                    1. [Answer / Point 1]
                    2. [Answer / Point 2]
                    3. [Answer / Point 3]
                    """
                        )
                    ]
                    + state['messages']
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    def tool_node(self, state):
        """Performs tool calls and logs tool selection"""

        result = []

        for tool_call in state['messages'][-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {}).copy()
            if 'self' in args:
                del args['self']

            if tool_name == "retriever":
                print(f"\n🔍 [Tool Selected]: RAG Service (PDF Vector DB) | Query: '{args.get('query', '')}'")
            elif tool_name in ["web_search", "duckduckgo_search"]:
                print(f"\n🌐 [Tool Selected]: Search Engine (DuckDuckGo) | Query: '{args.get('query', '')}'")
            else:
                print(f"\n🛠️ [Tool Selected]: {tool_name} | Query: '{args.get('query', '')}'")

            tool = self.tools_by_name[tool_name]
            observation = tool.invoke(args)

            if isinstance(observation, list):
                content_string = "\n".join(observation)
            else:
                content_string = str(observation)

            result.append(
                ToolMessage(
                    content=content_string,
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": result}

if __name__ == "__main__":
    from App.workflows.ta_workflow import TeachingAssistantWorkflow

    query = input("Enter your query: ")
    if query.strip():
        ta_workflow = TeachingAssistantWorkflow()
        output = ta_workflow.run_ta_workflow(query)
        print("\n--- AGENT RESPONSE ---")
        messages = output.get("messages", [])
        if messages and hasattr(messages[-1], "content") and messages[-1].content:
            print(messages[-1].content)
        else:
            print(output)