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
    RAG Service Tool: Retrieves internal course materials, syllabus, and concepts on C programming
    taught in the DSATM curriculum (from the 'Let Us C' textbook reference database).
    Use this tool for queries about C fundamentals, pointers, arrays, structures, loops, and DSATM course content.
    """
    embedding_service = EmbeddingService()
    context = embedding_service.retrieve_from_pdf(query)

    return context

@tool
def web_search(query: str) -> str:
    """
    Web Search Engine Tool: Searches the web live for external technical information, latest C language standards,
    programming news, library documentations, or real-time topics not present in the DSATM reference textbook.
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
                    - Analyze the user's query and dynamically choose the most appropriate tool:
                      1. Use `retriever` (RAG Service) if the query relates to core C programming concepts, DSATM course curriculum, pointers, memory allocation, control flow, functions, or textbook topics.
                      2. Use `web_search` (Search Engine) if the query requires live web information, recent C language updates (e.g. C23 features), external programming news, or documentation not in the textbook.

                    - Maintain your role strictly as a C programming teaching assistant. If a query is completely irrelevant to programming, politely refuse to answer.

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