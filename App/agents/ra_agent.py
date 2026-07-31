from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, ToolMessage, AIMessage

import os
from dotenv import load_dotenv

load_dotenv()

class ResearchAssistantAgent:
    def __init__(self):
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.tools_list = []

        self.llm_with_tools = llm.bind_tools(self.tools_list)


        self.tools_by_name = {tool.name: tool for tool in self.tools_list}

    def run_llm(self, state: dict) -> dict: # Brain
        result = {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                    content=f"""
                    You are a helpful Research Assistant Agent for faculty assistance at DSATM, Bangalore.
                    
                    Your task is to look for complex topics, references from academic and online
                    databases and generate high quality reports that can help the faculty write
                    their literature review.

                    Follow this process:

                    1. You have tools that can help with web searches. Use them to find links or abstract content.
                    2. Refer to articles & documentations.
                    3. You have access to research repositories, use them to extract paper content, details
                    and generate high quality reports that can help the faculty write their literature review.
                    
                    CRITICAL INSTRUCTION:
                     - Rely ONLY on your provided tools for real-world factual claims.
                     - You must ONLY execute a tools if its name EXACTLY matches one of the valid tool names: {self.tools_list}.
                    - DO NOT hallucinate or guess a short name like `web search` if the server provides an explicit name.
                    """
                        )
                    ]
                    + state['messages']
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

        print("\n\n-----------------------AI MESSAGE -------------------\n\n")
        print(result)

        return result

    def tool_node(self, state: dict) -> dict: # Action
        result = []

        for tool_call in state['messages'][-1].tool_calls:

            if 'self' in tool_call["args"]:
                del tool_call["args"]['self']

            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

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

        print("\n\n-----------------------Tool MESSAGE -------------------\n\n")
        print(result)
        
        return {"messages": result}

if __name__ ==  "__main__":
    # query = input("Enter your query: ")
    # print(run_llm(query))

    # chunk = retriever.invoke("What is a pointer?")
    # print("\n\nCHUNKS RETRIEVED ")
    # print(chunk)
    # print("\n\n")

    # search = web_search.invoke("LA Olympics 2028")
    # print(search)
    pass