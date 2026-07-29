from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing_extensions import TypedDict, Annotated

import operator

from App.agents.ta_agent import TeachingAssistantAgent

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessageState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "use_tools"

    return "end_workflow"

class TeachingAssistantWorkflow:
    def __init__(self):
        agent = TeachingAssistantAgent()
        workflow_builder = StateGraph(MessageState)

        workflow_builder.add_node("brain_node", agent.run_llm)
        workflow_builder.add_node("action_node", agent.tool_node)

        workflow_builder.add_edge(START, "brain_node")
        workflow_builder.add_conditional_edges(
            "brain_node",
            should_continue,
            {
                "use_tools": "action_node",
                "end_workflow": END
            }
        )

        workflow_builder.add_edge("action_node", "brain_node")

        self.workflow = workflow_builder.compile()

    def run_ta_workflow(self, query: str) -> dict:
       
        result = self.workflow.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )

        return result


if __name__ == "__main__":
    query = input("Enter your query: ")
    if query.strip():
        ta_workflow = TeachingAssistantWorkflow()
        output = ta_workflow.run_ta_workflow(query)
        print("\n--- WORKFLOW OUTPUT ---")
        messages = output.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            print(messages[-1].content)
        else:
            print(output)