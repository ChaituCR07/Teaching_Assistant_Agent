from fastapi import FastAPI

from App.workflows.ta_workflow import TeachingAssistantWorkflow

from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp_base_url = "http://127.0.0.1:5678/mcp"

    app.state.mcp_tools = []
    app.state.mcp_session = None
    app.state.transport_cm = None
    app.state.mcp_session_cm = None
    
    try:
        transport_cm = streamable_http_client(mcp_base_url)
        read_stream, write_stream, get_session_id = await transport_cm.__aenter__()

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()

        await session.initialize()

        app.state.mcp_session = session
        app.state.transport_cm = transport_cm
        app.state.mcp_session_cm = session_cm
        app.state.mcp_get_session_id = get_session_id

        response = await session.list_tools()
        raw_tools = response.tools if hasattr(response, "tools") else []

        app.state.mcp_tools = [
            convert_mcp_tool_to_langchain_tool(session, tool)
            for tool in raw_tools
        ]

        print(f"Successfully loaded {len(app.state.mcp_tools)} tools.")
        print(f"MCP session id: {get_session_id}")
    except Exception as e:
        print(f"CRITICIAL: Failed to connect to the MCP {e}")

    try:
        yield
    finally:
        if app.state.mcp_session_cm is not None:
            await app.state.mcp_session_cm.__aexit__(None, None, None)
        if app.state.transport_cm is not None:
            await app.state.transport_cm.__aexit__(None, None, None)
    
app = FastAPI(title="DSATM Assistant", version="0.0.1", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "DSATM Assistant Server is booted up......."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask-ta")
def ask_ta_agent(query: str):
    ta_workflow = TeachingAssistantWorkflow()
    response = ta_workflow.run_ta_workflow(query)
    return response["messages"][-1].content

@app.get("/mcp/tools")
async def fetch_mcp_tools():
    return {"tools": app.state.mcp_tools}