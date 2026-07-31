from fastapi import FastAPI

from App.workflows.ta_workflow import TeachingAssistantWorkflow

app = FastAPI(title="DSATM Assistant", version="0.0.1")

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