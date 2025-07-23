from fastapi import FastAPI
from backend.routers.endpoints import router as chat_router
from backend.routers.endpoints import router as cli_router

app = FastAPI(title="Supervisor Agent", description="""A Supervisor agent that understands the task and manages a 
team of Agents for dealing with that task.""")

# Register routers
app.include_router(chat_router)
app.include_router(cli_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)