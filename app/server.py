from fastapi import FastAPI
from app.api.routes import router
from app.core.config import get_settings
from app.core.version import __version__

settings = get_settings()

app = FastAPI(
    title="HR Chatbot API",
    description="An HR chatbot using RAG and LangChain with Ollama",
    version=__version__,
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT) 