from fastapi import FastAPI
from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="HR Chatbot",
    description="An HR chatbot using RAG and LangChain with Ollama",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 