from fastapi import FastAPI
from app.api.routes import router
from app.core.config import cors_allow_origins, get_settings
from app.core.version import __version__
from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()

app = FastAPI(
    title="HR Chatbot",
    description="An HR chatbot using RAG and LangChain with Ollama",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 