from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingType
from app.core.config import get_settings
from app.core.rag_stacks import get_default_stack
from app.core.version import __version__
from pydantic import BaseModel
from typing import Optional, List
import re

settings = get_settings()
default_stack = get_default_stack()
router = APIRouter()
rag_service = RAGService(
    model_name=default_stack["llm"],
    embedding_type=EmbeddingType(default_stack["embedding_type"]),
    embedding_model=default_stack["embedding_model"],
)

# Existing chat endpoint
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response, session_id = rag_service.process_message(
            request.message,
            request.session_id
        )
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for frontend integration
class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: dict = {}

@router.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Process the question through RAG service
        response, session_id = rag_service.process_message(
            request.question,
            request.conversation_id
        )
        
        # Extract sources from the response (filename + chunk quote lines)
        sources = []
        if "Sources" in response and ":" in response.split("Sources", 1)[-1][:80]:
            parts = re.split(r"\nSources[^\n]*:\n", response, maxsplit=1)
            if len(parts) == 2:
                answer = parts[0].strip()
                sources_text = parts[1].strip()
                sources = [
                    line.strip("- ").strip()
                    for line in sources_text.split("\n")
                    if line.strip().startswith("-")
                ]
            else:
                answer = response
        else:
            answer = response
        
        generation = getattr(rag_service, "last_generation", None)
        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "session_id": session_id,
                "provider": getattr(generation, "provider", None),
                "latency_ms": getattr(generation, "latency_ms", None),
                "prompt_tokens": getattr(generation, "prompt_tokens", None),
                "completion_tokens": getattr(generation, "completion_tokens", None),
                "total_tokens": getattr(generation, "total_tokens", None),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": __version__}

@router.get("/")
async def root():
    return {
        "title": "HR Chatbot",
        "version": __version__,
        "description": "An HR chatbot using RAG and LangChain with Ollama",
    } 