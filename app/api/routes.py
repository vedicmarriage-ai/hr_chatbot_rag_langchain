from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingType
from app.core.config import get_settings
from app.core.rag_stacks import get_default_stack
from app.core.version import __version__
from pydantic import BaseModel
from typing import Optional, List

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
        
        # Extract sources from the response
        sources = []
        if "Sources:" in response:
            # Split response to separate answer from sources
            parts = response.split("Sources:")
            answer = parts[0].strip()
            sources_text = parts[1].strip()
            # Extract source filenames
            sources = [line.strip("- ").strip() for line in sources_text.split("\n") if line.strip().startswith("-")]
        else:
            answer = response
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "confidence": 0.9,  # You can implement confidence scoring
                "model": "ollama",
                "session_id": session_id,
                "tokens_used": len(answer.split())  # Approximate token count
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