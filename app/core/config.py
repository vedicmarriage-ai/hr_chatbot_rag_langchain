from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path
from typing import List, Dict

class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCS_DIR: Path = DATA_DIR / "docs" / "Text_documents"
    CHROMA_PERSIST_DIRECTORY: Path = DATA_DIR / "chroma_db"
    
    # API settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Model settings - tuned for 8GB Mac (M3 Air)
    DEFAULT_LLM_MODEL: str = "qwen2.5:1.5b"
    DEFAULT_EMBEDDING_TYPE: str = "ollama"
    DEFAULT_EMBEDDING_MODEL: str = "bge-base"
    DEFAULT_OLLAMA_EMBEDDING_MODEL_KEY: str = "qwen3-0.6b"
    
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"
    )
    
    # LLM models available via Ollama (`ollama pull <name>`)
    AVAILABLE_MODELS: List[str] = [
        "qwen2.5:1.5b", # ~1GB - best RAG quality on 8GB RAM
        "llama3.2:1b",  # ~0.9GB - alternative small model
        "gemma2:2b",
        "tinyllama",    # ~0.3GB - small model
        "mistral:7b-instruct-v0.3-q2_K", # 2.7GB - may hang on 8GB RAM
        "llama2:7b-chat-q3_K_S",            # 2.9GB - may hang on 8GB RAM
    ]
    
    # Ollama embedding models (run via `ollama pull <name>`)
    AVAILABLE_OLLAMA_EMBEDDING_MODELS: Dict[str, Dict] = {
        "qwen3-0.6b": {
            "name": "qwen3-embedding:0.6b",
            "description": "Qwen3 embedding — best quality under 1GB, 32K context",
            "size": "639MB",
            "quality": "Excellent",
            "speed": "Fast",
            "dimensions": 1024,
            "num_ctx": 8192,
        },
        "nomic": {
            "name": "nomic-embed-text",
            "description": "Lightweight Ollama embedding model",
            "size": "274MB",
            "quality": "Good",
            "speed": "Fast",
            "dimensions": 768,
            "num_ctx": 8192,
        },
        "bge-m3": {
            "name": "bge-m3",
            "description": "Multilingual BGE-M3 — dense retrieval, 8K context",
            "size": "~1.2GB",
            "quality": "Excellent",
            "speed": "Medium",
            "dimensions": 1024,
            "num_ctx": 8192,
        },
    }
    
    # HuggingFace embedding models
    AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict] = {
        "mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "Larger model with high quality embeddings",
            "size": "420MB+",
            "quality": "High",
            "speed": "Slower on CPU",
            "dimensions": 768
        },
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Small and fast model with good performance",
            "size": "84MB",
            "quality": "Good",
            "speed": "Fast",
            "dimensions": 384
        },
        "bge-small": {
            "name": "BAAI/bge-small-en-v1.5",
            "description": "Small model optimized for semantic search",
            "size": "100MB",
            "quality": "Good",
            "speed": "Fast",
            "dimensions": 384
        },
        "bge-base": {
            "name": "BAAI/bge-base-en-v1.5",
            "description": "State-of-the-art model for RAG",
            "size": "440MB",
            "quality": "Excellent",
            "speed": "Medium",
            "dimensions": 768
        },
        "paraphrase": {
            "name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "description": "Optimized for paragraph detection",
            "size": "90MB",
            "quality": "Good",
            "speed": "Fast",
            "dimensions": 384
        }
    }
    
    # Vector store settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # RAG settings
    TOP_K_RESULTS: int = 3
    RERANK_ENABLED: bool = True
    RERANK_CANDIDATES: int = 12
    RERANK_CANDIDATES_POLICY: int = 16
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_MAX_CHUNK_CHARS: int = 800
    RERANK_LONG_CHUNK_CHARS: int = 350
    RERANK_QA_BONUS: float = 1.5
    RERANK_LEGAL_BONUS: float = 0.35
    RERANK_MAX_CHUNKS_PER_SOURCE: int = 2
    RERANK_MAX_CHUNKS_PER_SOURCE_POLICY: int = 3
    HYBRID_RETRIEVAL_ENABLED: bool = True
    # Best on current Terms/ALL sweep: always-on hybrid, denser BM25 weight
    HYBRID_POLICY_ONLY: bool = False
    HYBRID_DENSE_K: int = 12
    HYBRID_SPARSE_K: int = 12
    HYBRID_DENSE_WEIGHT: float = 0.3
    HYBRID_SPARSE_WEIGHT: float = 0.7
    SIMILARITY_THRESHOLD: float = 0.7

    # Generation / anti-hallucination
    LLM_TEMPERATURE: float = 0.0
    GROUNDING_NUMERIC_CHECK: bool = True
    GROUNDING_REFUSAL_MESSAGE: str = (
        "I don't have that information in our HR documents."
    )
    AI_CHAT_BASE_URL: str = os.getenv("AI_CHAT_BASE_URL", "")
    AI_SERVICE_TOKEN: str = os.getenv("AI_SERVICE_TOKEN", "")
    AI_CHAT_TIMEOUT_SECONDS: float = float(os.getenv("AI_CHAT_TIMEOUT_SECONDS", "30"))
    LOCAL_LLM_FALLBACK: bool = os.getenv("LOCAL_LLM_FALLBACK", "false").lower() == "true"
    LOCAL_FALLBACK_MODEL: str = os.getenv("LOCAL_FALLBACK_MODEL", "qwen2.5:1.5b")
    # -1 = keep embedding model loaded until Ollama exits
    OLLAMA_EMBED_KEEP_ALIVE: int = int(os.getenv("OLLAMA_EMBED_KEEP_ALIVE", "-1"))
    GENERATE_MAX_CHUNKS: int = int(os.getenv("GENERATE_MAX_CHUNKS", "3"))
    GENERATE_MAX_CHUNK_CHARS: int = int(os.getenv("GENERATE_MAX_CHUNK_CHARS", "400"))
    GENERATE_MAX_TOKENS: int = int(os.getenv("GENERATE_MAX_TOKENS", "1024"))
    CHAT_HISTORY_MAX_TURNS: int = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "3"))
    CHAT_HISTORY_MAX_CHARS: int = int(os.getenv("CHAT_HISTORY_MAX_CHARS", "280"))

    # API settings
    API_V1_STR: str = "/api/v1"
    
    # Environment settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    # Comma-separated browser origins. Same-origin nginx /hr-chat does not need CORS;
    # keep a default list for local Streamlit/dev and accidental cross-origin calls.
    CORS_ALLOW_ORIGINS: str = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:9101,"
        "https://www.vedicmarriage.com,https://staging.vedicmarriage.com,"
        "https://vedicmarriage.in,https://www.vedicmarriage.in",
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()


def cors_allow_origins(raw: str | None = None) -> List[str]:
    value = raw if raw is not None else get_settings().CORS_ALLOW_ORIGINS
    return [origin.strip() for origin in value.split(",") if origin.strip()]

# Create necessary directories
settings = get_settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.DOCS_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True) 