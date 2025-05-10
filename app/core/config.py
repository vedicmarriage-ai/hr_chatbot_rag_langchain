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
    
    # Model settings
    DEFAULT_LLM_MODEL: str = "tinyllama"
    DEFAULT_EMBEDDING_TYPE: str = "huggingface"
    DEFAULT_EMBEDDING_MODEL: str = "bge-base"  # Using BGE base as default due to its excellent RAG performance
    
    # Ollama settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-v0.3-q2_K")
    
    # Available models for local development
    AVAILABLE_MODELS: List[str] = [
        "mistral:7b-instruct-v0.3-q2_K",
        "llama2:7b-text-q2_K",
        "llama2:7b-chat-q3_K_S",
        "tinyllama"
    ]
    
    # Available embedding models configuration
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
    TOP_K_RESULTS: int = 4
    SIMILARITY_THRESHOLD: float = 0.7
    
    
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # Environment settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Create necessary directories
settings = get_settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.DOCS_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True) 