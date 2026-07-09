from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from enum import Enum
from typing import Dict, Any, Optional

settings = get_settings()

class EmbeddingType(Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

class EmbeddingModel:
    def __init__(self, name: str, description: str, size: str, quality: str, speed: str, dimensions: int):
        self.name = name
        self.description = description
        self.size = size
        self.quality = quality
        self.speed = speed
        self.dimensions = dimensions

class EmbeddingService:
    def __init__(
        self,
        embedding_type: EmbeddingType = EmbeddingType.OLLAMA,
        model_key: str = None,
        device: str = "cpu",
        batch_size: int = 32
    ):
        self.embedding_type = embedding_type
        self.device = device
        self.batch_size = batch_size
        
        if embedding_type == EmbeddingType.OLLAMA:
            self.model_key = model_key or settings.DEFAULT_OLLAMA_EMBEDDING_MODEL_KEY
            model_config = settings.AVAILABLE_OLLAMA_EMBEDDING_MODELS[self.model_key]
            self.embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_config["name"],
                temperature=0,
                num_thread=8,
                num_gpu=-1,
                num_ctx=model_config.get("num_ctx", 8192),
            )
        else:
            self.model_key = model_key or settings.DEFAULT_EMBEDDING_MODEL
            model_config = settings.AVAILABLE_EMBEDDING_MODELS[self.model_key]
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_config["name"],
                model_kwargs={'device': device},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': batch_size
                }
            )
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)
    
    def get_embedding(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
    
    @staticmethod
    def get_available_embedding_types():
        """Return list of available embedding types"""
        return [e.value for e in EmbeddingType]
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict]:
        """Return dictionary of available HuggingFace embedding models"""
        return settings.AVAILABLE_EMBEDDING_MODELS

    @staticmethod
    def get_available_ollama_models() -> Dict[str, Dict]:
        """Return dictionary of available Ollama embedding models"""
        return settings.AVAILABLE_OLLAMA_EMBEDDING_MODELS

    @staticmethod
    def get_vector_store_subdir(embedding_type: EmbeddingType, model_key: str) -> str:
        if embedding_type == EmbeddingType.OLLAMA:
            return f"ollama_{model_key}"
        return f"hf_{model_key}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.embedding_type == EmbeddingType.OLLAMA:
            model_config = settings.AVAILABLE_OLLAMA_EMBEDDING_MODELS[self.model_key]
            return {
                "type": "ollama",
                "model": model_config["name"],
                "description": model_config["description"],
                "size": model_config["size"],
                "quality": model_config["quality"],
                "speed": model_config["speed"],
                "dimensions": model_config["dimensions"],
            }
        model_config = settings.AVAILABLE_EMBEDDING_MODELS[self.model_key]
        return {
            "type": "huggingface",
            "model": model_config["name"],
            "description": model_config["description"],
            "size": model_config["size"],
            "quality": model_config["quality"],
            "speed": model_config["speed"],
            "dimensions": model_config["dimensions"]
        }
