from langchain_community.llms import Ollama
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self, model_name: str = None):
        # Use the configured model if none is specified
        if model_name is None:
            model_name = settings.OLLAMA_MODEL
            
        self.llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name,
            temperature=0.5,
            num_predict=150
        )
    
    def generate_response(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
    
    @staticmethod
    def get_available_models():
        """Return list of available Ollama models"""
        return settings.AVAILABLE_MODELS 