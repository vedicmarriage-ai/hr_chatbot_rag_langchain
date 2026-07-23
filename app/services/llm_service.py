from langchain_ollama import OllamaLLM
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self, model_name: str = None):
        # Use the configured model if none is specified
        if model_name is None:
            model_name = settings.OLLAMA_MODEL
            
        # Memory-efficient settings for 8GB Mac; larger models need more RAM
        small_models = ("tinyllama", "qwen", "llama3.2", "gemma2", "gemma")
        temperature = float(settings.LLM_TEMPERATURE)
        if any(tag in model_name.lower() for tag in small_models):
            llm_params = {
                "temperature": temperature,
                "num_predict": 512,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
            }
        else:
            llm_params = {
                "temperature": temperature,
                "num_predict": 1024,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
            }
            
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name,
            **llm_params
        )
    
    def generate_response(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
    
    @staticmethod
    def get_available_models():
        """Return list of available Ollama models"""
        return settings.AVAILABLE_MODELS
