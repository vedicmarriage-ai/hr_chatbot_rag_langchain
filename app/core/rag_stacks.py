"""Curated LLM + embedding combinations validated for 8GB Mac."""

from typing import Dict, List, Optional, Tuple

# Each stack is a tested combination — only these are exposed in the UI.
RAG_STACKS: List[Dict] = [
    {
        "id": "qwen-qwen3",
        "label": "Qwen 2.5 1.5B + Qwen3 Embed (Recommended)",
        "llm": "qwen2.5:1.5b",
        "embedding_type": "ollama",
        "embedding_model": "qwen3-0.6b",
    },
    {
        "id": "llama-qwen3",
        "label": "Llama 3.2 1B + Qwen3 Embed",
        "llm": "llama3.2:1b",
        "embedding_type": "ollama",
        "embedding_model": "qwen3-0.6b",
    },
    {
        "id": "llama-nomic",
        "label": "Llama 3.2 1B + Nomic Embed",
        "llm": "llama3.2:1b",
        "embedding_type": "ollama",
        "embedding_model": "nomic",
    },
    {
        "id": "gemma-qwen3",
        "label": "Gemma 2 2B + Qwen3 Embed",
        "llm": "gemma2:2b",
        "embedding_type": "ollama",
        "embedding_model": "qwen3-0.6b",
    },
    {
        "id": "gemma-nomic",
        "label": "Gemma 2 2B + Nomic Embed",
        "llm": "gemma2:2b",
        "embedding_type": "ollama",
        "embedding_model": "nomic",
    },
    {
        "id": "qwen-bge-m3",
        "label": "Qwen 2.5 1.5B + BGE-M3 Embed",
        "llm": "qwen2.5:1.5b",
        "embedding_type": "ollama",
        "embedding_model": "bge-m3",
    },
    {
        "id": "llama-bge-m3",
        "label": "Llama 3.2 1B + BGE-M3 Embed",
        "llm": "llama3.2:1b",
        "embedding_type": "ollama",
        "embedding_model": "bge-m3",
    },
    {
        "id": "qwen-hf-bge",
        "label": "Qwen 2.5 1.5B + BGE Base (HuggingFace)",
        "llm": "qwen2.5:1.5b",
        "embedding_type": "huggingface",
        "embedding_model": "bge-base",
    },
    {
        "id": "llama-hf-bge",
        "label": "Llama 3.2 1B + BGE Base (HuggingFace)",
        "llm": "llama3.2:1b",
        "embedding_type": "huggingface",
        "embedding_model": "bge-base",
    },
]

DEFAULT_STACK_ID = "qwen-qwen3"


def get_stack_by_id(stack_id: str) -> Optional[Dict]:
    for stack in RAG_STACKS:
        if stack["id"] == stack_id:
            return stack
    return None


def get_default_stack() -> Dict:
    return get_stack_by_id(DEFAULT_STACK_ID) or RAG_STACKS[0]


def is_valid_combination(
    llm: str, embedding_type: str, embedding_model: str
) -> bool:
    return any(
        s["llm"] == llm
        and s["embedding_type"] == embedding_type
        and s["embedding_model"] == embedding_model
        for s in RAG_STACKS
    )


def get_stacks_for_llm(llm: str) -> List[Dict]:
    return [s for s in RAG_STACKS if s["llm"] == llm]


def get_stack_labels() -> List[Tuple[str, str]]:
    return [(s["label"], s["id"]) for s in RAG_STACKS]
