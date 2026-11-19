from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from app.core.config import get_settings
from app.core.rag_stacks import is_valid_combination
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.services.document_service import DocumentService
from app.services.answer_generation import generate_from_docs
from app.services.reranker_service import rerank_documents
from app.services.hybrid_retriever import BM25Index, hybrid_search
from app.services.query_intent import should_use_hybrid_retrieval
from app.services.grounding import enforce_numeric_grounding
from typing import Any, List, Optional
from pathlib import Path
import re
import uuid
import os

from langchain_core.messages import AIMessage

settings = get_settings()

# Max chars of chunk text shown in the Sources footer
_SOURCE_QUOTE_CHARS = 180


def _chunk_label(doc: Document) -> str:
    """Human-readable chunk id from metadata (faq_id / legal_id / section)."""
    meta = doc.metadata or {}
    for key in ("faq_id", "legal_id", "legal_heading", "section"):
        val = meta.get(key)
        if val:
            return str(val)
    return ""


def _chunk_quote(text: str, max_chars: int = _SOURCE_QUOTE_CHARS) -> str:
    """Single-line excerpt so citations show evidence, not only filenames."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def format_source_citations(source_docs: List[Document]) -> str:
    """Cite each retrieved chunk with filename + optional id + quote."""
    if not source_docs:
        return ""
    lines: List[str] = []
    seen: set[str] = set()
    for doc in source_docs:
        src_path = doc.metadata.get("source", "Unknown")
        name = Path(str(src_path)).name if src_path != "Unknown" else "Unknown"
        label = _chunk_label(doc)
        quote = _chunk_quote(doc.page_content)
        key = f"{name}|{label}|{quote}"
        if key in seen:
            continue
        seen.add(key)
        if label:
            lines.append(f'- {name} [{label}]: "{quote}"')
        else:
            lines.append(f'- {name}: "{quote}"')
    return "\n\nSources (retrieved excerpts — not proof the answer is in them):\n" + "\n".join(
        lines
    )


class RerankRetriever(BaseRetriever):
    """Fetch a wide candidate set (hybrid optional), rerank, return top-k."""

    vectorstore: Any
    bm25_index: Optional[Any] = None
    k: int = 5
    fetch_k: int = 15
    fetch_k_policy: int = 35
    use_reranker: bool = True
    use_hybrid: bool = True
    hybrid_policy_only: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        from app.services.query_intent import is_policy_query

        policy = is_policy_query(query)
        fetch_k = self.fetch_k_policy if policy else self.fetch_k
        use_hybrid = self.use_hybrid and should_use_hybrid_retrieval(
            query, self.hybrid_policy_only
        )
        docs = hybrid_search(
            vectorstore=self.vectorstore,
            bm25_index=self.bm25_index,
            query=query,
            fetch_k=fetch_k,
            use_hybrid=use_hybrid,
        )
        if not self.use_reranker:
            return docs[: self.k]
        return rerank_documents(query, docs, self.k)


class RAGService:
    def __init__(
        self,
        model_name: str = None,
        embedding_type: EmbeddingType = EmbeddingType.OLLAMA,
        embedding_model: str = None,
    ):
        if embedding_model is None:
            embedding_model = (
                settings.DEFAULT_OLLAMA_EMBEDDING_MODEL_KEY
                if embedding_type == EmbeddingType.OLLAMA
                else settings.DEFAULT_EMBEDDING_MODEL
            )

        llm_name = model_name or settings.OLLAMA_MODEL
        if not is_valid_combination(
            llm_name,
            embedding_type.value,
            embedding_model,
        ):
            raise ValueError(
                f"Invalid RAG stack: llm={llm_name}, "
                f"embedding_type={embedding_type.value}, "
                f"embedding_model={embedding_model}. "
                f"Choose a combination from app.core.rag_stacks.RAG_STACKS."
            )

        self.llm_service = LLMService(model_name=model_name)
        self.embedding_service = EmbeddingService(
            embedding_type=embedding_type,
            model_key=embedding_model
        )
        self.document_service = DocumentService()
        
        vector_store_subdir = EmbeddingService.get_vector_store_subdir(
            embedding_type, embedding_model
        )
        vector_store_path = settings.CHROMA_PERSIST_DIRECTORY / vector_store_subdir
        
        if not os.path.exists(vector_store_path):
            raise ValueError(
                f"Vector store not found at {vector_store_path}. "
                f"Please run create_vector_store.py first for {embedding_type} - {embedding_model}."
            )
        
        print(f"🔍 Loading vector store from: {vector_store_path}")
        
        self.vectorstore = Chroma(
            persist_directory=str(vector_store_path),
            embedding_function=self.embedding_service.embeddings
        )

        self.bm25_index = None
        if settings.HYBRID_RETRIEVAL_ENABLED:
            try:
                self.bm25_index = BM25Index.from_vectorstore(self.vectorstore)
                print(
                    f"🔍 BM25 index ready ({len(self.bm25_index.documents)} chunks)"
                )
            except Exception as exc:
                print(f"⚠️ BM25 index unavailable, dense-only retrieval: {exc}")
        
        self.qa_template = """You are an HR assistant for Vedic Marriage. Answer ONLY using the Context below. Be concise and direct.

Rules:
- Use ONLY facts that appear in the Context. Do not use outside knowledge.
- If the Context does not contain the answer, reply exactly: "I don't have that information in our HR documents."
- Do NOT invent numbers, counts, percentages, prices, dates, or statistics. Only state a number if it appears verbatim in the Context.
- Do not guess or generalize (e.g. typical industry figures). If unsure, use the refusal sentence above.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        self.QA_PROMPT = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        self.retriever = RerankRetriever(
            vectorstore=self.vectorstore,
            bm25_index=self.bm25_index,
            k=settings.TOP_K_RESULTS,
            fetch_k=settings.RERANK_CANDIDATES,
            fetch_k_policy=settings.RERANK_CANDIDATES_POLICY,
            use_reranker=settings.RERANK_ENABLED,
            use_hybrid=settings.HYBRID_RETRIEVAL_ENABLED,
            hybrid_policy_only=settings.HYBRID_POLICY_ONLY,
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_service.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
    
    def _local_fallback(self, prompt: str) -> str:
        if getattr(self, "_fallback_llm", None) is None:
            self._fallback_llm = LLMService(model_name=settings.LOCAL_FALLBACK_MODEL)
        return self._fallback_llm.generate_response(prompt)

    def _chat_history_text(self) -> str:
        messages = getattr(getattr(self.memory, "chat_memory", None), "messages", []) or []
        lines = []
        for message in messages[-4:]:
            content = getattr(message, "content", "")
            if content:
                lines.append(str(content)[:500])
        return "\n".join(lines)

    def process_message(self, message: str, session_id: str = None) -> tuple[str, str]:
        if not session_id:
            session_id = str(uuid.uuid4())

        source_docs = self.retriever.invoke(message)
        generated = generate_from_docs(
            question=message,
            docs=source_docs,
            prompt_template=self.QA_PROMPT,
            chat_history=self._chat_history_text(),
            refusal=settings.GROUNDING_REFUSAL_MESSAGE,
            local_generate=self._local_fallback,
            fallback_enabled=settings.LOCAL_LLM_FALLBACK,
        )
        self.last_generation = generated
        answer = generated.text

        if generated.provider != "none":
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(answer)

        if settings.GROUNDING_NUMERIC_CHECK:
            answer, bad_claims = enforce_numeric_grounding(
                answer,
                docs=source_docs,
                refusal=settings.GROUNDING_REFUSAL_MESSAGE,
            )
            if bad_claims:
                print(
                    f"⚠️ Ungrounded numeric claim(s) blocked: {bad_claims}"
                )
                self._replace_last_ai_memory(answer)

        answer = answer + format_source_citations(source_docs)

        return answer, session_id

    def _replace_last_ai_memory(self, content: str) -> None:
        """Keep conversation memory aligned after a post-generation refusal."""
        try:
            msgs = self.memory.chat_memory.messages
            if msgs and isinstance(msgs[-1], AIMessage):
                msgs[-1].content = content
        except Exception:
            pass
    
    def update_knowledge_base(self, documents_directory: str):
        """Update the knowledge base with new documents"""
        self.vectorstore = self.document_service.process_documents(documents_directory)
        self.bm25_index = None
        if settings.HYBRID_RETRIEVAL_ENABLED:
            try:
                self.bm25_index = BM25Index.from_vectorstore(self.vectorstore)
            except Exception:
                self.bm25_index = None
        self.retriever = RerankRetriever(
            vectorstore=self.vectorstore,
            bm25_index=self.bm25_index,
            k=settings.TOP_K_RESULTS,
            fetch_k=settings.RERANK_CANDIDATES,
            fetch_k_policy=settings.RERANK_CANDIDATES_POLICY,
            use_reranker=settings.RERANK_ENABLED,
            use_hybrid=settings.HYBRID_RETRIEVAL_ENABLED,
            hybrid_policy_only=settings.HYBRID_POLICY_ONLY,
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_service.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
    
    @staticmethod
    def get_available_models():
        """Return list of available models"""
        return LLMService.get_available_models()
    
    @staticmethod
    def get_available_embedding_types():
        """Return list of available embedding types"""
        return EmbeddingService.get_available_embedding_types()
