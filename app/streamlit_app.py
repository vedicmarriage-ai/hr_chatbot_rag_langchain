import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingType, EmbeddingService
from app.core.config import get_settings
from app.core.rag_stacks import RAG_STACKS, DEFAULT_STACK_ID, get_stack_by_id

settings = get_settings()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None

def initialize_rag_service(model_name: str, embedding_type: str, embedding_model: str):
    """Initialize the RAG service with selected model and embedding type"""
    embedding_type_enum = EmbeddingType(embedding_type)
    return RAGService(
        model_name=model_name,
        embedding_type=embedding_type_enum,
        embedding_model=embedding_model
    )

def display_model_info(embedding_service: EmbeddingService, llm: str):
    """Display information about the selected stack"""
    model_info = embedding_service.get_model_info()
    
    st.sidebar.markdown("### Active Stack")
    st.sidebar.markdown(f"**LLM:** `{llm}`")
    st.sidebar.markdown(f"**Embedding type:** {model_info['type']}")
    st.sidebar.markdown(f"**Embedding model:** {model_info['model']}")
    st.sidebar.markdown(f"**Description:** {model_info['description']}")
    
    if model_info['type'] in ('huggingface', 'ollama'):
        st.sidebar.markdown(f"**Size:** {model_info['size']}")
        st.sidebar.markdown(f"**Quality:** {model_info['quality']}")
        st.sidebar.markdown(f"**Speed:** {model_info['speed']}")
        st.sidebar.markdown(f"**Dimensions:** {model_info['dimensions']}")

def main():
    st.title("HR Assistant Chatbot")
    
    stack_labels = {s["label"]: s["id"] for s in RAG_STACKS}
    default_label = next(
        (s["label"] for s in RAG_STACKS if s["id"] == DEFAULT_STACK_ID),
        RAG_STACKS[0]["label"],
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        selected_label = st.selectbox(
            "Select RAG Stack",
            options=list(stack_labels.keys()),
            index=list(stack_labels.keys()).index(default_label),
            help="Only tested LLM + embedding combinations are listed.",
        )
        stack = get_stack_by_id(stack_labels[selected_label])
        model_name = stack["llm"]
        embedding_type = stack["embedding_type"]
        embedding_model = stack["embedding_model"]

        st.caption(
            f"LLM: `{model_name}` · "
            f"Embeddings: `{embedding_type}` / `{embedding_model}`"
        )
        
        # Initialize button
        if st.button("Initialize Chatbot"):
            st.session_state.rag_service = initialize_rag_service(
                model_name,
                embedding_type,
                embedding_model
            )
            st.session_state.messages = []
            st.session_state.session_id = None
            st.success("Chatbot initialized successfully!")
            
            display_model_info(st.session_state.rag_service.embedding_service, model_name)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        if st.session_state.rag_service is None:
            st.error("Please initialize the chatbot first using the sidebar configuration.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, session_id = st.session_state.rag_service.process_message(
                    prompt,
                    st.session_state.session_id
                )
                st.session_state.session_id = session_id
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
