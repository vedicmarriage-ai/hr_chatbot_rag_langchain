# HR Chatbot

An intelligent HR chatbot built using RAG (Retrieval-Augmented Generation) and LangChain, powered by Ollama LLM.

## Features

- Conversational HR assistant
- Document-based knowledge retrieval
- Session management
- FastAPI backend
- Vector store for efficient document retrieval

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hr-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.server:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /chat`: Send messages to the chatbot
- `GET /health`: Health check endpoint

## Development

- Run tests: `pytest`
- Format code: `black .`
- Sort imports: `isort .`
- Lint code: `flake8`

## Project Structure

```
HR_chat_bot/
├── app/                    # Application code
│   ├── packages/          # Core chatbot implementation
│   ├── lib/              # Shared libraries
│   └── utils/            # Utility functions
├── data/                  # Data storage
│   └── docs/             # HR documents
├── tests/                # Test files
├── configs/              # Configuration files
└── scripts/              # Utility scripts
```

## License

[Your License] 