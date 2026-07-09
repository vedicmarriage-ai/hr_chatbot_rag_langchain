#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to print usage
print_usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  create-vector-store    Create vector store from documents"
    echo "  run-streamlit         Run the Streamlit app"
    echo "  run-api               Run the FastAPI server (uvicorn)"
    echo "  benchmark             Run vector store benchmarking"
    echo "  help                  Show this help message"
}

# Function to run command in conda environment
run_in_conda() {
    "$PROJECT_ROOT/run_in_conda.sh" "$@"
}

# Main script logic
case "$1" in
    "create-vector-store")
        run_in_conda python "$SCRIPT_DIR/create_vector_store.py" --embedding huggingface
        ;;
    "run-streamlit")
        run_in_conda streamlit run "$PROJECT_ROOT/app/streamlit_app.py"
        ;;
    "run-api")
        run_in_conda python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    "benchmark")
        run_in_conda python "$SCRIPT_DIR/benchmark_vector_stores.py"
        ;;
    "help"|"")
        print_usage
        ;;
    *)
        echo "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac 