import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

if __name__ == "__main__":
    # Get the path to the Streamlit app
    streamlit_app_path = os.path.join(project_root, "app", "streamlit_app.py")
    
    # Run the Streamlit app
    os.system(f"streamlit run {streamlit_app_path}") 