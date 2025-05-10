import docx2txt
import sys
from pathlib import Path

def read_docx(file_path):
    """Read and display the contents of a docx file"""
    try:
        # Extract text from the docx file
        text = docx2txt.process(file_path)
        print(f"\nContents of {Path(file_path).name}:")
        print("=" * 50)
        print(text)
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_docx.py <path_to_docx_file>")
        sys.exit(1)
        
    read_docx(sys.argv[1]) 