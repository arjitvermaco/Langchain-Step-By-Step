from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document
import os

def load_text_file(file_path: str) -> Optional[List[Document]]:
    """
    Load content from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        Optional[List[Document]]: List of documents or None if loading fails
    """
    try:
        loader = TextLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading text file: {str(e)}")
        return None

def load_pdf_file(file_path: str) -> Optional[List[Document]]:
    """
    Load content from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        Optional[List[Document]]: List of documents or None if loading fails
    """
    try:
        loader = PDFMinerLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading PDF file: {str(e)}")
        return None

def load_markdown_file(file_path: str) -> Optional[List[Document]]:
    """
    Load content from a Markdown file.
    
    Args:
        file_path (str): Path to the Markdown file
        
    Returns:
        Optional[List[Document]]: List of documents or None if loading fails
    """
    try:
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading Markdown file: {str(e)}")
        return None

def load_directory(
    dir_path: str,
    extensions: List[str] = [".txt", ".pdf", ".md"]
) -> Dict[str, List[Document]]:
    """
    Load all supported documents from a directory.
    
    Args:
        dir_path (str): Path to the directory
        extensions (List[str]): List of file extensions to process
        
    Returns:
        Dict[str, List[Document]]: Dictionary of file paths and their documents
    """
    documents = {}
    
    for file_path in Path(dir_path).rglob("*"):
        if file_path.suffix not in extensions:
            continue
            
        try:
            if file_path.suffix == ".txt":
                docs = load_text_file(str(file_path))
            elif file_path.suffix == ".pdf":
                docs = load_pdf_file(str(file_path))
            elif file_path.suffix == ".md":
                docs = load_markdown_file(str(file_path))
                
            if docs:
                documents[str(file_path)] = docs
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return documents

def get_document_metadata(documents: Dict[str, List[Document]]) -> List[Dict[str, Any]]:
    """
    Extract metadata from loaded documents.
    
    Args:
        documents (Dict[str, List[Document]]): Dictionary of loaded documents
        
    Returns:
        List[Dict[str, Any]]: List of document metadata
    """
    metadata = []
    
    for file_path, docs in documents.items():
        for doc in docs:
            metadata.append({
                "file_path": file_path,
                "page_content_length": len(doc.page_content),
                "metadata": doc.metadata
            })
    
    return metadata

# Usage example
if __name__ == "__main__":
    # Create sample files
    os.makedirs("sample_docs", exist_ok=True)
    
    with open("sample_docs/sample.txt", "w") as f:
        f.write("This is a sample text file.")
    
    with open("sample_docs/sample.md", "w") as f:
        f.write("# Sample Markdown\nThis is a sample markdown file.")
    
    # Load documents
    documents = load_directory("sample_docs", extensions=[".txt", ".md"])
    
    # Print document contents
    for file_path, docs in documents.items():
        print(f"\nFile: {file_path}")
        for doc in docs:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    
    # Print metadata summary
    print("\nDocument Metadata:")
    for meta in get_document_metadata(documents):
        print(f"File: {meta['file_path']}")
        print(f"Content Length: {meta['page_content_length']}")
        print(f"Metadata: {meta['metadata']}\n") 