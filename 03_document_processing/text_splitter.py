from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

def split_text_by_characters(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks using character count.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return splitter.split_text(text)

def split_text_recursively(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", " ", ""]
) -> List[str]:
    """
    Split text recursively using multiple separators.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (List[str]): List of separators to use for splitting
        
    Returns:
        List[str]: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    return splitter.split_text(text)

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (List[Document]): Documents to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return splitter.split_documents(documents)

def split_by_tokens(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into chunks based on token count.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum number of tokens per chunk
        chunk_overlap (int): Number of tokens to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return splitter.split_text(text)

def get_chunk_statistics(chunks: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics about the text chunks.
    
    Args:
        chunks (List[str]): List of text chunks
        
    Returns:
        Dict[str, Any]: Statistics about the chunks
    """
    lengths = [len(chunk) for chunk in chunks]
    
    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": sum(lengths) / len(chunks) if chunks else 0,
        "min_chunk_length": min(lengths) if chunks else 0,
        "max_chunk_length": max(lengths) if chunks else 0
    }

# Usage example
if __name__ == "__main__":
    # Sample text
    text = """
    This is a sample text that we'll use to demonstrate text splitting.
    It contains multiple paragraphs and sentences.
    
    This is the second paragraph with different content.
    We'll see how different splitters handle this text.
    
    Finally, this is the third paragraph.
    The text splitters should handle this appropriately.
    """
    
    # Try different splitting methods
    char_chunks = split_text_by_characters(text, chunk_size=100, chunk_overlap=20)
    recursive_chunks = split_text_recursively(text, chunk_size=100, chunk_overlap=20)
    token_chunks = split_by_tokens(text, chunk_size=50, chunk_overlap=10)
    
    # Print results
    print("Character Splitting:")
    for i, chunk in enumerate(char_chunks):
        print(f"Chunk {i + 1}: {chunk}\n")
    
    print("\nRecursive Splitting:")
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i + 1}: {chunk}\n")
    
    print("\nToken Splitting:")
    for i, chunk in enumerate(token_chunks):
        print(f"Chunk {i + 1}: {chunk}\n")
    
    # Print statistics
    print("\nChunk Statistics:")
    print("Character Splitting:", get_chunk_statistics(char_chunks))
    print("Recursive Splitting:", get_chunk_statistics(recursive_chunks))
    print("Token Splitting:", get_chunk_statistics(token_chunks)) 