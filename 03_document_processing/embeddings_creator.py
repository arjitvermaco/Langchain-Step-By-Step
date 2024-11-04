from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
import numpy as np
from dotenv import load_dotenv
import os

def initialize_embeddings(
    provider: str = "openai",
    model_name: str = "text-embedding-3-small"
) -> Any:
    """
    Initialize embeddings model.
    
    Args:
        provider (str): Provider to use ('openai' or 'huggingface')
        model_name (str): Name of the model to use
        
    Returns:
        Any: Initialized embeddings model
    """
    load_dotenv()
    
    if provider == "openai":
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

async def create_embeddings(
    texts: List[str],
    embeddings_model: Any
) -> List[List[float]]:
    """
    Create embeddings for a list of texts.
    
    Args:
        texts (List[str]): List of texts to embed
        embeddings_model (Any): Initialized embeddings model
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    try:
        embeddings = await embeddings_model.aembed_documents(texts)
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return []

def create_vector_store(
    documents: List[Document],
    embeddings_model: Any,
    store_type: str = "faiss",
    persist_directory: Optional[str] = None
) -> Any:
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): Documents to store
        embeddings_model (Any): Embeddings model to use
        store_type (str): Type of vector store ('faiss' or 'chroma')
        persist_directory (Optional[str]): Directory to persist the store
        
    Returns:
        Any: Initialized vector store
    """
    try:
        if store_type == "faiss":
            store = FAISS.from_documents(documents, embeddings_model)
            if persist_directory:
                store.save_local(persist_directory)
            return store
        elif store_type == "chroma":
            return Chroma.from_documents(
                documents,
                embeddings_model,
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

async def find_similar_documents(
    query: str,
    vector_store: Any,
    num_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Find similar documents in the vector store.
    
    Args:
        query (str): Query text
        vector_store (Any): Vector store to search
        num_results (int): Number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of similar documents with scores
    """
    try:
        results = await vector_store.asimilarity_search_with_score(
            query,
            k=num_results
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    except Exception as e:
        print(f"Error searching vector store: {str(e)}")
        return []

def calculate_embedding_statistics(embeddings: List[List[float]]) -> Dict[str, float]:
    """
    Calculate statistics about embeddings.
    
    Args:
        embeddings (List[List[float]]): List of embedding vectors
        
    Returns:
        Dict[str, float]: Statistics about the embeddings
    """
    if not embeddings:
        return {}
    
    embeddings_array = np.array(embeddings)
    return {
        "mean": float(np.mean(embeddings_array)),
        "std": float(np.std(embeddings_array)),
        "min": float(np.min(embeddings_array)),
        "max": float(np.max(embeddings_array)),
        "dimension": embeddings_array.shape[1]
    }

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize embeddings model
        embeddings_model = initialize_embeddings()
        
        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!"
        ]
        
        # Create embeddings
        embeddings = await create_embeddings(texts, embeddings_model)
        
        # Create documents
        documents = [Document(page_content=text) for text in texts]
        
        # Create vector store
        vector_store = create_vector_store(
            documents,
            embeddings_model,
            store_type="faiss",
            persist_directory="vector_store"
        )
        
        # Search for similar documents
        query = "quick animals jumping"
        similar_docs = await find_similar_documents(query, vector_store)
        
        # Print results
        print("\nSimilar Documents:")
        for doc in similar_docs:
            print(f"\nContent: {doc['content']}")
            print(f"Score: {doc['score']}")
        
        # Print embedding statistics
        print("\nEmbedding Statistics:")
        print(calculate_embedding_statistics(embeddings)) 