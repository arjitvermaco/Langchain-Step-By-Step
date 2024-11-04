from typing import Optional, Dict, Any, List
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

def initialize_qa_components(
    model_name: str = "gpt-3.5-turbo-instruct",
    temperature: float = 0.7
) -> tuple[OpenAI, OpenAIEmbeddings]:
    """
    Initialize the LLM and embeddings model.
    
    Args:
        model_name (str): Name of the OpenAI model
        temperature (float): Temperature for response generation
        
    Returns:
        tuple[OpenAI, OpenAIEmbeddings]: Initialized LLM and embeddings model
    """
    load_dotenv()
    
    llm = OpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return llm, embeddings

def create_knowledge_base(
    documents: List[Document],
    embeddings: OpenAIEmbeddings,
    save_path: Optional[str] = None
) -> FAISS:
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): Documents for the knowledge base
        embeddings (OpenAIEmbeddings): Embeddings model
        save_path (Optional[str]): Path to save the vector store
        
    Returns:
        FAISS: Vector store containing document embeddings
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    if save_path:
        vectorstore.save_local(save_path)
    
    return vectorstore

def setup_qa_chain(
    llm: OpenAI,
    vectorstore: FAISS,
    chain_type: str = "stuff"
) -> RetrievalQA:
    """
    Set up a question-answering chain.
    
    Args:
        llm (OpenAI): Language model
        vectorstore (FAISS): Vector store with documents
        chain_type (str): Type of QA chain to use
        
    Returns:
        RetrievalQA: Initialized QA chain
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

async def ask_question(
    qa_chain: RetrievalQA,
    question: str
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with sources.
    
    Args:
        qa_chain (RetrievalQA): QA chain to use
        question (str): Question to ask
        
    Returns:
        Dict[str, Any]: Answer and source documents
    """
    try:
        response = await qa_chain.ainvoke({"query": question})
        
        return {
            "answer": response["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["source_documents"]
            ]
        }
    except Exception as e:
        print(f"Error getting answer: {str(e)}")
        return {"answer": None, "sources": []}

def format_qa_response(response: Dict[str, Any]) -> str:
    """
    Format QA response for display.
    
    Args:
        response (Dict[str, Any]): QA response with answer and sources
        
    Returns:
        str: Formatted response string
    """
    if not response["answer"]:
        return "Sorry, I couldn't find an answer."
    
    formatted = f"Answer: {response['answer']}\n\nSources:\n"
    
    for i, source in enumerate(response["sources"], 1):
        formatted += f"\n{i}. Content: {source['content'][:200]}..."
        if source["metadata"]:
            formatted += f"\n   Metadata: {source['metadata']}"
    
    return formatted

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize components
        llm, embeddings = initialize_qa_components()
        
        # Create sample documents
        documents = [
            Document(
                page_content="The capital of France is Paris. It is known for the Eiffel Tower.",
                metadata={"source": "geography.txt", "page": 1}
            ),
            Document(
                page_content="Paris is the largest city in France and a global center for art and culture.",
                metadata={"source": "geography.txt", "page": 2}
            )
        ]
        
        # Create knowledge base
        vectorstore = create_knowledge_base(
            documents,
            embeddings,
            save_path="qa_knowledge_base"
        )
        
        # Set up QA chain
        qa_chain = setup_qa_chain(llm, vectorstore)
        
        # Ask questions
        questions = [
            "What is the capital of France?",
            "What is Paris known for?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = await ask_question(qa_chain, question)
            print(format_qa_response(response)) 