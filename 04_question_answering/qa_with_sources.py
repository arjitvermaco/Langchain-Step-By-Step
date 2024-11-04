from typing import Optional, Dict, Any, List
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

def initialize_models(
    model_name: str = "gpt-3.5-turbo-instruct",
    temperature: float = 0.7
) -> tuple[OpenAI, OpenAIEmbeddings]:
    """
    Initialize the language and embedding models.
    
    Args:
        model_name (str): Name of the OpenAI model
        temperature (float): Temperature for response generation
        
    Returns:
        tuple[OpenAI, OpenAIEmbeddings]: Initialized models
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

def create_custom_prompt() -> PromptTemplate:
    """
    Create a custom prompt template for QA with sources.
    
    Returns:
        PromptTemplate: Customized prompt template
    """
    template = """Given the following extracted parts of documents and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}

EXTRACTED DOCUMENTS:
{summaries}

Please provide a detailed answer and cite the specific sources used:
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["summaries", "question"]
    )

def setup_qa_with_sources(
    llm: OpenAI,
    vectorstore: FAISS,
    prompt: Optional[PromptTemplate] = None
) -> RetrievalQAWithSourcesChain:
    """
    Set up a QA chain that includes sources in its responses.
    
    Args:
        llm (OpenAI): Language model
        vectorstore (FAISS): Vector store with documents
        prompt (Optional[PromptTemplate]): Custom prompt template
        
    Returns:
        RetrievalQAWithSourcesChain: QA chain with source attribution
    """
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Number of sources to retrieve
        ),
        chain_type_kwargs={
            "prompt": prompt or create_custom_prompt()
        }
    )

async def ask_with_sources(
    qa_chain: RetrievalQAWithSourcesChain,
    question: str
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with detailed source citations.
    
    Args:
        qa_chain (RetrievalQAWithSourcesChain): QA chain to use
        question (str): Question to ask
        
    Returns:
        Dict[str, Any]: Answer and source information
    """
    try:
        response = await qa_chain.ainvoke({"question": question})
        return {
            "answer": response["answer"],
            "sources": response["sources"].split("\n")
        }
    except Exception as e:
        print(f"Error getting answer: {str(e)}")
        return {"answer": None, "sources": []}

def format_sources_response(response: Dict[str, Any]) -> str:
    """
    Format the QA response with source citations.
    
    Args:
        response (Dict[str, Any]): QA response with answer and sources
        
    Returns:
        str: Formatted response with citations
    """
    if not response["answer"]:
        return "Sorry, I couldn't find an answer."
    
    formatted = f"Answer:\n{response['answer']}\n\nSources:"
    
    for i, source in enumerate(response["sources"], 1):
        if source.strip():  # Skip empty sources
            formatted += f"\n[{i}] {source.strip()}"
    
    return formatted

def evaluate_source_quality(sources: List[str]) -> Dict[str, Any]:
    """
    Evaluate the quality and relevance of sources.
    
    Args:
        sources (List[str]): List of source citations
        
    Returns:
        Dict[str, Any]: Quality metrics for the sources
    """
    return {
        "num_sources": len(sources),
        "avg_length": sum(len(s) for s in sources) / len(sources) if sources else 0,
        "unique_sources": len(set(sources))
    }

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize models
        llm, embeddings = initialize_models()
        
        # Create sample documents with detailed sources
        documents = [
            Document(
                page_content="Python was created by Guido van Rossum and was released in 1991.",
                metadata={
                    "source": "python_history.txt",
                    "page": 1,
                    "reliability": "high",
                    "date": "2023"
                }
            ),
            Document(
                page_content="Python's name comes from Monty Python, not the snake.",
                metadata={
                    "source": "python_facts.txt",
                    "page": 3,
                    "reliability": "medium",
                    "date": "2023"
                }
            ),
            Document(
                page_content="Python is known for its simple, readable syntax and extensive standard library.",
                metadata={
                    "source": "programming_languages.txt",
                    "page": 7,
                    "reliability": "high",
                    "date": "2023"
                }
            )
        ]
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Set up QA chain with sources
        qa_chain = setup_qa_with_sources(llm, vectorstore)
        
        # Ask questions
        questions = [
            "When was Python created and by whom?",
            "Why is Python named Python?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = await ask_with_sources(qa_chain, question)
            print(format_sources_response(response))
            
            # Print source quality metrics
            quality_metrics = evaluate_source_quality(response["sources"])
            print("\nSource Quality Metrics:")
            print(f"Number of sources: {quality_metrics['num_sources']}")
            print(f"Average source length: {quality_metrics['avg_length']:.2f}")
            print(f"Unique sources: {quality_metrics['unique_sources']}") 