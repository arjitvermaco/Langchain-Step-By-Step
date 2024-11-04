from typing import Optional, Dict, Any, List
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_models() -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    Initialize the chat and embedding models.
    
    Returns:
        tuple[ChatOpenAI, OpenAIEmbeddings]: Initialized models
    """
    load_dotenv()
    
    chat_model = ChatOpenAI(
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return chat_model, embeddings

def create_conversation_memory() -> ConversationBufferMemory:
    """
    Create a conversation memory instance.
    
    Returns:
        ConversationBufferMemory: Memory instance for tracking conversation
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

def create_qa_prompt() -> PromptTemplate:
    """
    Create a prompt template for conversational QA.
    
    Returns:
        PromptTemplate: Customized prompt template
    """
    template = """Use the following conversation and context to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context and previous conversation:
"""
    
    return PromptTemplate.from_template(template)

def setup_conversational_qa(
    llm: ChatOpenAI,
    vectorstore: FAISS,
    memory: ConversationBufferMemory,
    prompt: Optional[PromptTemplate] = None
) -> ConversationalRetrievalChain:
    """
    Set up a conversational QA chain.
    
    Args:
        llm (ChatOpenAI): Language model
        vectorstore (FAISS): Vector store with documents
        memory (ConversationBufferMemory): Conversation memory
        prompt (Optional[PromptTemplate]): Custom prompt template
        
    Returns:
        ConversationalRetrievalChain: Conversational QA chain
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt} if prompt else None,
        return_source_documents=True
    )

async def ask_question(
    qa_chain: ConversationalRetrievalChain,
    question: str
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with context from conversation history.
    
    Args:
        qa_chain (ConversationalRetrievalChain): QA chain to use
        question (str): Question to ask
        
    Returns:
        Dict[str, Any]: Answer and related information
    """
    try:
        response = await qa_chain.ainvoke({"question": question})
        logger.info(f"Question asked: {question}")
        
        return {
            "answer": response["answer"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["source_documents"]
            ],
            "chat_history": response["chat_history"]
        }
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        return {"answer": None, "sources": [], "chat_history": []}

def format_conversation_response(response: Dict[str, Any]) -> str:
    """
    Format the conversational QA response.
    
    Args:
        response (Dict[str, Any]): QA response with answer and context
        
    Returns:
        str: Formatted response
    """
    if not response["answer"]:
        return "Sorry, I couldn't find an answer."
    
    formatted = f"Answer:\n{response['answer']}\n\nSources:"
    
    for i, source in enumerate(response["sources"], 1):
        formatted += f"\n[{i}] Content: {source['content'][:200]}..."
        if source["metadata"]:
            formatted += f"\n    Metadata: {source['metadata']}"
    
    return formatted

def get_chat_history(response: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract formatted chat history from response.
    
    Args:
        response (Dict[str, Any]): QA response containing chat history
        
    Returns:
        List[Dict[str, str]]: Formatted chat history
    """
    history = []
    for message in response["chat_history"]:
        if hasattr(message, "content"):
            role = "user" if "Human" in str(type(message)) else "assistant"
            history.append({
                "role": role,
                "content": message.content
            })
    return history

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize components
        chat_model, embeddings = initialize_models()
        memory = create_conversation_memory()
        
        # Create sample documents
        documents = [
            Document(
                page_content="Neural networks are a type of machine learning model inspired by the human brain.",
                metadata={"source": "ml_basics.txt", "topic": "neural_networks"}
            ),
            Document(
                page_content="Deep learning is a subset of machine learning that uses multiple layers of neural networks.",
                metadata={"source": "ml_basics.txt", "topic": "deep_learning"}
            )
        ]
        
        # Create vector store and QA chain
        vectorstore = FAISS.from_documents(documents, embeddings)
        qa_chain = setup_conversational_qa(
            chat_model,
            vectorstore,
            memory,
            create_qa_prompt()
        )
        
        # Simulate a conversation
        questions = [
            "What are neural networks?",
            "How does this relate to deep learning?",
            "Can you summarize what we discussed about neural networks and deep learning?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = await ask_question(qa_chain, question)
            print(format_conversation_response(response))
            
            # Print chat history
            print("\nChat History:")
            for msg in get_chat_history(response):
                print(f"{msg['role'].title()}: {msg['content']}") 