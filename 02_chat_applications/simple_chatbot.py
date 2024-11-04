from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def create_chat_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """
    Create and initialize a ChatOpenAI model.
    
    Args:
        model_name (str): Name of the model to use
        temperature (float): Temperature setting for response generation
        
    Returns:
        ChatOpenAI: Initialized chat model
    """
    load_dotenv()
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def init_chat(system_message: str = "You are a helpful AI assistant.") -> List[SystemMessage]:
    """
    Initialize a new chat with a system message.
    
    Args:
        system_message (str): The system message defining chatbot behavior
        
    Returns:
        List[SystemMessage]: Initial message list with system message
    """
    return [SystemMessage(content=system_message)]

async def send_message(
    message: str, 
    messages: List[HumanMessage | AIMessage | SystemMessage],
    chat_model: ChatOpenAI
) -> Optional[str]:
    """
    Send a message and get response from the chatbot.
    
    Args:
        message (str): The user's message
        messages (List): Current message history
        chat_model (ChatOpenAI): The chat model to use
        
    Returns:
        Optional[str]: The chatbot's response or None if an error occurs
    """
    try:
        # Add user message to history
        messages.append(HumanMessage(content=message))
        
        # Get response from chat model
        response = await chat_model.ainvoke(messages)
        
        # Add AI response to history
        messages.append(AIMessage(content=response.content))
        
        return response.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

def clear_history(messages: List[HumanMessage | AIMessage | SystemMessage]) -> List[SystemMessage]:
    """
    Reset the conversation history to initial state.
    
    Args:
        messages (List): Current message history
        
    Returns:
        List[SystemMessage]: Reset message list with only system message
    """
    return [messages[0]]  # Keep only system message

def get_chat_history(messages: List[HumanMessage | AIMessage | SystemMessage]) -> List[Dict[str, str]]:
    """
    Get the conversation history in a structured format.
    
    Args:
        messages (List): Current message history
        
    Returns:
        List[Dict[str, str]]: List of messages with role and content
    """
    history = []
    for message in messages[1:]:  # Skip system message
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            history.append({"role": "assistant", "content": message.content})
    return history

# Usage example
if __name__ == "__main__":
    chat_model = create_chat_model()
    messages = init_chat("You are a helpful AI assistant specialized in Python programming.")
    
    async def main():
        response = await send_message(
            "What's the difference between a list and a tuple in Python?",
            messages,
            chat_model
        )
        print("Bot:", response)
        
        # Get another response
        response = await send_message(
            "Can you show me an example of when to use each?",
            messages,
            chat_model
        )
        print("Bot:", response)
        
        # Print chat history
        print("\nChat History:")
        for message in get_chat_history(messages):
            print(f"{message['role'].title()}: {message['content']}")