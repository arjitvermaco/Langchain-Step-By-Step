"""
Basic Chat Implementation with LangChain

This file implements the most basic form of chat interaction with an LLM.
It demonstrates the fundamental concepts of:
1. Initializing a chat model
2. Sending a single message
3. Handling the response

Step by Step Process When Run:
1. Main function is called, which:
   - Initializes the chat model by:
     * Loading environment variables
     * Creating ChatOpenAI instance with specified temperature
   
2. send_basic_message function:
   - Takes a user message and chat model
   - Sends message to OpenAI API
   - Waits for response
   - Returns response content or None if error

3. Error Handling:
   - Catches any API or processing errors
   - Returns None and prints error message

Key Concepts:
- No message history is maintained
- Each message is independent
- No system context is provided
- Simple async/await pattern

Use Case:
Best for simple, single-turn interactions where context isn't needed.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

def initialize_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize a basic chat model.
    
    Args:
        temperature (float): Response randomness (0.0 to 1.0)
        
    Returns:
        ChatOpenAI: Initialized chat model
    """
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

async def send_basic_message(
    message: str,
    chat_model: ChatOpenAI
) -> Optional[str]:
    """
    Send a basic message without context or history.
    
    Args:
        message (str): User's message
        chat_model (ChatOpenAI): The chat model
        
    Returns:
        Optional[str]: Model's response or None if error
    """
    try:
        response = await chat_model.ainvoke(message)
        return response.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    async def main():
        chat_model = initialize_chat_model()
        response = await send_basic_message(
            "What is Python?",
            chat_model
        )
        print(f"Bot: {response}")
    
    asyncio.run(main())