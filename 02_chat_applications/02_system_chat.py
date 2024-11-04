"""
System-Enabled Chat Implementation

This file builds upon basic_chat by adding system message support and basic
message history. It demonstrates:
1. System message initialization
2. Message history tracking
3. Contextual responses

Step by Step Process When Run:
1. Main function:
   - Initializes chat model
   - Creates system messages list with:
     * System behavior definition
     * Initial context setting
   
2. create_system_messages function:
   - Takes system prompt
   - Creates initial message list
   - Sets up chat behavior

3. send_message_with_system function:
   - Takes user message, message history, and chat model
   - Appends user message to history
   - Gets response from API
   - Appends AI response to history
   - Returns response content

4. Message Flow:
   a. System message defines behavior
   b. User message added to context
   c. AI response generated with full context
   d. Both messages saved in history

Key Concepts:
- Maintains message history
- System message provides context
- Messages build upon each other
- Stateful conversation

Use Case:
Suitable for conversations where consistent AI behavior and basic
context maintenance are needed.
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import asyncio

def initialize_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    """Initialize chat model with temperature setting."""
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_system_messages(system_prompt: str) -> List[SystemMessage]:
    """
    Create initial message list with system prompt.
    
    Args:
        system_prompt (str): System behavior definition
        
    Returns:
        List[SystemMessage]: Initial messages list
    """
    return [SystemMessage(content=system_prompt)]

async def send_message_with_system(
    message: str,
    messages: List[SystemMessage | HumanMessage | AIMessage],
    chat_model: ChatOpenAI
) -> Optional[str]:
    """
    Send message with system context.
    
    Args:
        message (str): User's message
        messages (List): Current message history
        chat_model (ChatOpenAI): The chat model
        
    Returns:
        Optional[str]: Model's response or None if error
    """
    try:
        # Add user message
        messages.append(HumanMessage(content=message))
        
        # Get response
        response = await chat_model.ainvoke(messages)
        
        # Add AI response to history
        messages.append(AIMessage(content=response.content))
        
        return response.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    async def main():
        chat_model = initialize_chat_model()
        messages = create_system_messages(
            "You are a helpful Python programming assistant."
        )
        
        print("Chat initialized. Type 'exit' to end the conversation.")
        
        while True:
            # Get user input
            user_message = input("\nYou: ").strip()
            
            # Check for exit command
            if user_message.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Get and print response
            response = await send_message_with_system(
                user_message,
                messages,
                chat_model
            )
            print(f"\nBot: {response}")
        
    asyncio.run(main())