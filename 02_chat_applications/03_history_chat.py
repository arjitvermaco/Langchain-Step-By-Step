"""
Advanced Chat with History Management

This file implements a complete chat system with:
1. Session management
2. History tracking
3. Message counting
4. Timestamp tracking

Step by Step Process When Run:
1. Main function:
   - Initializes chat model
   - Creates new chat session with:
     * System prompt
     * Message counter
     * Creation timestamp
   
2. create_chat_session function:
   - Sets up new session dictionary
   - Initializes message list with system prompt
   - Sets up tracking metrics

3. send_message_with_history function:
   a. Receives message and session
   b. Processes message:
      - Adds to message history
      - Gets AI response
      - Updates message count
   c. Returns comprehensive response:
      - AI response
      - Full chat history
      - Message count

4. get_chat_history function:
   - Formats message history
   - Excludes system message
   - Creates role-based format

5. Session Tracking:
   - Counts total messages
   - Tracks creation time
   - Maintains full history

Key Concepts:
- Complete session management
- Detailed message tracking
- History formatting
- Metadata maintenance

Use Case:
Ideal for applications requiring:
- Long-term conversation tracking
- Analytics
- Session management
- History review capabilities
"""

from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def initialize_chat_model(temperature: float = 0.7) -> ChatOpenAI:
    """Initialize chat model with temperature setting."""
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_chat_session(system_prompt: str) -> Dict[str, Any]:
    """
    Create a new chat session with system prompt.
    
    Args:
        system_prompt (str): System behavior definition
        
    Returns:
        Dict[str, Any]: Chat session information
    """
    return {
        "messages": [SystemMessage(content=system_prompt)],
        "message_count": 0,
        "created_at": datetime.now()
    }

def get_chat_history(
    messages: List[SystemMessage | HumanMessage | AIMessage]
) -> List[Dict[str, str]]:
    """
    Get formatted chat history.
    
    Args:
        messages (List): Message history
        
    Returns:
        List[Dict[str, str]]: Formatted history
    """
    history = []
    for message in messages[1:]:  # Skip system message
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        history.append({
            "role": role,
            "content": message.content
        })
    return history

async def send_message_with_history(
    message: str,
    session: Dict[str, Any],
    chat_model: ChatOpenAI
) -> Optional[Dict[str, Any]]:
    """
    Send message with history tracking.
    
    Args:
        message (str): User's message
        session (Dict[str, Any]): Chat session
        chat_model (ChatOpenAI): The chat model
        
    Returns:
        Optional[Dict[str, Any]]: Response information or None if error
    """
    try:
        messages = session["messages"]
        
        # Add user message
        messages.append(HumanMessage(content=message))
        
        # Get response
        response = await chat_model.ainvoke(messages)
        
        # Add AI response
        messages.append(AIMessage(content=response.content))
        
        # Update session
        session["message_count"] += 2  # User message + AI response
        
        return {
            "response": response.content,
            "history": get_chat_history(messages),
            "message_count": session["message_count"]
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize chat model and session
        chat_model = initialize_chat_model()
        session = create_chat_session("You are a helpful Python programming assistant.")
        
        print("Chat initialized. Type 'quit' to exit.")
        
        while True:
            # Get user input
            user_message = input("\nYou: ").strip()
            
            # Check for quit command
            if user_message.lower() in ['quit', 'exit']:
                print("\nEnding chat session...")
                break
            
            # Send message and get response
            response = await send_message_with_history(
                user_message,
                session,
                chat_model
            )
            
            if response:
                print(f"\nBot: {response['response']}")
            else:
                print("\nError: Failed to get response")
                
    import asyncio
    asyncio.run(main())