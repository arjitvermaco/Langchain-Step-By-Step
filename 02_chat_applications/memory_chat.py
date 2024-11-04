from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv
import os
import asyncio

def create_memory(memory_type: str = "buffer") -> Any:
    """
    Create a conversation memory instance.
    
    Args:
        memory_type (str): Type of memory to create ("buffer" or "summary")
        
    Returns:
        ConversationBufferMemory or ConversationSummaryMemory
    """
    if memory_type == "buffer":
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    elif memory_type == "summary":
        return ConversationSummaryMemory(
            llm=ChatOpenAI(temperature=0),
            memory_key="chat_history",
            return_messages=True
        )
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

def init_chat_with_memory(
    system_message: str = "You are a helpful AI assistant.",
    memory_type: str = "buffer"
) -> tuple[ChatOpenAI, Any, List[SystemMessage]]:
    """
    Initialize chat components with memory.
    
    Args:
        system_message (str): System message for the chat
        memory_type (str): Type of memory to use
        
    Returns:
        tuple: (chat_model, memory, messages)
    """
    load_dotenv()
    
    chat_model = ChatOpenAI(
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    memory = create_memory(memory_type)
    messages = [
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="chat_history")
    ]
    
    return chat_model, memory, messages

async def send_message_with_memory(
    message: str,
    chat_model: ChatOpenAI,
    memory: Any,
    messages: List[Any]
) -> Optional[str]:
    """
    Send a message and get response using memory.
    
    Args:
        message (str): User message
        chat_model (ChatOpenAI): Chat model instance
        memory (Any): Memory instance
        messages (List): Message history
        
    Returns:
        Optional[str]: Bot response
    """
    try:
        # Add user message to memory
        memory.chat_memory.add_user_message(message)
        
        # Get chat history from memory
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        # Prepare messages with history
        current_messages = messages.copy()
        current_messages.extend(chat_history)
        current_messages.append(HumanMessage(content=message))
        
        # Get response
        response = await chat_model.ainvoke(current_messages)
        
        # Add response to memory
        memory.chat_memory.add_ai_message(response.content)
        
        return response.content
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return None

def get_memory_contents(memory: Any) -> List[Dict[str, str]]:
    """
    Get the contents of memory in a structured format.
    
    Args:
        memory (Any): Memory instance
        
    Returns:
        List[Dict[str, str]]: List of messages from memory
    """
    messages = memory.chat_memory.messages
    return [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        }
        for msg in messages
    ]

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize chat with buffer memory
        chat_model, memory, messages = init_chat_with_memory(
            system_message="You are a knowledgeable AI assistant specialized in science.",
            memory_type="buffer"
        )
        
        print("Chat initialized. Type 'exit' to end the conversation.")
        print("Type 'history' to see chat history.")
        
        # Main chat loop
        while True:
            # Get user input
            user_message = input("\nYou: ").strip()
            
            # Check for exit command
            if user_message.lower() == 'exit':
                print("Goodbye!")
                break
                
            # Check for history command    
            if user_message.lower() == 'history':
                print("\nChat History:")
                for msg in get_memory_contents(memory):
                    print(f"{msg['role'].title()}: {msg['content']}")
                continue
            
            # Get response from AI
            response = await send_message_with_memory(
                user_message,
                chat_model,
                memory,
                messages
            )
            
            # Print AI response
            print(f"\nBot: {response}")

    # Run the async main function
    asyncio.run(main())