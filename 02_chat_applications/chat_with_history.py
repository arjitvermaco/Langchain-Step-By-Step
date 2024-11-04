from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from dotenv import load_dotenv

class ChatMessage(BaseModel):
    """Model for chat messages with metadata."""
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class ChatSession(BaseModel):
    """Model for chat sessions."""
    session_id: str
    created_at: datetime
    messages: List[ChatMessage]
    system_message: str

def create_chat_session(
    session_id: str,
    system_message: str = "You are a helpful AI assistant."
) -> ChatSession:
    """
    Create a new chat session.
    
    Args:
        session_id (str): Unique identifier for the session
        system_message (str): System message for the chat
        
    Returns:
        ChatSession: New chat session instance
    """
    return ChatSession(
        session_id=session_id,
        created_at=datetime.now(),
        messages=[],
        system_message=system_message
    )

class ChatHistoryManager:
    """Manager for chat sessions with persistence."""
    
    def __init__(self, storage_dir: str = "chat_history"):
        load_dotenv()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.chat_model = ChatOpenAI(
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.active_sessions: Dict[str, ChatSession] = {}
        self.load_sessions()
    
    def load_sessions(self) -> None:
        """Load saved sessions from storage."""
        for file_path in self.storage_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                session = ChatSession(
                    session_id=data["session_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    messages=[ChatMessage(**msg) for msg in data["messages"]],
                    system_message=data["system_message"]
                )
                self.active_sessions[session.session_id] = session
    
    def save_session(self, session_id: str) -> None:
        """
        Save a session to storage.
        
        Args:
            session_id (str): ID of session to save
        """
        session = self.active_sessions.get(session_id)
        if session:
            file_path = self.storage_dir / f"{session_id}.json"
            with open(file_path, "w") as f:
                json.dump(
                    session.dict(),
                    f,
                    default=str,
                    indent=2
                )
    
    async def send_message(
        self,
        session_id: str,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Send a message in a specific chat session.
        
        Args:
            session_id (str): Session ID
            message (str): User message
            metadata (Dict): Optional metadata for the message
            
        Returns:
            Optional[str]: Bot response
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                session = create_chat_session(session_id)
                self.active_sessions[session_id] = session
            
            # Create message list for the model
            messages = [SystemMessage(content=session.system_message)]
            messages.extend([
                HumanMessage(content=msg.content) if msg.role == "user"
                else AIMessage(content=msg.content)
                for msg in session.messages
            ])
            messages.append(HumanMessage(content=message))
            
            # Get response
            response = await self.chat_model.ainvoke(messages)
            
            # Save messages
            session.messages.append(ChatMessage(
                role="user",
                content=message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            ))
            session.messages.append(ChatMessage(
                role="assistant",
                content=response.content,
                timestamp=datetime.now(),
                metadata={}
            ))
            
            # Save session
            self.save_session(session_id)
            
            return response.content
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return None
    
    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get message history for a session.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of messages or None if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return [msg.dict() for msg in session.messages]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if session was deleted, False otherwise
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            file_path = self.storage_dir / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

# Usage example
if __name__ == "__main__":
    async def main():
        manager = ChatHistoryManager()
        session_id = "test_session"
        
        # First message
        response = await manager.send_message(
            session_id,
            "Tell me about the solar system.",
            metadata={"topic": "astronomy"}
        )
        print("Bot:", response)
        
        # Second message
        response = await manager.send_message(
            session_id,
            "What's the largest planet?",
            metadata={"topic": "astronomy", "subtopic": "planets"}
        )
        print("Bot:", response)
        
        # Print session history
        print("\nChat History:")
        history = manager.get_session_history(session_id)
        for msg in history:
            print(f"{msg['role'].title()} ({msg['timestamp']}): {msg['content']}") 