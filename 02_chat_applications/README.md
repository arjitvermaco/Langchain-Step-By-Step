# Chat Applications with LangChain

This directory demonstrates how to build different types of chat applications using LangChain, from simple chatbots to complex conversational systems with memory and history management.

## Files Overview

### 1. simple_chatbot.py
A basic chatbot implementation to understand fundamental concepts.

Key Features:
- Basic message handling
- System message configuration
- Simple response generation
- Chat history tracking

Example usage:
```python
# Create a basic chatbot
chat_model = create_chat_model(temperature=0.7)
messages = init_chat("You are a helpful assistant")

# Send a message
response = await send_message(
    "What is Python?",
    messages,
    chat_model
)
```

### 2. memory_chat.py
Implements chat applications with different types of memory systems.

Features:
- Multiple memory types (Buffer, Summary)
- Context awareness
- Memory management
- Conversation state tracking

Example usage:
```python
# Initialize chat with memory
chat_model, memory, messages = init_chat_with_memory(
    system_message="You are a coding expert",
    memory_type="buffer"
)

# Chat with context
response = await send_message_with_memory(
    "How do I use async in Python?",
    chat_model,
    memory,
    messages
)
```

### 3. chat_with_history.py
Advanced chat implementation with persistent history and session management.

Key Features:
- Session management
- Persistent storage
- Message metadata
- Conversation analytics

Example usage:
```python
# Create a chat session
session = create_chat_session(
    session_id="user_123",
    system_message="You are an AI tutor"
)

# Send messages with metadata
response = await send_message(
    session_id="user_123",
    message="Explain machine learning",
    metadata={"topic": "AI", "difficulty": "beginner"}
)
```

## Key Concepts Explained

### Chat Memory Types

1. **Buffer Memory**
   - Stores complete conversation history
   - Good for short conversations
   - Maintains full context
   ```python
   memory = create_memory(memory_type="buffer")
   ```

2. **Summary Memory**
   - Keeps condensed conversation history
   - Better for long conversations
   - Saves token usage
   ```python
   memory = create_memory(memory_type="summary")
   ```

### Session Management
- Unique session IDs
- Persistent storage
- Metadata tracking
- History management

## Getting Started

1. Basic Setup:
```bash
pip install langchain langchain-openai
export OPENAI_API_KEY=your_api_key_here
```

2. Create a Simple Chat:
```python
from simple_chatbot import create_chat_model, init_chat, send_message

chat_model = create_chat_model()
messages = init_chat()
response = await send_message("Hello!", messages, chat_model)
```

## Best Practices

### Memory Management
- Clear memory when appropriate
- Use summary memory for long conversations
- Implement memory size limits
- Regular cleanup of old sessions

### Response Handling
- Implement error handling
- Validate responses
- Format output consistently
- Handle rate limits

### Session Management
- Implement session timeouts
- Clean up old sessions
- Secure storage of sensitive data
- Regular backups

## Common Use Cases

1. **Customer Support Bot**
```python
# Initialize support bot
chat_model, memory, messages = init_chat_with_memory(
    system_message="You are a helpful customer support agent"
)

# Handle customer query
response = await send_message_with_memory(
    "I need help with my order",
    chat_model,
    memory,
    messages
)
```

2. **Educational Assistant**
```python
# Create educational session
session = create_chat_session(
    session_id="student_123",
    system_message="You are a patient math tutor"
)

# Tutorial interaction
response = await send_message(
    session_id="student_123",
    message="Help me understand calculus",
    metadata={"subject": "math", "level": "advanced"}
)
```

## Tips for Development

1. **Response Quality**
   - Adjust temperature based on use case
   - Implement response validation
   - Format responses consistently
   - Handle edge cases

2. **Performance**
   - Implement caching
   - Use async operations
   - Batch process when possible
   - Monitor memory usage

3. **User Experience**
   - Clear error messages
   - Appropriate response times
   - Context maintenance
   - Natural conversation flow

## Advanced Features

1. **Analytics Integration**
   - Track conversation metrics
   - Analyze user patterns
   - Monitor performance
   - Generate insights

2. **Multi-Modal Support**
   - Handle different input types
   - Process attachments
   - Support rich responses
   - Integrate media handling 