# Question Answering with LangChain

This directory demonstrates how to build different types of question-answering (QA) systems using LangChain. From basic QA to advanced conversational systems with source citations.

## Files Overview

### 1. qa_basic.py
Basic question-answering implementation with document retrieval.

Key Features:
- Document-based QA
- Vector store integration
- Knowledge base creation
- Response formatting

Example usage:
```python
# Initialize components
llm, embeddings = initialize_qa_components()

# Create knowledge base
vectorstore = create_knowledge_base(documents, embeddings)

# Set up QA chain
qa_chain = setup_qa_chain(llm, vectorstore)

# Ask questions
response = await ask_question(qa_chain, "What is LangChain?")
```

### 2. qa_with_sources.py
Advanced QA system that provides source citations and references.

Features:
- Source attribution
- Citation formatting
- Source quality evaluation
- Confidence scoring

Example usage:
```python
# Set up QA chain with sources
qa_chain = setup_qa_with_sources(
    llm,
    vectorstore,
    create_custom_prompt()
)

# Get answer with sources
response = await ask_with_sources(
    qa_chain,
    "What are the key features of Python?"
)

# Evaluate source quality
quality_metrics = evaluate_source_quality(response["sources"])
```

### 3. conversational_qa.py
Implements a conversational QA system with memory and context management.

Key Features:
- Conversation memory
- Context retention
- Follow-up questions
- Chat history tracking

Example usage:
```python
# Initialize conversational QA
chat_model, memory = initialize_models()
qa_chain = setup_conversational_qa(
    chat_model,
    vectorstore,
    memory
)

# Have a conversation
response = await ask_question(
    qa_chain,
    "Tell me about neural networks"
)
```

## Key Concepts Explained

### Question Answering Types

1. **Basic QA**
   - Simple question-answer pairs
   - Document retrieval
   - Answer generation

2. **QA with Sources**
   - Source attribution
   - Citation management
   - Confidence scoring
   - Source verification

3. **Conversational QA**
   - Context maintenance
   - Memory management
   - Follow-up handling
   - History tracking

## Getting Started

1. Basic setup:
```bash
pip install langchain langchain-openai faiss-cpu
export OPENAI_API_KEY=your_api_key_here
```

2. Create a knowledge base:
```python
# Prepare documents
documents = [
    Document(page_content="...", metadata={"source": "..."}),
    # More documents...
]

# Create vector store
vectorstore = create_knowledge_base(
    documents,
    embeddings,
    save_path="qa_knowledge_base"
)
```

## Best Practices

### Document Preparation
- Clean and preprocess documents
- Add meaningful metadata
- Organize by topics
- Update regularly

### Question Processing
- Validate input questions
- Handle edge cases
- Implement fallbacks
- Format responses consistently

### Source Management
- Verify source reliability
- Track source usage
- Update sources regularly
- Implement source ranking

## Common Use Cases

1. **Documentation Helper**
```python
# Load technical documentation
docs = load_documentation()
qa_chain = setup_qa_with_sources(llm, docs)

# Answer technical questions
response = await ask_with_sources(
    qa_chain,
    "How do I implement feature X?"
)
```

2. **Research Assistant**
```python
# Set up conversational research assistant
qa_chain = setup_conversational_qa(
    chat_model,
    research_docs,
    memory
)

# Research interaction
response = await ask_question(
    qa_chain,
    "What are the latest developments in AI?"
)
```

## Tips for Improvement

1. **Answer Quality**
   - Use appropriate temperature settings
   - Implement answer validation
   - Add confidence scores
   - Format responses clearly

2. **Source Handling**
   - Implement source ranking
   - Track source freshness
   - Validate citations
   - Update sources regularly

3. **Conversation Management**
   - Clear context when needed
   - Manage memory size
   - Handle topic switches
   - Implement fallback responses

## Advanced Features

1. **Multi-Source Integration**
   - Combine multiple knowledge bases
   - Weight sources by reliability
   - Cross-reference answers
   - Merge similar responses

2. **Answer Validation**
   - Check answer relevance
   - Verify source citations
   - Validate against known facts
   - Implement confidence thresholds 