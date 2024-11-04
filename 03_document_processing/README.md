# Document Processing with LangChain

This directory focuses on document handling, text processing, and embedding creation using LangChain. Learn how to load, split, and create embeddings for various document types.

## Files Overview

### 1. document_loader.py
Handles loading documents from different file formats.

Key Features:
- Multiple format support (PDF, TXT, Markdown)
- Batch document loading
- Metadata extraction
- Error handling

Example usage:
```python
# Load a single text file
documents = load_text_file("sample.txt")

# Load multiple documents from a directory
documents = load_directory(
    "docs/",
    extensions=[".txt", ".pdf", ".md"]
)
```

### 2. text_splitter.py
Splits documents into manageable chunks for processing.

Splitting Methods:
- Character-based splitting
- Recursive splitting
- Token-based splitting
- Overlap handling

Example usage:
```python
# Split by characters
chunks = split_text_by_characters(
    text,
    chunk_size=1000,
    chunk_overlap=200
)

# Split recursively with custom separators
chunks = split_text_recursively(
    text,
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)
```

### 3. embeddings_creator.py
Creates and manages document embeddings for semantic search and analysis.

Features:
- Multiple embedding providers (OpenAI, HuggingFace)
- Vector store creation
- Similarity search
- Embedding statistics

Example usage:
```python
# Initialize embeddings
embeddings_model = initialize_embeddings(
    provider="openai",
    model_name="text-embedding-3-small"
)

# Create embeddings
embeddings = await create_embeddings(texts, embeddings_model)

# Create vector store
vector_store = create_vector_store(
    documents,
    embeddings_model,
    store_type="faiss"
)
```

## Key Concepts Explained

### Document Loading
The process of reading and parsing different file formats into a standardized Document format that LangChain can process.

### Text Splitting
Breaking down large documents into smaller chunks that can be:
- Processed efficiently
- Fit within token limits
- Used for semantic search
- Analyzed independently

### Embeddings
Vector representations of text that capture semantic meaning, useful for:
- Semantic search
- Document comparison
- Clustering
- Recommendation systems

## Getting Started

1. Install dependencies:
```bash
pip install "unstructured[all-docs]"
pip install python-magic-bin  # For Windows
pip install faiss-cpu  # For vector storage
```

2. Basic document processing workflow:
```python
# 1. Load documents
documents = load_directory("your_docs/")

# 2. Split into chunks
chunks = split_documents(documents)

# 3. Create embeddings
embeddings_model = initialize_embeddings()
vector_store = create_vector_store(chunks, embeddings_model)
```

## Best Practices

### Document Loading
- Validate file types before loading
- Handle encoding issues gracefully
- Extract and preserve metadata
- Implement batch processing for large directories

### Text Splitting
- Choose appropriate chunk sizes for your use case
- Use overlap to maintain context
- Consider token limits of your LLM
- Test different splitting methods

### Embeddings
- Cache embeddings for reuse
- Monitor embedding costs
- Use appropriate vector stores
- Implement similarity thresholds

## Common Use Cases

1. **Document Search System**
```python
# Create searchable document base
docs = load_directory("knowledge_base/")
chunks = split_documents(docs)
vector_store = create_vector_store(chunks, embeddings_model)

# Search for similar documents
results = await find_similar_documents(
    "your query",
    vector_store
)
```

2. **Content Analysis**
```python
# Analyze document structure
chunks = split_text_recursively(text)
stats = get_chunk_statistics(chunks)

# Create and analyze embeddings
embeddings = await create_embeddings(chunks, embeddings_model)
stats = calculate_embedding_statistics(embeddings)
```

## Tips for Optimization

1. **Memory Efficiency**
   - Process large files in chunks
   - Clean up unused embeddings
   - Use appropriate batch sizes

2. **Speed Optimization**
   - Cache frequently used embeddings
   - Use async operations for I/O
   - Implement parallel processing

3. **Quality Improvement**
   - Validate document content
   - Clean text before processing
   - Use appropriate chunk sizes
   - Test different embedding models 