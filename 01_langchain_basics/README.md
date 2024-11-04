# LangChain Basics

This directory contains fundamental implementations of LangChain concepts. Each file focuses on a specific core concept to help you get started with LangChain.

## Files Overview

### 1. basic_llm.py
This file introduces you to working with Large Language Models (LLMs) in LangChain.
- How to initialize an LLM
- Basic prompt handling
- Async response generation
- Error handling for API calls

Key concepts:
```python
# Initialize an LLM
llm = initialize_llm(model_name="gpt-4", temperature=0.7)

# Generate responses
response = await generate_response(llm, "Your prompt here")
```

### 2. prompt_templates.py
Learn how to create and manage prompt templates for consistent LLM interactions.
- Basic prompt templates
- Structured output templates
- Template management
- Output parsing

Example usage:
```python
template = create_basic_template(
    template="Summarize this: {input_text}",
    input_variables=["input_text"],
    template_name="summarizer"
)
```

### 3. chains_basic.py
Introduction to LangChain's chain concept for building workflows.
- Creating simple chains
- Chain management
- Async chain execution
- Input/output handling

Example:
```python
chain = create_chain(
    llm=llm,
    template="Translate {text} to {language}",
    input_variables=["text", "language"],
    chain_name="translator"
)
```

### 4. advanced_chains.py
Advanced chain concepts and complex workflows.
- Sequential chains
- Transform chains
- Multiple chain integration
- JSON parsing and formatting
- Complex workflow management

Example workflow:
```python
results = await run_advanced_chain_workflow(
    input_text="Your text here",
    llm=llm
)
```

## Key Concepts Explained

### Chains
Chains are sequences of operations that can be combined to create complex workflows. Think of them like a pipeline where each step processes the output of the previous step.

### Prompt Templates
Templates help you structure your prompts consistently. They're like reusable forms where you can fill in different variables each time.

### LLMs
Large Language Models are the AI models that power your applications. You can think of them as very sophisticated text processing engines.

## Getting Started

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
```bash
OPENAI_API_KEY=your_api_key_here
```

3. Start with basic_llm.py to understand the fundamentals
4. Move on to prompt_templates.py to learn about structuring prompts
5. Explore chains_basic.py for simple workflows
6. Finally, dive into advanced_chains.py for complex implementations

## Tips for Beginners

- Always start with simple chains before moving to complex workflows
- Test your prompts with different inputs to understand their behavior
- Use lower temperatures (0.0-0.7) for more focused responses
- Keep your chain steps modular for better maintenance
- Use type hints and docstrings to make your code more readable 