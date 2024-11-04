from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

def initialize_llm(model_name: str = "gpt-4", temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize an OpenAI LLM with specified parameters.
    
    Args:
        model_name (str): Name of the OpenAI model to use
        temperature (float): Temperature setting for response generation
        
    Returns:
        OpenAI: Initialized LangChain LLM instance
    """
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key,
       
    )

async def generate_response(llm, prompt: str) -> Optional[str]:
    """
    Generate a response using the provided LLM.
    
    Args:
        llm (OpenAI): The LangChain LLM instance to use
        prompt (str): The input prompt for the LLM
        
    Returns:
        Optional[str]: The generated response or None if an error occurs
    """
    try:
        response = await llm.ainvoke(prompt)
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    llm = initialize_llm()
    
    async def main():
        response = await generate_response(
            llm,
            "Explain what LangChain is in one sentence."
        )
        print(response)

    asyncio.run(main())