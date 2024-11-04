from typing import Optional, Dict, Any
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio

# Dictionary to store chains
chains: Dict[str, LLMChain] = {}

def initialize_llm(temperature: float = 0.7) -> OpenAI:
    """
    Initialize the OpenAI language model.
    
    Args:
        temperature (float): Temperature setting for response generation
        
    Returns:
        OpenAI: Initialized language model
    """
    load_dotenv()
    return OpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_chain(
    llm: OpenAI,
    template: str,
    input_variables: list[str],
    chain_name: str
) -> LLMChain:
    """
    Create a LangChain chain with a custom prompt template.
    
    Args:
        llm (OpenAI): The language model to use
        template (str): The prompt template string
        input_variables (list[str]): List of input variables for the template
        chain_name (str): Name to identify the chain
        
    Returns:
        LLMChain: The created chain
    """
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    chains[chain_name] = chain
    return chain

async def run_chain(
    chain_name: str,
    inputs: Dict[str, Any]
) -> Optional[str]:
    """
    Run a specific chain with given inputs.
    
    Args:
        chain_name (str): Name of the chain to run
        inputs (Dict[str, Any]): Input variables for the chain
        
    Returns:
        Optional[str]: The chain's output or None if an error occurs
    """
    try:
        chain = chains.get(chain_name)
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        response = await chain.ainvoke(inputs)
        return response["text"]
    except Exception as e:
        print(f"Error running chain: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    llm = initialize_llm()
    
    # Create a simple translation chain
    template = """Translate the following text from {source_lang} to {target_lang}:
    
    Text: {text}
    
    Translation:"""
    
    create_chain(
        llm=llm,
        template=template,
        input_variables=["source_lang", "target_lang", "text"],
        chain_name="translator"
    )
    
    async def main():
        response = await run_chain(
            "translator",
            {
                "source_lang": "English",
                "target_lang": "French",
                "text": "Hello, how are you?"
            }
        )
        print(response)
    
    # Run the async main function
    asyncio.run(main())