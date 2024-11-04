# 01_langchain_basics/advanced_chains.py
from typing import Dict, Any
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
import os

def initialize_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Initialize the language model."""
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_extraction_chain(llm: ChatOpenAI) -> LLMChain:
    """Create a chain for extracting key information from text."""
    template = """Extract key information from the following text:
    
    Text: {input_text}
    
    Extract and format the following information as JSON:
    - main_topics: List of main topics discussed
    - key_points: List of key points
    - entities: List of important entities (people, organizations, etc.)
    """
    
    prompt = PromptTemplate(
        input_variables=["input_text"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="extracted_info")

def create_analysis_chain(llm: ChatOpenAI) -> LLMChain:
    """Create a chain for analyzing the extracted information."""
    template = """Analyze the following extracted information:
    
    {extracted_info}
    
    Provide a detailed analysis including:
    1. Relationships between topics
    2. Key insights
    3. Potential implications
    """
    
    prompt = PromptTemplate(
        input_variables=["extracted_info"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="analysis_info")

def create_summary_chain(llm: ChatOpenAI) -> LLMChain:
    """Create a chain for generating a final summary."""
    template = """Based on the analysis and extracted information:
    
    Extracted Information: {extracted_info}
    Analysis: {analysis_info}
    
    Generate a concise summary that captures the most important points and insights.
    """
    
    prompt = PromptTemplate(
        input_variables=["extracted_info", "analysis_info"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="summary")

def create_json_parsing_chain() -> TransformChain:
    """Create a chain for parsing and formatting JSON data."""
    def transform_json(inputs: Dict[str, str]) -> Dict[str, Any]:
        try:
            if isinstance(inputs["extracted_info"], str):
                parsed_data = json.loads(inputs["extracted_info"])
            else:
                parsed_data = inputs["extracted_info"]
            
            return {
                "parsed_data": parsed_data
            }
        except json.JSONDecodeError:
            return {
                "parsed_data": {"error": "Failed to parse JSON"}
            }
    
    return TransformChain(
        input_variables=["extracted_info"],
        output_variables=["parsed_data"],
        transform=transform_json
    )

async def run_advanced_chain_workflow(input_text: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Run the complete chain workflow."""
    try:
        extraction_chain = create_extraction_chain(llm)
        json_chain = create_json_parsing_chain()
        analysis_chain = create_analysis_chain(llm)
        summary_chain = create_summary_chain(llm)
        
        sequential_chain = SequentialChain(
            chains=[extraction_chain, json_chain, analysis_chain, summary_chain],
            input_variables=["input_text"],
            output_variables=["extracted_info", "parsed_data", "analysis_info", "summary"],
            return_all=True
        )
        
        results = await sequential_chain.ainvoke({"input_text": input_text})
        return results
    
    except Exception as e:
        print(f"Error in chain workflow: {str(e)}")
        return {}

def format_chain_results(results: Dict[str, Any]) -> str:
    """Format the results from the chain workflow."""
    if not results:
        return "Error processing the workflow"
    
    formatted = "Chain Workflow Results:\n\n"
    
    if "parsed_data" in results:
        formatted += "Extracted Information:\n"
        formatted += json.dumps(results["parsed_data"], indent=2)
        formatted += "\n\n"
    
    if "analysis_info" in results:
        formatted += "Analysis:\n"
        formatted += results["analysis_info"]
        formatted += "\n\n"
    
    if "summary" in results:
        formatted += "Final Summary:\n"
        formatted += results["summary"]
    
    return formatted

# Usage example
if __name__ == "__main__":
    import asyncio
    async def main():
        llm = initialize_llm()
        
        input_text = """
        Artificial Intelligence has transformed various industries in recent years.
        Companies like OpenAI, Google, and Microsoft are leading the development of
        large language models. These models have applications in healthcare,
        finance, and education. The technology raises important ethical
        considerations regarding privacy, bias, and accountability. Researchers are
        working on making AI systems more transparent and explainable.
        """
        
        results = await run_advanced_chain_workflow(input_text, llm)
        print(format_chain_results(results))

    asyncio.run(main())