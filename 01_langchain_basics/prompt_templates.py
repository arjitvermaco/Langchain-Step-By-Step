from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field

# Dictionary to store created templates
templates: Dict[str, PromptTemplate] = {}

# Dictionary to store output parsers 
output_parsers: Dict[str, StructuredOutputParser] = {}

def create_basic_template(
    template: str,
    input_variables: List[str],
    template_name: str
) -> PromptTemplate:
    """
    Create and store a basic prompt template.
    
    Args:
        template (str): The template string
        input_variables (List[str]): List of input variables
        template_name (str): Name to identify the template
        
    Returns:
        PromptTemplate: The created prompt template
    """
    prompt_template = PromptTemplate(
        input_variables=input_variables,
        template=template
    )
    templates[template_name] = prompt_template
    return prompt_template

def create_structured_template(
    template: str,
    input_variables: List[str], 
    output_schemas: List[ResponseSchema],
    template_name: str
) -> PromptTemplate:
    """
    Create a template with structured output parsing.
    
    Args:
        template (str): The template string
        input_variables (List[str]): List of input variables
        output_schemas (List[ResponseSchema]): Schema for structured output
        template_name (str): Name to identify the template
        
    Returns:
        PromptTemplate: The created prompt template with parser
    """
    parser = StructuredOutputParser.from_response_schemas(output_schemas)
    format_instructions = parser.get_format_instructions()
    
    template_with_parser = template + "\n\n{format_instructions}"
    prompt_template = PromptTemplate(
        input_variables=input_variables + ["format_instructions"],
        template=template_with_parser,
        partial_variables={"format_instructions": format_instructions}
    )
    
    templates[template_name] = prompt_template
    output_parsers[template_name] = parser
    return prompt_template

def get_template(template_name: str) -> Optional[PromptTemplate]:
    """
    Retrieve a stored template by name.
    
    Args:
        template_name (str): Name of the template to retrieve
        
    Returns:
        Optional[PromptTemplate]: The requested template or None if not found
    """
    return templates.get(template_name)

def get_parser(template_name: str) -> Optional[StructuredOutputParser]:
    """
    Retrieve a stored output parser by template name.
    
    Args:
        template_name (str): Name of the template whose parser to retrieve
        
    Returns:
        Optional[StructuredOutputParser]: The parser or None if not found
    """
    return output_parsers.get(template_name)

# Example usage
if __name__ == "__main__":
    # Create a basic template
    basic_template = """
    Create a summary of the following text:
    
    Text: {input_text}
    
    Summary:
    """
    
    create_basic_template(
        template=basic_template,
        input_variables=["input_text"],
        template_name="summarizer"
    )
    
    # Create a structured template
    analysis_schemas = [
        ResponseSchema(name="sentiment", description="The sentiment of the text (positive/negative/neutral)"),
        ResponseSchema(name="key_points", description="Main points from the text as a list"),
        ResponseSchema(name="word_count", description="Number of words in the text")
    ]
    
    analysis_template = """
    Analyze the following text and provide structured information:
    
    Text: {input_text}
    
    Provide a detailed analysis including sentiment, key points, and word count.
    """
    
    create_structured_template(
        template=analysis_template,
        input_variables=["input_text"],
        output_schemas=analysis_schemas,
        template_name="structured_analyzer"
    )