from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
def create_weather_tool() -> Tool:
    """Create a tool for fetching weather information."""
    
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        try:
            # Using a free weather API (replace with your preferred service)
            api_key = os.getenv("WEATHER_API_KEY")
            url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
            response = requests.get(url)
            data = response.json()
            
            return f"Weather in {location}: {data['current']['condition']['text']}, " \
                   f"Temperature: {data['current']['temp_c']}°C"
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    return Tool(
        name="weather",
        description="Get current weather information for a location",
        func=get_weather
    )

def create_data_analysis_tool() -> Tool:
    """Create a tool for basic data analysis."""
    
    def analyze_data(data_str: str) -> str:
        """Analyze numerical data and provide basic statistics."""
        try:
            # Convert string input to list of numbers
            data = [float(x) for x in data_str.split(',')]
            df = pd.Series(data)
            
            analysis = {
                "mean": df.mean(),
                "median": df.median(),
                "std": df.std(),
                "min": df.min(),
                "max": df.max()
            }
            
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
    
    return Tool(
        name="data_analysis",
        description="Analyze numerical data (comma-separated numbers)",
        func=analyze_data
    )

def create_time_tool() -> Tool:
    """Create a tool for time-related operations."""
    
    def get_time_info(timezone: str = "UTC") -> str:
        """Get current time information."""
        try:
            now = datetime.now()
            return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            return f"Error getting time: {str(e)}"
    
    return Tool(
        name="time_info",
        description="Get current time information",
        func=get_time_info
    )

def create_text_analysis_tool() -> Tool:
    """Create a tool for text analysis."""
    
    def analyze_text(text: str) -> str:
        """Perform basic text analysis."""
        try:
            analysis = {
                "character_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.splitlines()),
                "unique_words": len(set(text.lower().split()))
            }
            
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error analyzing text: {str(e)}"
    
    return Tool(
        name="text_analysis",
        description="Analyze text and provide statistics",
        func=analyze_text
    )

def combine_custom_tools() -> List[Tool]:
    """Combine all custom tools."""
    return [
        create_weather_tool(),
        create_data_analysis_tool(),
        create_time_tool(),
        create_text_analysis_tool()
    ]

async def test_tool(tool: Tool, input_value: str) -> str:
    """
    Test a specific tool with input.
    
    Args:
        tool (Tool): Tool to test
        input_value (str): Input for the tool
        
    Returns:
        str: Tool's output
    """
    try:
        return await tool.ainvoke(input_value)
    except Exception as e:
        return f"Error testing tool: {str(e)}"

def format_tool_result(tool_name: str, input_value: str, output: str) -> str:
    """
    Format tool test results.
    
    Args:
        tool_name (str): Name of the tool
        input_value (str): Input used
        output (str): Tool's output
        
    Returns:
        str: Formatted result
    """
    return f"""
Tool: {tool_name}
Input: {input_value}
Output: {output}
"""

# Usage example
if __name__ == "__main__":
    async def main():
        load_dotenv()
        
        # Create all custom tools
        tools = combine_custom_tools()
        
        # Test cases for each tool
        test_cases = [
            ("weather", "London"),
            ("data_analysis", "1,2,3,4,5,6,7,8,9,10"),
            ("time_info", "UTC"),
            ("text_analysis", "Hello world!\nThis is a test message.\nMultiple lines here.")
        ]
        
        # Run tests
        print("Testing Custom Tools:")
        for tool_name, test_input in test_cases:
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                result = await test_tool(tool, test_input)
                print(format_tool_result(tool_name, test_input, result))
            else:
                print(f"Tool '{tool_name}' not found")
        
        # Example of using tools with an agent
        llm = ChatOpenAI(temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant with access to various analysis tools."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Test the agent with a complex query
        query = """Analyze the following:
        1. Get the current weather in New York
        2. Analyze these numbers: 15,25,35,45,55
        3. Get the current time
        """
        
        response = await agent_executor.ainvoke({"input": query})
        print("\nAgent Test Results:")
        print(response["output"]) 
        
    # Actually run the async main function
    asyncio.run(main())