from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.file_management import WriteFileTool, ReadFileTool
# from langchain.tools.python.tool import PythonAstREPLTool
from dotenv import load_dotenv
import os
import asyncio

def initialize_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Initialize the ChatOpenAI model."""
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_search_tools() -> List[Tool]:
    """Create web search and information retrieval tools."""
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    return [
        Tool(
            name="web_search",
            description="Search the web for current information",
            func=search.run
        ),
        Tool(
            name="wikipedia",
            description="Search Wikipedia articles for detailed information",
            func=wikipedia.run
        )
    ]

def create_file_tools() -> List[Tool]:
    """Create file management tools."""
    return [
        WriteFileTool(),
        ReadFileTool()
    ]

def create_python_tools() -> List[Tool]:
    """Create Python execution and calculation tools."""
    return [
        Tool(
            name="python_repl",
            description="Execute Python code for calculations or data processing",
            func=PythonREPLTool().run
        ),
        Tool(
            name="python_ast",
            description="Safely execute Python code with AST parsing",
            func=PythonREPLTool().run
        )
    ]

def create_advanced_prompt() -> ChatPromptTemplate:
    """Create a detailed prompt template for the multi-tool agent."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AI assistant with access to multiple tools.
        For each task:
        1. Break down complex requests into steps
        2. Choose the most appropriate tool for each step
        3. Explain your reasoning and tool selection
        4. Provide detailed results with proper formatting
        5. Handle errors gracefully and try alternative approaches
        
        Available tools include:
        - Web search for current information
        - Wikipedia for detailed knowledge
        - Python execution for calculations
        - File operations for data management"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

def setup_multi_tool_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    prompt: ChatPromptTemplate
) -> AgentExecutor:
    """Set up an agent with multiple tools."""
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

async def execute_multi_step_task(
    agent: AgentExecutor,
    task: str
) -> Optional[Dict[str, Any]]:
    """
    Execute a complex task using multiple tools.
    
    Args:
        agent (AgentExecutor): Configured agent
        task (str): Complex task description
        
    Returns:
        Optional[Dict[str, Any]]: Task results and execution steps
    """
    try:
        response = await agent.ainvoke({
            "input": task,
        })
        return {
            "output": response["output"],
            "steps": response.get("intermediate_steps", []),
            "tools_used": set(step[0].name for step in response.get("intermediate_steps", []))
        }
    except Exception as e:
        print(f"Error executing task: {str(e)}")
        return None

def format_execution_results(results: Optional[Dict[str, Any]]) -> str:
    """
    Format the execution results with detailed step information.
    
    Args:
        results (Optional[Dict[str, Any]]): Task execution results
        
    Returns:
        str: Formatted results
    """
    if not results:
        return "Task execution failed."
    
    formatted = f"Final Output:\n{results['output']}\n\nExecution Steps:"
    
    for i, (tool, result) in enumerate(results["steps"], 1):
        formatted += f"\n\nStep {i}:"
        formatted += f"\nTool Used: {tool.name}"
        formatted += f"\nTool Description: {tool.description}"
        formatted += f"\nResult: {result[:200]}..."
    
    formatted += f"\n\nTools Used: {', '.join(results['tools_used'])}"
    return formatted

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize components
        llm = initialize_llm()
        
        # Combine all tools
        tools = (
            create_search_tools() +
            create_file_tools() +
            create_python_tools()
        )
        
        # Set up agent
        prompt = create_advanced_prompt()
        agent = setup_multi_tool_agent(llm, tools, prompt)
        
        # Example complex tasks
        tasks = [
            """Calculate the population density of France:
            1. Find the current population of France
            2. Find the total area of France in square kilometers
            3. Calculate the density and format it nicely""",
            
            """Create a summary of recent AI developments:
            1. Search for recent AI news
            2. Save the findings to a file
            3. Create a Python script to analyze common themes"""
        ]
        
        # Execute tasks
        for task in tasks:
            print(f"\nExecuting Task: {task}\n")
            results = await execute_multi_step_task(agent, task)
            print(format_execution_results(results))

    # Run the async main function
    asyncio.run(main())