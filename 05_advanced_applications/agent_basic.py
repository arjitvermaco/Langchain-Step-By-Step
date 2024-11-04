from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from dotenv import load_dotenv
import os
import asyncio

def initialize_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize the ChatOpenAI model.
    
    Args:
        temperature (float): Temperature for response generation
        
    Returns:
        ChatOpenAI: Initialized chat model
    """
    load_dotenv()
    return ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_basic_tools() -> List[Tool]:
    """
    Create a list of basic tools for the agent.
    
    Returns:
        List[Tool]: List of available tools
    """
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
            description="Search Wikipedia articles for information",
            func=wikipedia.run
        )
    ]

def create_agent_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for the agent.
    
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to various tools.
        Use these tools to provide accurate and up-to-date information.
        Always explain your thought process and cite sources when possible."""),
        # MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

def setup_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    prompt: ChatPromptTemplate
) -> AgentExecutor:
    """
    Set up the agent with tools and prompt.
    
    Args:
        llm (ChatOpenAI): Language model
        tools (List[Tool]): Available tools
        prompt (ChatPromptTemplate): Prompt template
        
    Returns:
        AgentExecutor: Configured agent executor
    """
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

async def run_agent(
    agent: AgentExecutor,
    query: str
) -> Optional[Dict[str, Any]]:
    """
    Run the agent with a query.
    
    Args:
        agent (AgentExecutor): Configured agent
        query (str): User query
        
    Returns:
        Optional[Dict[str, Any]]: Agent's response and actions
    """
    try:
        response = await agent.ainvoke({"input": query})
        return {
            "output": response["output"],
            "steps": response.get("intermediate_steps", [])
        }
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        return None

def format_agent_response(response: Optional[Dict[str, Any]]) -> str:
    """
    Format the agent's response for display.
    
    Args:
        response (Optional[Dict[str, Any]]): Agent's response
        
    Returns:
        str: Formatted response
    """
    if not response:
        return "Sorry, I encountered an error while processing your request."
    
    formatted = f"Answer: {response['output']}\n\nSteps taken:"
    
    for i, (tool, result) in enumerate(response["steps"], 1):
        formatted += f"\n\n{i}. Used tool: {tool.name}"
        formatted += f"\n   Result: {result[:200]}..."
    
    return formatted

# Usage example
if __name__ == "__main__":
    async def main():
        # Initialize components
        llm = initialize_llm()
        tools = create_basic_tools()
        prompt = create_agent_prompt()
        
        # Set up agent
        agent = setup_agent(llm, tools, prompt)
        
        # Example queries
        queries = [
            "What is the capital of France and what's the current weather there?",
            "Who won the latest FIFA World Cup and tell me about the final match?"
        ]
        
        # Run queries
        for query in queries:
            print(f"\nQuery: {query}")
            response = await run_agent(agent, query)
            print(format_agent_response(response))
            
    # Actually run the async main function
    asyncio.run(main())