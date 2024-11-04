from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from database_tools import create_database_tools
from dotenv import load_dotenv
import os

def initialize_ai_components(db_path: str) -> tuple[ChatOpenAI, List[Any], ChatPromptTemplate]:
    """
    Initialize AI components for database interaction.
    
    Args:
        db_path (str): Path to SQLite database
        
    Returns:
        tuple: (AI model, database tools, prompt template)
    """
    load_dotenv()
    
    # Initialize AI model
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get database tools
    tools = create_database_tools(db_path)
    
    # Create specialized prompt for database operations
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database expert AI assistant. Help users interact with the SQLite database by:
        1. Understanding natural language queries
        2. Converting them to proper SQL
        3. Using the appropriate database tools
        4. Explaining the results in a clear way
        
        Available tools:
        - database_query: Execute SQL queries
        - database_schema: Get table schema information
        
        Always check the schema before creating queries. Explain your thought process."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    return llm, tools, prompt

def setup_database_agent(
    llm: ChatOpenAI,
    tools: List[Any],
    prompt: ChatPromptTemplate
) -> AgentExecutor:
    """
    Set up an agent specialized in database operations.
    
    Args:
        llm (ChatOpenAI): Language model
        tools (List[Any]): Database tools
        prompt (ChatPromptTemplate): Prompt template
        
    Returns:
        AgentExecutor: Configured database agent
    """
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

async def process_database_query(
    agent: AgentExecutor,
    query: str
) -> Optional[Dict[str, Any]]:
    """
    Process a natural language database query.
    
    Args:
        agent (AgentExecutor): Database agent
        query (str): Natural language query
        
    Returns:
        Optional[Dict[str, Any]]: Query results and agent's explanation
    """
    try:
        response = await agent.ainvoke({"input": query})
        return {
            "result": response["output"],
            "steps": response.get("intermediate_steps", [])
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None

def format_database_response(response: Optional[Dict[str, Any]]) -> str:
    """
    Format the database query response.
    
    Args:
        response (Optional[Dict[str, Any]]): Query response
        
    Returns:
        str: Formatted response
    """
    if not response:
        return "Sorry, I couldn't process your database query."
    
    formatted = f"Results:\n{response['result']}\n\nSteps Taken:"
    
    for i, (tool, result) in enumerate(response["steps"], 1):
        formatted += f"\n\nStep {i}:"
        formatted += f"\nAction: {tool.name}"
        if len(result) > 200:
            formatted += f"\nResult: {result[:200]}..."
        else:
            formatted += f"\nResult: {result}"
    
    return formatted

# Usage example
if __name__ == "__main__":
    async def main():
        # Database path
        db_path = "sample.db"
        
        # Initialize components
        llm, tools, prompt = initialize_ai_components(db_path)
        
        # Set up database agent
        agent = setup_database_agent(llm, tools, prompt)
        
        # Example natural language queries
        queries = [
            "What tables are in the database?",
            "Show me all users and their email addresses",
            "How many users are in the database?",
            "Find users created in the last 24 hours",
            "What's the structure of the users table?"
        ]
        
        # Process queries
        print("AI Database Assistant Demo:\n")
        
        for query in queries:
            print(f"\nQuestion: {query}")
            response = await process_database_query(agent, query)
            print(format_database_response(response))
            print("\n" + "="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 