from typing import List, Dict, Any, Optional
from langchain_core.tools import Tool
import sqlite3
import pandas as pd
import json
from datetime import datetime

def create_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Create a connection to SQLite database.
    
    Args:
        db_path (str): Path to SQLite database
        
    Returns:
        Optional[sqlite3.Connection]: Database connection or None if error
    """
    try:
        return sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def execute_query(conn: sqlite3.Connection, query: str) -> Optional[List[Dict[str, Any]]]:
    """
    Execute SQL query and return results.
    
    Args:
        conn (sqlite3.Connection): Database connection
        query (str): SQL query to execute
        
    Returns:
        Optional[List[Dict[str, Any]]]: Query results or None if error
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

def get_table_schema(conn: sqlite3.Connection, table_name: str) -> Optional[List[Dict[str, str]]]:
    """
    Get schema information for a table.
    
    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of the table
        
    Returns:
        Optional[List[Dict[str, str]]]: Schema information or None if error
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        return [
            {
                "name": col[1],
                "type": col[2],
                "nullable": not col[3],
                "primary_key": bool(col[5])
            }
            for col in columns
        ]
    except Exception as e:
        print(f"Error getting schema: {str(e)}")
        return None

def create_query_tool(db_path: str) -> Tool:
    """
    Create a tool for querying SQLite database.
    
    Args:
        db_path (str): Path to SQLite database
        
    Returns:
        Tool: Database query tool
    """
    def query_database(query: str) -> str:
        """Execute SQL query and return formatted results."""
        try:
            conn = create_connection(db_path)
            if not conn:
                return "Error: Could not connect to database"
            
            results = execute_query(conn, query)
            conn.close()
            
            if results is None:
                return "Error executing query"
            
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Tool(
        name="database_query",
        description="Execute SQL queries on the database. Input should be a valid SQL query.",
        func=query_database
    )

def create_schema_tool(db_path: str) -> Tool:
    """
    Create a tool for getting database schema information.
    
    Args:
        db_path (str): Path to SQLite database
        
    Returns:
        Tool: Schema information tool
    """
    def get_schema(table_name: str) -> str:
        """Get schema information for specified table."""
        try:
            conn = create_connection(db_path)
            if not conn:
                return "Error: Could not connect to database"
            
            schema = get_table_schema(conn, table_name)
            conn.close()
            
            if schema is None:
                return f"Error getting schema for table {table_name}"
            
            return json.dumps(schema, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Tool(
        name="database_schema",
        description="Get schema information for a database table. Input should be the table name.",
        func=get_schema
    )

def create_database_tools(db_path: str) -> List[Tool]:
    """
    Create all database-related tools.
    
    Args:
        db_path (str): Path to SQLite database
        
    Returns:
        List[Tool]: List of database tools
    """
    return [
        create_query_tool(db_path),
        create_schema_tool(db_path)
    ]

# Usage example
if __name__ == "__main__":
    async def main():
        # Create sample database
        db_path = "sample.db"
        conn = create_connection(db_path)
        
        if conn:
            # Create sample table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert sample data
            conn.execute("""
                INSERT OR REPLACE INTO users (name, email) VALUES 
                ('John Doe', 'john@example.com'),
                ('Jane Smith', 'jane@example.com')
            """)
            
            conn.commit()
            conn.close()
            
            # Create tools
            tools = create_database_tools(db_path)
            
            # Test query tool
            query_tool = next(t for t in tools if t.name == "database_query")
            schema_tool = next(t for t in tools if t.name == "database_schema")
            
            # Test queries
            test_cases = [
                ("Query all users", "SELECT * FROM users"),
                ("Get schema", "users")
            ]
            
            print("Testing Database Tools:")
            
            # Test query tool
            query_result = await query_tool.ainvoke(test_cases[0][1])
            print(f"\nQuery: {test_cases[0][1]}")
            print("Results:")
            print(query_result)
            
            # Test schema tool
            schema_result = await schema_tool.ainvoke(test_cases[1][1])
            print(f"\nTable: {test_cases[1][1]}")
            print("Schema:")
            print(schema_result) 