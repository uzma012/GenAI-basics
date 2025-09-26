import os
import json
from llm_client import get_groq_client


# Initialize Groq client
client = get_groq_client()

def query_with_mcp(prompt: str, mcp_servers: list, model: str = "llama-3.3-70b-versatile"):
    """Query Groq model with MCP server tools."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "mcp",
                "server_label": server["label"],
                "server_url": server["url"],
                "server_description": server.get("description", ""),
                "require_approval": server.get("require_approval", "never"),
                "headers": server.get("headers", {})
            }
            for server in mcp_servers
        ],
        stream=False
    )
    return response

if __name__ == "__main__":
    # Register local MCP server
    mcp_servers = [
        {
            "label": "weather_server",
            "url": "http://127.0.0.1:5001/weather",
            "description": "Provides weather information from natural language queries",
            "require_approval": "never"
        }
    ]

    user_prompt = "What's the weather like in Paris tomorrow?"
    result = query_with_mcp(user_prompt, mcp_servers)

    # Pretty print JSON response
    print(json.dumps(result, indent=2))
