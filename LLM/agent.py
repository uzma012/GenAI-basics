
from llm_client import get_groq_client
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from duckduckgo_search import DDGS

# 1. Define a working search tool
@tool
def search_tool(query: str) -> str:
    """Search the internet using DuckDuckGo."""
    results = list(DDGS().text(query, max_results=3))
    if results:
        return results[0]["body"]
    return "No results found."

# 2. Init Groq client wrapped as LangChain LLM
client = get_groq_client()
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",   
    temperature=0,
    openai_api_key=client.api_key,
    openai_api_base="https://api.groq.com/openai/v1"
)

# 3. Initialize agent with tools
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 4. Run a query with tool calling
resp = agent.run("Find the latest news about Groq AI chips")
print("\nFinal Answer:\n", resp)
