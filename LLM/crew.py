from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv

load_dotenv()

# Set API key in env
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Pass Groq model in one string
model_name = "groq/llama-3.1-8b-instant"

# Define Agents
search_agent = Agent(
    role="Searcher",
    goal="Find latest AI news",
    backstory="An expert in finding the latest developments in artificial intelligence.",
    llm=model_name  # 👈 use string
)

summarizer_agent = Agent(
    role="Summarizer",
    goal="Summarize news in simple terms",
    backstory="A specialist in distilling complex information into clear summaries.",
    llm=model_name
)

# Define Tasks
task1 = Task(
    description="Search about Groq AI chips",
    expected_output="A list of the most recent updates on Groq AI chips",
    agent=search_agent
)

task2 = Task(
    description="Summarize the search results",
    expected_output="A clear and concise summary of the Groq AI chip news",
    agent=summarizer_agent
)

# Orchestrate with Crew
crew = Crew(
    agents=[search_agent, summarizer_agent],
    tasks=[task1, task2]
)

result = crew.kickoff()
print(result)
