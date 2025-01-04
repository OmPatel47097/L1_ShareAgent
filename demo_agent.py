import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app
# Load environment variables from .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

# web search agent
internet_search_agaent = Agent(
    name = "Web Search Agent", 
    role = "Search the internet for the information as a expert researcher",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always provide sources for the information you find", "Use the tools to find the most accurate information"],
    show_tool_calls = True,
    markdown = True)

# Equity Lens - Finance/Stock Market Agent
equity_lens_agent = Agent(
    name = "Equity Lens Agent", 
    role = "You are the expert in the stock market and finance domain",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_fundamentals=True, stock_price=True, company_info=True, technical_indicators=True)],
    instructions = ["Use tables to display the data you get (where necesary)"],
    show_tool_calls = True,
    markdown = True)

app=Playground(agents=[equity_lens_agent,internet_search_agaent]).get_app()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("demo:app", host="127.0.0.1", port=8080, reload=True)

