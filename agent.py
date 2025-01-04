from phi.agent import Agent
import yfinance as yf
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    instructions = ["Use tables to display the data you get"],
    show_tool_calls = True,
    markdown = True)

multi_agent = Agent(
    model = Groq(id = "llama-3.3-70b-versatile"),
    team=[internet_search_agaent, equity_lens_agent],
    instructions = ["Always provide sources for the information", "Use tables to display the data"],
    show_tool_calls=True,
    markdown = True)

multi_agent.print_response("Serach about the AAPL and share latest news for AAPL.", stream=True)