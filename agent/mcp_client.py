from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
load_dotenv()

## mcp client
async def main():
    client = MultiServerMCPClient(
        {
            "math_server": {
                "command":"python",
                ## Make sure to update to the full absolute path to your math_server.py file
                "args": ["/Users/sarojrai/Desktop/AI Agents/Agentic-Courses/agent/stdio_mcp.py"],
                "transport": "stdio" 
            },
            "weather_server": {
                "url": "http://localhost:8000/mcp", ## /mcp ensures that all mcp server is running
                "transport": "streamable_http"
            }

        }
    )

    ## now mcp client has access of all tools
    ## initialize the mcp client
    tool_client = await client.get_tools()
    llm = ChatGroq(model="openai/gpt-oss-120b", max_tokens=456, reasoning_effort="medium")
    agent = create_agent(
        llm,
        tools=tool_client,   
    )
    math_res = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})

    weather_res = await agent.ainvoke({"messages":[{"role":"user", "content": "what is the weather in nyc"}]})

    print("Math response: ", {math_res["messages"][-1].content})

    print("Weather response: ", {weather_res["messages"][-1].content})

asyncio.run(main())