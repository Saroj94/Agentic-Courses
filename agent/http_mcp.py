from mcp.server.fastmcp import FastMCP

## initialize
mcp = FastMCP("weather")

## tool
@mcp.tool()
async def get_weather(location: str)-> str:
    """Get the weather information of given location"""
    return "Today is rainy all day."



## run
if __name__=="__main__":
    mcp.run(transport="streamable-http")
    