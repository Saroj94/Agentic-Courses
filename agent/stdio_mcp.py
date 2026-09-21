from mcp.server.fastmcp import FastMCP

## initialize
mcp = FastMCP("Math")

@mcp.tool()
def sum(a:int, b:int)->int:
    """Add two numbers
    
    Args:
        a: int
        b: int
    """
    return a+b

@mcp.tool()
def multiply(a: int , b:int)->int:
    """
    Multiply two numbers
    """
    return a*b

## run mcp server
if __name__=="__main__":
    mcp.run(transport="stdio")