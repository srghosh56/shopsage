"""The assistant's tools as an MCP server (stdio). Uses the MCP Python SDK 2.x `MCPServer`.

Add to an MCP client with:
  {"command": "uv", "args": ["--directory", "<repo>", "run", "python", "-m",
                             "shopsage.agent.mcp_server"]}
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from shopsage.agent.tools import TOOLS


def build_server() -> MCPServer:
    server = MCPServer(
        name="shopsage",
        instructions=(
            "Tools over a UK online gift-ware wholesaler's data (UCI Online Retail II, "
            "2009-12-01 to 2011-12-09): customer profiles with RFM, segment and CLV; "
            "recommendations with evidence; explanations; similar items; product search; "
            "segment summaries; read-only SQL. All tools return JSON."
        ),
    )
    for fn in TOOLS:
        server.add_tool(fn)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
