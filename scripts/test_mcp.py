"""Test the MCP server's tools in-process (no stdio transport needed)."""

import asyncio

from fastmcp import Client

from mcp_server.server import mcp


async def main():
    async with Client(mcp) as client:
        # 1. List what the server exposes
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])
        prompts = await client.list_prompts()
        print("Prompts:", [p.name for p in prompts])

        # 2. Call list_sources
        print("\n--- list_sources ---")
        res = await client.call_tool("list_sources", {})
        print(res.data)

        # 3. Call search_brain
        print("\n--- search_brain: 'how to use interrupts' ---")
        res = await client.call_tool(
            "search_brain", {"query": "how to use interrupts", "top_k": 3}
        )
        for chunk in res.data:
            print(
                f"  [{chunk['source_type']}] trust={chunk['trust_score']:.2f} "
                f"score={chunk['vector_score']:.3f}  {chunk['title']}"
            )

        # 4. Call search_brain with a trust filter (docs-quality only)
        print("\n--- search_brain: docs only, min_trust=0.9 ---")
        res = await client.call_tool(
            "search_brain",
            {"query": "how to use interrupts", "top_k": 3, "min_trust": 0.9},
        )
        for chunk in res.data:
            print(
                f"  [{chunk['source_type']}] trust={chunk['trust_score']:.2f}  "
                f"{chunk['title']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
