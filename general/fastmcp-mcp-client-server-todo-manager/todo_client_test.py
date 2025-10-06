import asyncio
from fastmcp import Client

async def main():
    # Option A: Connect to local Python script (stdio)
    client = Client("todo_server.py")

    # Option B: In-memory (for tests)
    # from todo_server import mcp
    # client = Client(mcp)

    async with client:
        await client.ping()
        print("[OK] Connected")

        # Create a few todos
        t1 = await client.call_tool("create_todo", {"title": "Write README", "priority": "high"})
        t2 = await client.call_tool("create_todo", {"title": "Refactor utils", "description": "Split helpers into modules"})
        t3 = await client.call_tool("create_todo", {"title": "Add tests", "priority": "low"})
        print("Created IDs:", t1.data["id"], t2.data["id"], t3.data["id"])

        # List open
        open_list = await client.call_tool("list_todos", {"status": "open"})
        print("Open IDs:", [t["id"] for t in open_list.data["items"]])

        # Complete one
        updated = await client.call_tool("complete_todo", {"todo_id": t2.data["id"]})
        print("Completed:", updated.data["id"], "status:", updated.data["status"])

        # Search
        found = await client.call_tool("search_todos", {"query": "readme"})
        print("Search 'readme':", [t["id"] for t in found.data["items"]])

        # Resources
        stats = await client.read_resource("stats://todos")
        print("Stats:", getattr(stats[0], "text", None) or stats[0])

        todo2 = await client.read_resource(f"todo://{t2.data['id']}")
        print("todo://{id}:", getattr(todo2[0], "text", None) or todo2[0])

        # Prompt
        prompt_msgs = await client.get_prompt("suggest_next_action", {"pending": 2, "project": "MCP tutorial"})
        msgs_pretty = [
            {"role": m.role, "content": getattr(m, "content", None) or getattr(m, "text", None)}
            for m in getattr(prompt_msgs, "messages", [])
        ]
        print("Prompt messages:", msgs_pretty)

if __name__ == "__main__":
    asyncio.run(main())
