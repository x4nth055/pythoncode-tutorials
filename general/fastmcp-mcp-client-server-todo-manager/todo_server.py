from typing import Literal
from itertools import count
from datetime import datetime, timezone
from fastmcp import FastMCP

# In-memory storage for demo purposes
TODOS: list[dict] = []
_id = count(start=1)

mcp = FastMCP(name="Todo Manager")

@mcp.tool
def create_todo(
    title: str,
    description: str = "",
    priority: Literal["low", "medium", "high"] = "medium",
) -> dict:
    """Create a todo (id, title, status, priority, timestamps)."""
    todo = {
        "id": next(_id),
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    TODOS.append(todo)
    return todo

@mcp.tool
def list_todos(status: Literal["open", "done", "all"] = "open") -> dict:
    """List todos by status ('open' | 'done' | 'all')."""
    if status == "all":
        items = TODOS
    elif status == "open":
        items = [t for t in TODOS if t["status"] == "open"]
    else:
        items = [t for t in TODOS if t["status"] == "done"]
    return {"items": items}

@mcp.tool
def complete_todo(todo_id: int) -> dict:
    """Mark a todo as done."""
    for t in TODOS:
        if t["id"] == todo_id:
            t["status"] = "done"
            t["completed_at"] = datetime.now(timezone.utc).isoformat()
            return t
    raise ValueError(f"Todo {todo_id} not found")

@mcp.tool
def search_todos(query: str) -> dict:
    """Case-insensitive search in title/description."""
    q = query.lower().strip()
    items = [t for t in TODOS if q in t["title"].lower() or q in t["description"].lower()]
    return {"items": items}

# Read-only resources
@mcp.resource("stats://todos")
def todo_stats() -> dict:
    """Aggregated stats: total, open, done."""
    total = len(TODOS)
    open_count = sum(1 for t in TODOS if t["status"] == "open")
    done_count = total - open_count
    return {"total": total, "open": open_count, "done": done_count}

@mcp.resource("todo://{id}")
def get_todo(id: int) -> dict:
    """Fetch a single todo by id."""
    for t in TODOS:
        if t["id"] == id:
            return t
    raise ValueError(f"Todo {id} not found")

# A reusable prompt
@mcp.prompt
def suggest_next_action(pending: int, project: str | None = None) -> str:
    """Render a small instruction for an LLM to propose next action."""
    base = f"You have {pending} pending TODOs. "
    if project:
        base += f"They relate to the project '{project}'. "
    base += "Suggest the most impactful next action in one short sentence."
    return base

if __name__ == "__main__":
    # Default transport is stdio; you can also use transport="http", host=..., port=...
    mcp.run()
