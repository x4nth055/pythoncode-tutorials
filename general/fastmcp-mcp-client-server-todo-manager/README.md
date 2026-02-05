# Build a real MCP client and server in Python with FastMCP (Todo Manager example)

This folder contains the code that accompanies the article:

- Article: https://www.thepythoncode.com/article/fastmcp-mcp-client-server-todo-manager

What’s included
- `todo_server.py`: FastMCP MCP server exposing tools, resources, and a prompt for a Todo Manager.
- `todo_client_test.py`: A small client script that connects to the server and exercises all features.
- `requirements.txt`: Python dependencies for this tutorial.

Quick start
1) Install requirements
```bash
python -m venv .venv && source .venv/bin/activate  # or use your preferred env manager
pip install -r requirements.txt
```

2) Run the server (stdio transport by default)
```bash
python todo_server.py
```

3) In a separate terminal, run the client
```bash
python todo_client_test.py
```

Optional: run the server over HTTP
- In `todo_server.py`, replace the last line with:
```python
mcp.run(transport="http", host="127.0.0.1", port=8000)
```
- Then change the client constructor to `Client("http://127.0.0.1:8000/mcp")`.

Notes
- Requires Python 3.10+.
- The example uses in-memory storage for simplicity.
- For production tips (HTTPS, auth, containerization), see the article.
