# Sample Knowledge Base

This directory is for documents that the AI agent can search through using RAG (Retrieval-Augmented Generation).

## Supported File Types

- `.txt` — Plain text files
- `.md` — Markdown files

## How It Works

1. When the agent starts, it loads all `.txt` and `.md` files from this directory.
2. Each document is split into chunks of approximately 500 tokens.
3. Chunks are embedded using a local sentence-transformer model (`all-MiniLM-L6-v2`).
4. When Claude decides to search documents, the query is embedded and compared against all chunks.
5. The most relevant chunks are returned to Claude as context.

## Tips

- Add reference documents, documentation, or notes here.
- Larger files are automatically chunked, so don't worry about file size.
- The agent will only search when it decides it's relevant to the user's question.

## Example Content

### Python Best Practices

Python is a versatile programming language known for its readability and simplicity. Here are some key best practices:

1. **Use type hints**: Python 3.5+ supports type hints which make code more readable and enable static analysis tools like mypy.
2. **Follow PEP 8**: The Python Enhancement Proposal 8 (PEP 8) is the style guide for Python code. Use tools like `black` and `flake8` to enforce it.
3. **Write docstrings**: Use docstrings to document functions, classes, and modules. Follow the Google or NumPy style conventions.
4. **Use virtual environments**: Always use virtual environments (`venv`, `conda`) to isolate project dependencies.
5. **Handle errors gracefully**: Use try/except blocks and specific exception types. Avoid bare `except:` clauses.
6. **Use list comprehensions**: They are more Pythonic and often faster than equivalent for-loops.
7. **Leverage the standard library**: Python's standard library is extensive. Before adding a dependency, check if the stdlib has what you need.

### Anthropic SDK Quick Reference

The Anthropic Python SDK (`anthropic`) provides a clean interface to the Claude API:

- `client.messages.create()` — Send a message and get a response
- `client.messages.stream()` — Stream a response token by token
- `client.messages.parse()` — Get structured output validated against a schema
- Tool use — Define tools with JSON schemas, Claude decides when to call them
- Extended thinking — Enable with `thinking={"type": "adaptive"}` for deep reasoning
- Prompt caching — Use `cache_control={"type": "ephemeral"}` to cache large contexts
