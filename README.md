# Python Tools MCP

A Model Context Protocol (MCP) server that exposes modern Python development tools.

## Features

- Manage Python dependencies using `uv`
- Run tests with `pytest` and coverage measurement
- Lint and format code with `ruff`
- Type checking with `basedpyright` or `mypy`
- Analyze and improve code quality with tools like `vulture`, `radon`, and `bandit`
- Check docstring coverage with `interrogate`
- Profile Python code with `py-spy`

## Installation

```bash
uv sync
```

## Usage

Start the server:

```bash
uvx python_tools_mcp
```

Then use the tools in your MCP-compatible application.

### Example Tools

- `pytest`: Run tests with optional coverage
- `ruff_check`: Lint Python code
- `ruff_format`: Format Python code
- `uv_add`: Add dependencies
- `uv_sync`: Install dependencies
- `coverage_analyze`: Analyze existing coverage data

## License

MIT

