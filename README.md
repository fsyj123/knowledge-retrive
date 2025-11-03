# MCP Knowledge Tools

This repository implements a Model Context Protocol (MCP) server that exposes three
HTTP-backed retrieval tools for internal knowledge bases.

## Available tools

| Tool name | Description | Dataset ID |
|-----------|-------------|------------|
| `query_ux_knowledge` | Retrieves UX guidelines, templates, and examples. | `cab02597-6315-456c-92d3-19a65e3e7efd` |
| `query_lean_knowledge` | Retrieves Lean / Continuous Improvement documentation. | `67659dbe-4387-4122-8eb9-1d2005bea6a2` |
| `query_automation_step` | Retrieves automation process steps and related materials. | `b68de37f-a9f7-41fc-948f-eb89ca145770` |

Each tool sends a POST request to the Dify dataset retrieval API:

```
POST https://api.dify.ai/v1/datasets/{data_id}/retrieve
Authorization: Bearer <dataset token>
Content-Type: application/json
{"query": "<your question>"}
```

## Configuration

Set the dataset token through the `DIFY_DATASET_TOKEN` environment variable. A
fallback token (`dataset-gCRaKZgnKtvqLdeuoCFjKiME`) is bundled for quick testing,
but production deployments should override it.

```
export DIFY_DATASET_TOKEN="dataset-..."
```

## Installation

The project is published as a standard Python package and can be installed with
your preferred Python packaging tool.

### Using `uv`

[`uv`](https://docs.astral.sh/uv/) offers fast Python environment management and
package installation.

```
uv venv
source .venv/bin/activate
uv pip install .
```

### Using `pip`

```
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Running the MCP server

The `pyproject.toml` file defines a console script entry point. After installation
you can launch the server via:

```
mcp-knowledge-server
```

Alternatively you can run the module directly:

```
python -m mcp_knowledge_tools.server
```

The server uses `FastMCP` from the Model Context Protocol reference
implementation, so it can be registered with compatible MCP clients.
