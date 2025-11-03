"""Model Context Protocol server exposing internal knowledge retrieval tools."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import httpx
from mcp.server.fastmcp import FastMCP

_DATASET_TOKEN_ENV = "DIFY_DATASET_TOKEN"
_DEFAULT_DATASET_TOKEN = "dataset-gCRaKZgnKtvqLdeuoCFjKiME"
_DATASET_URL_TEMPLATE = "https://api.dify.ai/v1/datasets/{data_id}/retrieve"

_UX_DATASET_ID = "cab02597-6315-456c-92d3-19a65e3e7efd"
_LEAN_DATASET_ID = "67659dbe-4387-4122-8eb9-1d2005bea6a2"
_AUTOMATION_DATASET_ID = "b68de37f-a9f7-41fc-948f-eb89ca145770"

mcp = FastMCP("internal-knowledge-retriever")


def _build_headers() -> Dict[str, str]:
    """Build HTTP headers for dataset API requests."""

    token = os.getenv(_DATASET_TOKEN_ENV, _DEFAULT_DATASET_TOKEN)
    if not token:
        raise RuntimeError(
            "A dataset token is required. Set the DIFY_DATASET_TOKEN environment variable."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _format_documents(items: Iterable[Any]) -> str:
    """Convert a list of dataset documents into a human readable string."""

    parts: List[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            content = item.get("content") or item.get("answer") or item.get("text")
            metadata = item.get("metadata") or {}
            score = item.get("score")
            section_lines = [f"{index}. {content}" if content else f"{index}. (no content)"]

            if score is not None:
                section_lines.append(f"   score: {score}")
            if metadata:
                section_lines.append(f"   metadata: {json.dumps(metadata, ensure_ascii=False)}")

            parts.append("\n".join(section_lines))
        else:
            parts.append(f"{index}. {item}")

    return "\n\n".join(parts)


def _format_payload(payload: Dict[str, Any]) -> str:
    """Format the API payload for return to the client."""

    for key in ("items", "data", "documents", "results"):
        items = payload.get(key)
        if isinstance(items, list):
            if not items:
                return "No matching documents were found."
            formatted = _format_documents(items)
            if formatted.strip():
                return formatted

    # Fall back to returning the raw payload for unexpected shapes.
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _query_dataset(query: str, data_id: str) -> str:
    """Execute a retrieval query against the configured dataset."""

    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")

    url = _DATASET_URL_TEMPLATE.format(data_id=data_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        response = await client.post(url, headers=_build_headers(), json={"query": query})
        response.raise_for_status()
    payload = response.json()
    return _format_payload(payload)


@mcp.tool()
async def query_ux_knowledge(query: str) -> str:
    """Retrieve internal UX guidance, templates, or examples relevant to the query."""

    return await _query_dataset(query, _UX_DATASET_ID)


@mcp.tool()
async def query_lean_knowledge(query: str) -> str:
    """Retrieve Lean and continuous improvement methodology references."""

    return await _query_dataset(query, _LEAN_DATASET_ID)


@mcp.tool()
async def query_automation_step(query: str) -> str:
    """Retrieve automation process documentation and step-by-step guides."""

    return await _query_dataset(query, _AUTOMATION_DATASET_ID)


if __name__ == "__main__":
    mcp.run()
