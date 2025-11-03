"""
Model Context Protocol server exposing internal knowledge retrieval tools
— HTTP SSE 版本（Starlette 挂载 FastMCP.sse_app）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import httpx
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
import uvicorn

from mcp.server.fastmcp import FastMCP

# ===== 常量 / 配置 =====
_DATASET_TOKEN_ENV = "DIFY_DATASET_TOKEN"
_DEFAULT_DATASET_TOKEN = "dataset-gCRaKZgnKtvqLdeuoCFjKiME"
_DATASET_URL_TEMPLATE = "https://api.dify.ai/v1/datasets/{data_id}/retrieve"

_UX_DATASET_ID = "cab02597-6315-456c-92d3-19a65e3e7efd"
_LEAN_DATASET_ID = "67659dbe-4387-4122-8eb9-1d2005bea6a2"
_AUTOMATION_DATASET_ID = "b68de37f-a9f7-41fc-948f-eb89ca145770"

# ===== 初始化 FastMCP（名称保留原样）=====
mcp = FastMCP("internal-knowledge-retriever")


# ===== 工具函数 =====
def _build_headers() -> Dict[str, str]:
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
    for key in ("items", "data", "documents", "results"):
        items = payload.get(key)
        if isinstance(items, list):
            if not items:
                return "No matching documents were found."
            formatted = _format_documents(items)
            if formatted.strip():
                return formatted
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _query_dataset(query: str, data_id: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    url = _DATASET_URL_TEMPLATE.format(data_id=data_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        response = await client.post(url, headers=_build_headers(), json={"query": query})
        response.raise_for_status()
    payload = response.json()
    return _format_payload(payload)


# ===== MCP 工具 =====
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


# ===== Starlette + SSE 挂载（与第一段一致的思路）=====
def build_app() -> Starlette:
    app = Starlette(
        routes=[
            # mcp.sse_app() 会提供默认 /sse 与 /messages 两个端点
            Mount("/", app=mcp.sse_app()),
            Route("/healthz", lambda request: PlainTextResponse("ok"), methods=["GET"]),
        ]
    )
    # 可按需收紧 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
