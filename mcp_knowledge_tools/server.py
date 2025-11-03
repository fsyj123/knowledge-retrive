"""
Model Context Protocol server exposing internal knowledge retrieval tools
— HTTP SSE 版本（仅保留 records[*].segment.content）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

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

# ===== 初始化 FastMCP =====
mcp = FastMCP("internal-knowledge-retriever")


# ===== HTTP 工具函数 =====
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


async def _dataset_retrieve(query: str, data_id: str) -> Dict[str, Any]:
    """
    调用 Dify 检索接口，返回原始 payload（dict）。
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    url = _DATASET_URL_TEMPLATE.format(data_id=data_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.post(url, headers=_build_headers(), json={"query": query})
        resp.raise_for_status()
    return resp.json()


def _shrink_records(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留 records[*].segment.content（若无则 sign_content/text），其余字段不动。
    """
    out = dict(payload)  # 浅拷贝顶层
    records = payload.get("records")
    if not isinstance(records, list):
        return out

    new_records: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            new_records.append(rec)
            continue

        seg = rec.get("segment")
        content: Optional[str] = None
        if isinstance(seg, dict):
            # 优先 content，其次 sign_content，最后 text（兼容不同返回）
            content = (
                seg.get("content")
                or seg.get("sign_content")
                or seg.get("text")
            )

        # 复制 record，但 segment 替换为仅含 content 的对象
        rec_copy = dict(rec)
        rec_copy["segment"] = {"content": content} if content is not None else {"content": ""}
        new_records.append(rec_copy)

    out["records"] = new_records
    return out


# ===== MCP 工具 =====
@mcp.tool()
async def query_ux_knowledge(query: str) -> Dict[str, Any]:
    """
    Retrieve internal UX guidance, templates, or examples relevant to the query.
    仅返回 records[*].segment.content，避免响应过长。
    """
    raw = await _dataset_retrieve(query, _UX_DATASET_ID)
    return _shrink_records(raw)


@mcp.tool()
async def query_lean_knowledge(query: str) -> Dict[str, Any]:
    """
    Retrieve Lean and continuous improvement methodology references.
    仅返回 records[*].segment.content，避免响应过长。
    """
    raw = await _dataset_retrieve(query, _LEAN_DATASET_ID)
    return _shrink_records(raw)


@mcp.tool()
async def query_automation_step(query: str) -> Dict[str, Any]:
    """
    Retrieve automation process documentation and step-by-step guides.
    仅返回 records[*].segment.content，避免响应过长。
    """
    raw = await _dataset_retrieve(query, _AUTOMATION_DATASET_ID)
    return _shrink_records(raw)


# ===== Starlette + SSE 挂载 =====
def build_app() -> Starlette:
    app = Starlette(
        routes=[
            # mcp.sse_app() 默认提供 /sse 与 /messages 两个端点
            Mount("/", app=mcp.sse_app()),
            Route("/healthz", lambda request: PlainTextResponse("ok"), methods=["GET"]),
        ]
    )
    # 按需调整 CORS
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
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
