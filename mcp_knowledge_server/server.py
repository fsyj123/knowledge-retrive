"""
Model Context Protocol server exposing internal knowledge retrieval tools
— HTTP SSE 版本（仅保留 records[*].segment.content）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ===== 常量 / 配置 =====
_DATASET_TOKEN_ENV = "DIFY_DATASET_TOKEN"
_DEFAULT_DATASET_TOKEN = "dataset-gCRaKZgnKtvqLdeuoCFjKiME"
_DATASET_URL_TEMPLATE = "https://api.dify.ai/v1/datasets/{data_id}/retrieve"

_UX_DATASET_ID = "cab02597-6315-456c-92d3-19a65e3e7efd"
_LEAN_DATASET_ID = "67659dbe-4387-4122-8eb9-1d2005bea6a2"
_AUTOMATION_DATASET_ID = "b68de37f-a9f7-41fc-948f-eb89ca145770"

# ===== 数据模型（结构化输出）=====
class KnowledgeSegment(BaseModel):
    """知识片段内容"""
    content: str = Field(..., description="知识片段的内容文本")


class KnowledgeRecord(BaseModel):
    """知识检索结果记录"""
    segment: KnowledgeSegment = Field(..., description="知识片段")


class KnowledgeRetrievalResult(BaseModel):
    """知识检索结果"""
    records: List[KnowledgeRecord] = Field(..., description="检索到的知识记录列表")


# ✅ 修复 ForwardRef 问题：让 Pydantic 重新解析嵌套类型
KnowledgeRetrievalResult.model_rebuild()

# ===== 初始化 FastMCP =====
mcp = FastMCP(
    name="internal-knowledge-retriever",
    instructions="Its useful for retriving internal knowledge from the knowledge base."
)


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


def _dataset_retrieve(query: str, data_id: str) -> Dict[str, Any]:
    """
    同步调用 Dify 检索接口，返回原始 payload（dict）。
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    url = _DATASET_URL_TEMPLATE.format(data_id=data_id)
    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        resp = client.post(url, headers=_build_headers(), json={"query": query})
        resp.raise_for_status()
        return resp.json()


def _shrink_records(payload: Dict[str, Any]) -> KnowledgeRetrievalResult:
    """
    只保留 records[*].segment.content（若无则 sign_content/text），其余字段不动。
    返回结构化的 Pydantic 模型。
    """
    records = payload.get("records", [])
    if not isinstance(records, list):
        return KnowledgeRetrievalResult(records=[])

    new_records: List[KnowledgeRecord] = []
    for rec in records:
        if not isinstance(rec, dict):
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

        # 创建结构化的记录（content 为空字符串时也保留，与原始行为一致）
        content_str = content if content is not None else ""
        new_records.append(
            KnowledgeRecord(segment=KnowledgeSegment(content=content_str))
        )

    return KnowledgeRetrievalResult(records=new_records)




# ===== MCP 工具（同步版本）=====
@mcp.tool()
def query_ux_knowledge(query: str) -> KnowledgeRetrievalResult:
    """
    Retrieve internal UX guidance, templates, or examples relevant to the query.
    
    This tool searches the UX knowledge base for design guidelines, templates,
    best practices, and examples that match the query. It returns structured
    knowledge segments containing relevant content.
    
    Args:
        query: The search query or question about UX-related topics.
             Examples: "How to design a login form?", "What are the best practices for mobile navigation?"
    
    Returns:
        KnowledgeRetrievalResult: A structured result containing a list of knowledge records.
                                Each record contains a segment with content text relevant to the query.
                                Only the segment content is returned to keep responses concise.
    
    Note:
        The response is optimized to only include segment content, avoiding excessively long responses.
    """
    raw = _dataset_retrieve(query, _UX_DATASET_ID)
    return _shrink_records(raw)


@mcp.tool()
def query_lean_knowledge(query: str) -> KnowledgeRetrievalResult:
    """
    Retrieve Lean and continuous improvement methodology references.
    
    This tool searches the Lean knowledge base for methodologies, frameworks,
    tools, and best practices related to continuous improvement, waste reduction,
    and process optimization.
    
    Args:
        query: The search query or question about Lean methodologies.
             Examples: "What is the 5S methodology?", "How to implement Kaizen?"
    
    Returns:
        KnowledgeRetrievalResult: A structured result containing a list of knowledge records.
                                Each record contains a segment with content text relevant to the query.
                                Only the segment content is returned to keep responses concise.
    
    Note:
        The response is optimized to only include segment content, avoiding excessively long responses.
    """
    raw = _dataset_retrieve(query, _LEAN_DATASET_ID)
    return _shrink_records(raw)


@mcp.tool()
def query_automation_step(query: str) -> KnowledgeRetrievalResult:
    """
    Retrieve automation process documentation and step-by-step guides.
    
    This tool searches the automation knowledge base for process documentation,
    step-by-step guides, automation workflows, and related technical documentation.
    
    Args:
        query: The search query or question about automation processes.
             Examples: "How to set up CI/CD pipeline?", "What are the steps for automated testing?"
    
    Returns:
        KnowledgeRetrievalResult: A structured result containing a list of knowledge records.
                                Each record contains a segment with content text relevant to the query.
                                Only the segment content is returned to keep responses concise.
    
    Note:
        The response is optimized to only include segment content, avoiding excessively long responses.
    """
    raw = _dataset_retrieve(query, _AUTOMATION_DATASET_ID)
    return _shrink_records(raw)


def main():
    mcp.run(transport="stdio")
    
if __name__ == "__main__":
    main()
