"""
项目 1 · 学习笔记 MCP Server
  python week5/project1/server.py            stdio 模式（给 agent.py / Claude Desktop 用）
  python week5/project1/server.py --http     HTTP 模式，地址 http://localhost:8000/mcp（可远程、多客户端）
提醒：stdio 模式下不能 print 到 stdout，日志一律写 stderr。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp.server import MCPServer

import kb

mcp = MCPServer("jet-notes")


@mcp.tool()
def search_notes(query: str, top_k: int = 4) -> str:
    """在 Jet 的大模型/AI Agent 学习笔记（每周笔记 + 问答记录）里检索，返回最相关的片段和出处。
    回答任何关于 Jet 学过的内容（API、反向传播、Transformer、Agent、MCP、RAG 等）的问题前都要先调用。
    query 请写成一句完整的问题，保留用户的意图；英文术语保持原样（如 tool_call_id、GPT-2）。
    问题有多个方面时，可以分别调用多次。"""
    idx = kb.get_index()
    best = idx.best_cos(query)
    print(f"[server] search_notes({query!r}) best_cos={best:.3f}", file=sys.stderr)
    if best < kb.THRESHOLD:
        return f"笔记里没有找到相关内容（最高相似度 {best:.2f}，低于门槛 {kb.THRESHOLD}）。"
    hits = idx.search(query, k=top_k)
    return "\n\n".join(f"【{i + 1}】出处：{c['source']}\n{c['text']}" for i, (c, s) in enumerate(hits))


@mcp.tool()
def read_section(source: str) -> str:
    """读取某个出处对应的整个小节全文。search_notes 返回的片段不完整、需要上下文时使用，
    source 填 search_notes 结果里的"出处"原文。"""
    sec = kb.get_index().sections
    if source in sec:
        return sec[source]
    near = [s for s in sec if source in s or s in source]
    return sec[near[0]] if near else "没有这个出处。可以先用 search_notes 检索。"


@mcp.tool()
def list_notes() -> str:
    """列出知识库里有哪些笔记文件和小节标题"""
    return "\n".join(kb.get_index().sections.keys())


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", port=8000)
    else:
        mcp.run()
