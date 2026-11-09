"""
第 5 周 L3（上）：知识库检索 MCP Server —— 把 L2 的 search() 包装成一个标准工具
不用手动运行：kb_agent.py 会把它当子进程启动；也可以接进 Claude Desktop（见课程消息）。
单独调试：mcp dev week5/kb_server.py

注意：stdio 模式下 stdout 是和 Agent 通信的"电话线"，这个文件里不能 print 到 stdout，
要打日志就写到 stderr（print(..., file=sys.stderr)）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 让下面能 import 同目录的 rag1
from mcp.server import MCPServer

import rag1  # 直接复用 L2 的切块、向量化、检索，一行都不重写

mcp = MCPServer("yunfan-kb")


@mcp.tool()
def search_kb(query: str, top_k: int = 3) -> str:
    """在云帆云帮助文档知识库里做语义检索，返回最相关的几段原文和出处。
    回答任何关于云帆云产品、计费、退款、账号、云服务器的问题之前都要先调用它。
    一个问题包含多个方面时，可以分别用不同的 query 调用多次。"""
    hits = rag1.search(query, k=top_k)
    good = [(s, c) for s, c in hits if s >= rag1.THRESHOLD]
    print(f"[kb_server] search_kb({query!r}) 最高分 {hits[0][0]:.3f}", file=sys.stderr)
    if not good:
        return f"没有找到相关内容（最高相似度 {hits[0][0]:.2f}，低于门槛 {rag1.THRESHOLD}）。"
    return "\n\n".join(f"[{c['source']}]（相似度 {s:.2f}）\n{c['text']}" for s, c in good)


@mcp.tool()
def list_docs() -> str:
    """列出知识库里有哪些文档和小节标题"""
    chunks, _ = rag1.get_index()
    return "\n".join(sorted({c["source"] for c in chunks}))


if __name__ == "__main__":
    mcp.run()
