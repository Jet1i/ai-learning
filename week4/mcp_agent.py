# 第 4 周 L4（下）：让 L3 的 Agent 通过 MCP 使用工具
# 先装一次：pip install mcp
# 运行：python week4/mcp_agent.py
#
# 整个文件和 agent.py 只差两处：
#   1. 工具列表不再手写，而是启动时问 MCP Server 要（list_tools），再翻译成 OpenAI 的 tools 格式
#   2. 执行工具不再调本地函数，而是发给 MCP Server（call_tool）
# Agent 循环本身一个字没变。换一个 MCP Server（GitHub、数据库、浏览器……），Agent 就有了新能力。

import asyncio
import json
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai import OpenAI

MODEL = "deepseek-chat"
MAX_STEPS = 8
SERVER = StdioServerParameters(  # 告诉 Client：用当前 Python 启动 mcp_server.py 这个子进程
    command=sys.executable,
    args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
)


def to_openai_tool(t):
    # MCP 的工具描述 → OpenAI tools 格式。两边都用 JSON Schema，只是外面包装不同
    return {"type": "function", "function": {
        "name": t.name, "description": t.description or "", "parameters": t.input_schema}}


async def run_agent(llm, mcp, tools, messages):
    for step in range(1, MAX_STEPS + 1):
        resp = llm.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = await mcp.call_tool(call.function.name, args)  # ← 唯一变化：发给 MCP Server
            text = "\n".join(c.text for c in result.content if c.type == "text")
            print(f"  [第 {step} 步] MCP → {call.function.name}({args}) → {text[:60]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    return "（超过最大步数，停止）"


async def main():
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    async with Client(SERVER) as mcp:
        listed = await mcp.list_tools()  # 握手后第一件事：你有哪些工具？
        print("从 MCP Server 发现的工具：", [t.name for t in listed.tools])
        tools = [to_openai_tool(t) for t in listed.tools]

        messages = [{"role": "system", "content": "你是一个会用工具的助手，回答简洁，用中文。"}]
        while True:
            q = input("\n你: ").strip()
            if q in ("q", "quit", "exit"):
                break
            messages.append({"role": "user", "content": q})
            print("Agent:", await run_agent(llm, mcp, tools, messages))


if __name__ == "__main__":
    asyncio.run(main())
