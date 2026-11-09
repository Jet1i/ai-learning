"""
项目 1 · 学习助手 Agent（通过 MCP 使用 server.py 的工具）
  python week5/project1/agent.py           自动以子进程方式启动 server.py（stdio）
  python week5/project1/agent.py --http    连接已经在运行的 http://localhost:8000/mcp
"""
import asyncio
import json
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai import OpenAI

MODEL = "deepseek-chat"
MAX_STEPS = 6
SERVER = ("http://localhost:8000/mcp" if "--http" in sys.argv else
          StdioServerParameters(command=sys.executable,
                                args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")]))
SYSTEM = (
    "你是 Jet 的学习助手。Jet 在系统学习大模型和 AI Agent，笔记都在知识库里。"
    "回答和学习内容有关的问题时，必须先用 search_notes 查笔记，以笔记为准，并在末尾列出出处。"
    "片段不完整时用 read_section 读整节。笔记里没有就明确说「笔记里没有」，然后可以给出你自己的简短解释，但要标明「（笔记外补充）」。"
    "闲聊不用检索。用中文，简洁。"
)


def to_openai_tool(t):
    return {"type": "function", "function": {
        "name": t.name, "description": t.description or "", "parameters": t.input_schema}}


async def run_agent(llm, mcp, tools, messages):
    for step in range(1, MAX_STEPS + 1):
        msg = llm.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                          temperature=0).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = await mcp.call_tool(call.function.name, args)
            text = "\n".join(c.text for c in result.content if c.type == "text")
            print(f"  [第 {step} 步] {call.function.name}({args}) → {len(text)} 字")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "（超过最大步数，停止）"


async def main():
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    async with Client(SERVER) as mcp:
        tools = [to_openai_tool(t) for t in (await mcp.list_tools()).tools]
        print("工具：", [t["function"]["name"] for t in tools], "（q 退出）")
        messages = [{"role": "system", "content": SYSTEM}]
        while (q := input("\n你: ").strip()) not in ("q", "quit", "exit"):
            messages.append({"role": "user", "content": q})
            print("助手:", await run_agent(llm, mcp, tools, messages))


if __name__ == "__main__":
    asyncio.run(main())
