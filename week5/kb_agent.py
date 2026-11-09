"""
第 5 周 L3（下）：知识库问答 Agent = 第 4 周的 mcp_agent + 本周的 kb_server
运行：python week5/kb_agent.py

和 week4/mcp_agent.py 相比，只改了两处：
  1. SERVER 指向 kb_server.py
  2. system prompt 换成"客服 + 先检索再回答"的规矩
Agent 循环一个字没变 —— 这就是 MCP 的价值：换个 Server，Agent 就有了新能力。

和 L2 rag1.py 的区别（重点）：
  rag1：每个问题都先检索 1 次、固定取 3 块，再生成（流程写死在代码里）
  这里：由模型自己决定 要不要查、查几次、用什么关键词查（叫 Agentic RAG）
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
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_server.py")],
)
SYSTEM = (
    "你是云帆云的客服助手，回答简洁，用中文。"
    "凡是关于云帆云的问题，必须先用 search_kb 查知识库，只根据查到的内容回答，并注明出处（文档名 > 小节）。"
    "问题包含多个方面时，分别检索。知识库里查不到就说「知识库里没有相关信息」，不要编。"
    "闲聊和与云帆云无关的问题不用检索。"
)


def to_openai_tool(t):
    return {"type": "function", "function": {
        "name": t.name, "description": t.description or "", "parameters": t.input_schema}}


async def run_agent(llm, mcp, tools, messages):
    for step in range(1, MAX_STEPS + 1):
        resp = llm.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:  # 停止条件 1：模型不再调工具
            return msg.content
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = await mcp.call_tool(call.function.name, args)
            text = "\n".join(c.text for c in result.content if c.type == "text")
            print(f"  [第 {step} 步] {call.function.name}({args}) → {text[:50]!r}…")  # 工具日志：排查幻觉靠它
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "（超过最大步数，停止）"  # 停止条件 2：步数上限


async def main():
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    async with Client(SERVER) as mcp:
        listed = await mcp.list_tools()
        print("从 MCP Server 发现的工具：", [t.name for t in listed.tools])
        tools = [to_openai_tool(t) for t in listed.tools]
        messages = [{"role": "system", "content": SYSTEM}]
        while (q := input("\n你: ").strip()) not in ("q", "quit", "exit"):
            messages.append({"role": "user", "content": q})
            print("Agent:", await run_agent(llm, mcp, tools, messages))


if __name__ == "__main__":
    asyncio.run(main())
