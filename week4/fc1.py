# 第 4 周 L1：Function Calling —— 让模型"调用"你写的函数
# 运行：python week4/fc1.py
#
# 核心事实：模型自己什么都执行不了。它只会输出一段 JSON：
#   "我想调用 get_weather，参数是 {"city": "北京"}"
# 真正去执行函数的是你的 Python 代码，执行完再把结果发回给模型。

import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"


# ---------- 1. 你自己写的普通 Python 函数（模型看不到代码，只看到下面的说明书） ----------
def get_weather(city):
    fake = {"北京": "晴，12°C", "上海": "小雨，16°C", "深圳": "多云，24°C"}
    return fake.get(city, "未知城市")


def calculator(expression):
    # 只允许数字和 + - * / ( ) . 防止模型让你执行危险代码
    if not set(expression) <= set("0123456789+-*/(). "):
        return "错误：只支持四则运算"
    return str(eval(expression))


FUNCTIONS = {"get_weather": get_weather, "calculator": calculator}

# ---------- 2. 工具说明书（JSON Schema）：这就是模型唯一能看到的东西 ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市今天的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算四则运算表达式，例如 (3+4)*2",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def ask(question):
    print("=" * 60)
    print("用户:", question)
    messages = [{"role": "user", "content": question}]

    # ---------- 3. 第一次调用：把问题 + 说明书一起发过去 ----------
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message

    if not msg.tool_calls:  # 模型觉得不需要工具，直接回答了
        print("模型直接回答:", msg.content)
        return

    # ---------- 4. 模型返回的不是答案，而是"调用请求" ----------
    messages.append(msg)  # 把模型的调用请求也记进对话历史
    for call in msg.tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)  # arguments 是一段 JSON 字符串
        print(f"模型请求调用: {name}({args})")

        result = FUNCTIONS[name](**args)  # ← 真正执行的是你的代码
        print(f"  你的代码执行结果: {result}")

        # ---------- 5. 把结果用 role="tool" 发回去，tool_call_id 要对上 ----------
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # ---------- 6. 第二次调用：模型看到结果后，组织成人话 ----------
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    print("模型最终回答:", resp.choices[0].message.content)


ask("北京今天天气怎么样？")
ask("123 乘以 45 再加 678 等于多少？")
ask("你好，介绍一下你自己")
