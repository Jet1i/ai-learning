# 第 4 周 L3：一个完整的 Agent（原生 tools + while 循环），约 100 行
# 运行：python week4/agent.py   然后直接打字提问，输入 q 退出
#
# 和 L2 的区别：不再靠 prompt 约定格式 + 正则解析，而是用 L1 学的 tools 参数（模型输出结构化 JSON，更稳）。
# 和 L1 的区别：把"执行工具 → 发回结果"放进 while 循环，模型想调几轮就调几轮。
# Agent = 模型 + 工具 + 循环 + 记忆（messages）。

import datetime
import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
MAX_STEPS = 8  # 防止死循环烧钱


# ---------- 工具：普通 Python 函数 ----------
def get_weather(city):
    fake = {"北京": "晴，12°C", "上海": "小雨，16°C", "深圳": "多云，24°C"}
    return fake.get(city, "未知城市")


def calculator(expression):
    if not set(expression) <= set("0123456789+-*/(). "):
        return "错误：只支持四则运算"
    return str(eval(expression))


def get_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M ") + "星期" + "一二三四五六日"[now.weekday()]


def read_note(filename):
    # 读 notes/ 里的学习笔记：Agent 第一次接触"你自己的数据"
    path = os.path.join(os.path.dirname(__file__), "..", "notes", os.path.basename(filename))
    if not os.path.exists(path):
        return "文件不存在。可用文件：" + ", ".join(os.listdir(os.path.dirname(path)))
    return open(path, encoding="utf-8").read()[:3000]  # 截断，别把上下文撑爆


FUNCTIONS = {"get_weather": get_weather, "calculator": calculator, "get_time": get_time, "read_note": read_note}


def tool(name, description, **params):
    # 小工具函数：少写点 JSON Schema 样板代码
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                       "required": list(params)}}}


TOOLS = [
    tool("get_weather", "查询某个城市今天的天气", city="城市名，如 北京"),
    tool("calculator", "计算四则运算表达式。涉及数字计算时必须用它，不要心算", expression="如 (16-12)*2"),
    tool("get_time", "获取当前日期、时间和星期"),
    tool("read_note", "读取用户的学习笔记文件，如 week03-transformer.md。不知道文件名时随便传一个，会返回文件列表",
         filename="文件名"),
]

SYSTEM = "你是 Jet 的学习助手，由 DeepSeek 模型驱动。需要事实或计算时优先用工具，不要编造。回答简洁，用中文。"


def run_agent(messages):
    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:  # 模型不再要工具 → 任务完成
            return msg.content

        for call in msg.tool_calls:  # 可能一次返回多个（并行调用）
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            try:
                result = FUNCTIONS[name](**args)
            except Exception as e:  # 工具报错也发回去，让模型自己想办法
                result = f"工具出错：{e}"
            print(f"  [第 {step} 步] {name}({args}) → {result[:60]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "（超过最大步数，停止）"


if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM}]  # 放在循环外：多轮对话共享记忆
    while True:
        q = input("\n你: ").strip()
        if q in ("q", "quit", "exit"):
            break
        messages.append({"role": "user", "content": q})
        print("Agent:", run_agent(messages))
