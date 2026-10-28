# 第 4 周 L2：从零手写 ReAct 循环（不用 tools 参数，纯文本版）
# 运行：python week4/react1.py
#
# ReAct = Reason + Act（2022 年论文）。让模型反复做三件事：
#   Thought:     我现在该做什么（推理）
#   Action:      调用哪个工具、参数是什么（行动）
#   Observation: 工具返回的结果（由你的代码填进去，不是模型写的）
# 直到模型写出 Final Answer。
# 这里故意不用 L1 的 tools 参数，只靠 prompt + 字符串解析，你会看清 Agent 的本质就是一个 while 循环。

import os
import re

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"


def get_weather(city):
    fake = {"北京": "晴，12°C", "上海": "小雨，16°C", "深圳": "多云，24°C"}
    return fake.get(city, "未知城市")


def calculator(expression):
    if not set(expression) <= set("0123456789+-*/(). "):
        return "错误：只支持四则运算"
    return str(eval(expression))


TOOLS = {"get_weather": get_weather, "calculator": calculator}

SYSTEM = """你是一个会使用工具的助手。可用工具：
- get_weather[城市名]：查询城市今天的天气
- calculator[表达式]：计算四则运算，例如 calculator[(3+4)*2]

严格按下面的格式回答，每次只写一个 Action，然后停下等待 Observation：
Thought: 你的思考
Action: 工具名[参数]

拿到足够信息后，写：
Thought: 我已经知道答案了
Final Answer: 给用户的最终回答
"""


def react(question, max_steps=6):
    print("=" * 60)
    print("用户:", question)
    # 整个 ReAct 过程就是一段不断变长的文本
    transcript = f"Question: {question}\n"

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": transcript}],
            stop=["Observation:"],  # 关键：不让模型自己编造 Observation
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        print(f"\n--- 第 {step} 轮，模型输出 ---\n{text}")
        transcript += text + "\n"

        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        m = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
        if not m:
            observation = "格式错误：请按 Action: 工具名[参数] 的格式写"
        else:
            name, arg = m.group(1), m.group(2)
            if name in TOOLS:
                observation = TOOLS[name](arg)
            else:
                observation = f"没有叫 {name} 的工具"

        print(f"--- 你的代码执行，得到 Observation: {observation}")
        transcript += f"Observation: {observation}\n"

    return "超过最大步数，没能完成"


answer = react("北京和上海哪个更暖和？温差几度？")
print("\n最终答案:", answer)
