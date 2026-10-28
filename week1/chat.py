import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个耐心的 AI 老师"},
        {"role": "user", "content": "用一句话解释什么是大模型"},
    ],
)
print(resp.choices[0].message.content)
print(resp.usage)  # 这次调用消耗了多少 token