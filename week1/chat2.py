import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
messages = [{"role": "system", "content": "你是一个耐心的 AI 老师"}]

while True:
    user = input("你：")
    if user == "exit":
        break
    messages.append({"role": "user", "content": user})
    messages = [messages[0]] + messages[1:][-10:]
    resp = client.chat.completions.create(model="deepseek-chat", messages=messages)
    reply = resp.choices[0].message.content
    print("AI：", reply)
    
    print(f"[本轮输入 {resp.usage.prompt_tokens} tokens]")