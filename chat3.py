import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

SYSTEM = """你是售前需求分析助手。从客户描述中提取信息，只输出 JSON，格式如下：
"""

text = "我们是做跨境电商的，客服每天要回上千条英文咨询，人手不够、回复慢。老板想半年内上线智能客服，预算大概 20 万。"

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
    
    temperature=0,  # 0 表示最稳定，越高越有创意
)
data = json.loads(resp.choices[0].message.content)
print(data)
