import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

STRICT = """你是售前需求分析助手。从客户描述中提取信息，只输出 JSON，格式如下：
{"industry": "行业", "pain_points": ["痛点1", "痛点2"], "budget": "预算，没提到就填 null", "solution": "一句话推荐方案"}"""
LOOSE = "你是售前需求分析助手。从客户描述中提取行业、痛点、预算，并给出一句话推荐方案，用 JSON 输出。"

text = "我们是做跨境电商的，客服每天要回上千条英文咨询，人手不够、回复慢。老板想半年内上线智能客服，预算大概 20 万。"

def run(system, strict, n=10):
    fails = 0
    extra = {"response_format": {"type": "json_object"}} if strict else {}
    for i in range(n):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            temperature=1.5,
            **extra,
        )
        out = resp.choices[0].message.content
        try:
            data = json.loads(out)
            data["pain_points"]
            print(f"  第 {i+1} 次 ✅")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            fails += 1
            print(f"  第 {i+1} 次 ❌ {type(e).__name__}：{out[:60]!r}")
    return fails

print("宽松版：")
loose_fails = run(LOOSE, strict=False)
print("严格版：")
strict_fails = run(STRICT, strict=True)
print(f"\n失败次数：宽松版 {loose_fails}/10，严格版 {strict_fails}/10")