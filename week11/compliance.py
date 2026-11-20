"""
第 11 周 L4b:合规的三个"硬功夫" —— 租户隔离、脱敏、审计日志
运行:python week11/compliance.py        (不用联网、不用 API key)

场景:你的公司做"银行客服 Agent"SaaS,同一套系统服务两家银行(两个租户)。
⚠️ 第 1 部分的 naive 检索是故意有漏洞的。
"""
import re, json, time, hashlib

# ======================================================================
# 第 1 部分:多租户 RAG —— 一家银行的员工搜到了另一家银行的内部文档
# ======================================================================
DOCS = [  # 所有租户的文档放在同一个向量库里(很常见的省事做法)
    {"tenant": "bank_a", "text": "甲银行 VIP 客户理财产品年化利率 3.8%,内部审批额度上限 500 万。"},
    {"tenant": "bank_a", "text": "甲银行信用卡挂失请拨打客服热线,24 小时受理。"},
    {"tenant": "bank_b", "text": "乙银行 VIP 客户理财产品,具体利率请咨询客户经理。"},
    {"tenant": "bank_b", "text": "乙银行信用卡挂失可在手机银行 App 自助办理。"},
]

def score(q, t):
    """玩具相似度:两个字一组(bigram)的重合数,代替第 5 周的向量相似度。"""
    grams = lambda s: {s[i:i+2] for i in range(len(s) - 1)}
    return len(grams(q) & grams(t))

def search_naive(question, k=2):
    return sorted(DOCS, key=lambda d: -score(question, d["text"]))[:k]

def search_prompt_tenant(question, claimed_tenant, k=2):
    """租户来自请求里用户自己填的字段(或者让模型从对话里判断)—— 同样是漏洞。"""
    return [d for d in search_naive(question, k=10) if d["tenant"] == claimed_tenant][:k]

SESSIONS = {"token_xyz": {"user": "li", "tenant": "bank_b"}}   # 登录后服务器端保存的会话

def search_safe(question, token, k=2):
    tenant = SESSIONS[token]["tenant"]               # ① 租户只从服务器端的登录会话取
    pool = [d for d in DOCS if d["tenant"] == tenant] # ② 先过滤、再打分:别家文档根本不进候选
    return sorted(pool, key=lambda d: -score(question, d["text"]))[:k]

def part1():
    q = "VIP 客户理财利率是多少?"
    print("[第 1 部分] 乙银行员工 li 问:", q)
    print("  naive(不分租户):")
    for d in search_naive(q): print(f"    [{d['tenant']}] {d['text']}")
    print("  租户由用户自己声明(li 在请求里改成 bank_a):")
    for d in search_prompt_tenant(q, "bank_a"): print(f"    [{d['tenant']}] {d['text']}")
    print("  safe(租户取自登录会话 + 先过滤后检索):")
    for d in search_safe(q, "token_xyz"): print(f"    [{d['tenant']}] {d['text']}")
    print()

# ======================================================================
# 第 2 部分:脱敏后再调公网 API —— 什么出了域、什么没出
# ======================================================================
# (?<!\d) 和 (?!\d):前后不能紧挨着数字。不加的话,手机号正则会咬进身份证中间的 11 位,
# 身份证正则会咬掉 19 位卡号的前 18 位 —— 我第一版就是这么写错的。
PATTERNS = [
    ("IDCARD", r"(?<!\d)\d{17}[\dXx](?!\d)"),
    ("CARD",   r"(?<!\d)\d{16,19}(?!\d)"),
    ("PHONE",  r"(?<!\d)1[3-9]\d{9}(?!\d)"),
]

def mask(text):
    mapping, n = {}, {}
    for kind, pat in PATTERNS:
        def repl(m):
            n[kind] = n.get(kind, 0) + 1
            tag = f"<{kind}_{n[kind]}>"
            mapping[tag] = m.group()
            return tag
        text = re.sub(pat, repl, text)
    return text, mapping

def unmask(text, mapping):
    for tag, v in mapping.items(): text = text.replace(tag, v)
    return text

def fake_external_llm(prompt):          # 代表公网上的大模型 API
    return "已为您记录:卡号 <CARD_1> 的挂失申请,短信将发送至 <PHONE_1>。"

def part2():
    user = "我叫王小明,卡号 6222020200112233445 丢了,手机 13812345678,身份证 110101199003071234"
    masked, mapping = mask(user)
    print("[第 2 部分] 脱敏后再调公网 API")
    print("  用户原话:   ", user)
    print("  发出域的:   ", masked)
    print("  留在本地的映射:", mapping)
    reply = fake_external_llm(masked)
    print("  模型回答:   ", reply)
    print("  还原后给用户:", unmask(reply, mapping))
    print("  ⚠️ 注意:'王小明' 这个名字没被正则抓到,已经出域了\n")
    return masked

# ======================================================================
# 第 3 部分:审计日志 —— 出了事能回答"谁、什么时候、看了什么、做了什么"
# ======================================================================
def audit(user, tenant, question_masked, docs, tools, model, answer_masked):
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user, "tenant": tenant,
        "question": question_masked,                     # 存脱敏后的,日志本身也是敏感数据
        "retrieved": [hashlib.sha1(d["text"].encode()).hexdigest()[:8] for d in docs],  # 存文档 ID,不存原文
        "tools": tools, "model": model,
        "answer": answer_masked,
    }
    print("[第 3 部分] 一条审计日志")
    print("  " + json.dumps(rec, ensure_ascii=False, indent=2).replace("\n", "\n  "))

if __name__ == "__main__":
    part1()
    masked = part2()
    audit("li", "bank_b", masked, search_safe("信用卡挂失", "token_xyz"),
          [{"name": "freeze_card", "approved_by": "li", "result": "ok"}],
          "deepseek-chat@2026-11", fake_external_llm(masked))
