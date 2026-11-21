"""
第 12 周 L2:多 Agent —— 拆成几个 Agent 什么时候更好，什么时候反而更差

场景：银行客服。用户一句话里有两件事：冻结丢失的卡 + 推荐低风险理财。
三种写法对比：
  single   一个 Agent,4 个工具全给它(第 4 周的写法)
  router   路由 Agent 先分类 → 交给一个专员 Agent(专员只有自己那几个工具)
           ⚠️ 故意有两个坑：只能选一个类别;交接时只传一句"摘要"
  orch     编排者(orchestrator)拆成多个子任务 → 分别派给专员 → 汇总
           交接时传原话 + 结构化字段

运行：python week12/multi_agent.py            (三种都跑)
     python week12/multi_agent.py router     (只跑一种)
     MODEL=fake python week12/multi_agent.py (不联网，用规则模拟模型，只看流程)
所有工具都是假的，只打印。
"""
import os, sys, json, re
from openai import OpenAI

MODEL = os.environ.get("MODEL", "deepseek-chat")
if MODEL != "fake":
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

USER = "我尾号 8821 的信用卡昨天丢了，赶紧帮我冻结。另外我想把 5 万块放个低风险的理财，有推荐吗？"

# ---------- 假工具 ----------
LOG = []        # 记录真正执行了哪些动作
def freeze_card(tail):   LOG.append(f"冻结卡 {tail}"); return f"尾号 {tail} 的卡已冻结"
def query_card(tail):    return f"尾号 {tail}:信用卡，状态正常，额度 3 万"
def product_info(risk):  LOG.append(f"查理财 {risk}"); return "稳盈 30 天:R1 低风险，年化约 2.1%,1 万起" if risk.startswith("低") else "进取混合:R4 中高风险"
def file_complaint(text):LOG.append("登记投诉"); return "投诉已登记，编号 C1024"

def T(name, desc, prop):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {prop: {"type": "string"}}, "required": [prop]}}}
TOOLS = {
    "freeze_card":    (freeze_card,    T("freeze_card", "冻结银行卡，参数为卡号后四位", "tail")),
    "query_card":     (query_card,     T("query_card", "查询银行卡状态，参数为卡号后四位", "tail")),
    "product_info":   (product_info,   T("product_info", "按风险等级查理财产品，参数如 低风险/高风险", "risk")),
    "file_complaint": (file_complaint, T("file_complaint", "登记客户投诉", "text")),
}
SPECIALISTS = {   # 每个专员只拿自己的工具(权限隔离)
    "卡务": ["freeze_card", "query_card"],
    "理财": ["product_info"],
    "投诉": ["file_complaint"],
}

STATS = {"calls": 0, "tokens": 0}

# ---------- 调模型(fake 模式用规则模拟)----------
def llm(messages, tools=None):
    STATS["calls"] += 1
    if MODEL == "fake":
        return fake_llm(messages, tools)
    r = client.chat.completions.create(model=MODEL, messages=messages, tools=tools or None)
    STATS["tokens"] += r.usage.total_tokens
    return r.choices[0].message

def agent(system, user_text, tool_names, max_steps=5):
    """第 4 周的 while 循环 Agent,只是工具可以限定"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    tools = [TOOLS[n][1] for n in tool_names]
    for _ in range(max_steps):
        m = llm(msgs, tools)
        if not getattr(m, "tool_calls", None):
            return m.content
        msgs.append(m if MODEL != "fake" else {"role": "assistant", "content": None, "tool_calls": m.tool_calls_raw})
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            out = TOOLS[tc.function.name][0](**args)
            print(f"      🔧 {tc.function.name}({args}) → {out}")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "(步数用完)"

# ---------- 写法 1:单 Agent ----------
def run_single():
    return agent("你是银行客服，可以使用工具。", USER, list(TOOLS))

# ---------- 写法 2:路由 + 专员(故意有坑)----------
def run_router():
    m = llm([{"role": "system", "content":
              "你是路由。把用户问题分到一个类别：卡务/理财/投诉。"
              '只输出 JSON:{"category": "...", "summary": "一句话概括用户需求"}'},
             {"role": "user", "content": USER}])
    route = json.loads(re.search(r"\{.*\}", m.content, re.S).group())
    print(f"    路由结果:{route}")
    # 坑 1:只选一个类别;坑 2:专员只看到 summary,看不到原话
    return agent(f"你是银行{route['category']}专员，可以使用工具。", route["summary"],
                 SPECIALISTS[route["category"]])

# ---------- 写法 3:编排者 + 专员 + 汇总 ----------
def run_orch():
    m = llm([{"role": "system", "content":
              "你是编排者。把用户问题拆成子任务，每个子任务分到 卡务/理财/投诉 之一，"
              "并抽出关键字段(卡号尾号、金额、风险偏好等,原文里有就照抄)。"
              '只输出 JSON:{"tasks": [{"category": "...", "goal": "...", "fields": {...}}]}'},
             {"role": "user", "content": USER}])
    plan = json.loads(re.search(r"\{.*\}", m.content, re.S).group())
    print(f"    拆分结果:{json.dumps(plan, ensure_ascii=False)}")
    results = []
    for t in plan["tasks"]:
        handoff = f"用户原话：{USER}\n你只负责这一件事:{t['goal']}\n已知字段:{json.dumps(t['fields'], ensure_ascii=False)}"
        print(f"    → 派给【{t['category']}】专员")
        results.append(f"[{t['category']}] " + agent(f"你是银行{t['category']}专员，只处理分给你的事。",
                                                     handoff, SPECIALISTS[t["category"]]))
    m = llm([{"role": "system", "content": "把各专员的结果合成一段给客户的回复，简洁。"},
             {"role": "user", "content": "\n".join(results)}])
    return m.content

# ---------- fake 模型:用规则假装是 LLM,只为不联网也能看流程 ----------
class _F:   # 模拟 openai 返回的对象
    def __init__(self, content=None, calls=None):
        self.content = content
        self.tool_calls = [type("TC", (), {"id": f"c{i}", "function": type("Fn", (), {"name": n, "arguments": json.dumps(a, ensure_ascii=False)})})
                           for i, (n, a) in enumerate(calls or [])] or None
        self.tool_calls_raw = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                               for i, (n, a) in enumerate(calls or [])]
def fake_llm(msgs, tools):
    sys_, last = msgs[0]["content"], msgs[-1]
    names = [t["function"]["name"] for t in (tools or [])]
    if sys_.startswith("你是路由"):
        return _F('{"category": "卡务", "summary": "用户信用卡丢失，需要冻结"}')   # 只记住第一件事，也丢了尾号
    if sys_.startswith("你是编排者"):
        return _F(json.dumps({"tasks": [
            {"category": "卡务", "goal": "冻结丢失的信用卡", "fields": {"尾号": "8821"}},
            {"category": "理财", "goal": "推荐低风险理财", "fields": {"金额": "5万", "风险": "低风险"}}]}, ensure_ascii=False))
    if sys_.startswith("把各专员"):
        return _F("已为您冻结尾号 8821 的卡;低风险理财推荐稳盈 30 天(R1,年化约 2.1%)。")
    if last["role"] == "tool":
        return _F("已处理完毕。")
    text = last["content"]
    calls, tail = [], re.search(r"\d{4}", text)
    if "freeze_card" in names and "冻结" in text:
        if not tail: return _F("请问您的卡号后四位是多少？")
        calls.append(("freeze_card", {"tail": tail.group()}))
    if "product_info" in names and "理财" in text:
        calls.append(("product_info", {"risk": "低风险"}))
    return _F(None, calls) if calls else _F("好的。")

# ---------- 跑 ----------
modes = sys.argv[1:] or ["single", "router", "orch"]
for mode in modes:
    LOG.clear(); STATS.update(calls=0, tokens=0)
    print(f"\n===== {mode} =====")
    answer = {"single": run_single, "router": run_router, "orch": run_orch}[mode]()
    print(f"    最终回复:{answer}")
    print(f"    实际执行的动作:{LOG or '无'}")
    print(f"    调用模型 {STATS['calls']} 次" + (f",共 {STATS['tokens']} token" if STATS['tokens'] else ""))
