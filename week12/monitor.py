"""
第 12 周 L3:上线以后怎么知道它"还好不好"——监控、评测、灰度

纯 Python,不联网。模拟银行客服 Agent 上线后 1 万条请求日志(数据是编的，规律是真实系统常见的)。

三步(每步先看"天真做法",再看为什么不够):
  1. 监控延迟：只看平均值 → 看不到最慢的那批用户      → 改看 P50/P95/P99
  2. 发新版本:全量上线 + 只看总体指标 → 某一类问题悄悄变差 → 灰度 + 分类别看
  3. 上线前：拿"黄金测试集"回归评测，挡住明显变差的版本

运行：python week12/monitor.py
"""
import random
random.seed(42)

# ---------- 造日志 ----------
CATS = ["查余额", "信用卡", "理财", "投诉"]
def make_log(version, n):
    logs = []
    for _ in range(n):
        cat = random.choices(CATS, weights=[40, 30, 20, 10])[0]
        in_tok = random.randint(1500, 3500)            # 系统提示 + RAG + 历史
        out_tok = random.randint(80, 300)
        # 首字延迟 TTFT(秒):大多数很快;5% 碰上长上下文 / 排队，很慢
        ttft = random.uniform(0.3, 0.9) if random.random() > 0.05 else random.uniform(4, 12)
        # 转人工概率：基础 8%;v2 把 prompt 改成"回答尽量简短",投诉类变差(这是要找的"事故")
        p_human = 0.08
        if version == "v2" and cat == "投诉":
            p_human = 0.30
        if version == "v2" and cat == "查余额":
            p_human = 0.03      # v2 让查余额变好了一点，正好把总体平均"掩盖"住
        to_human = random.random() < p_human
        thumbs_down = random.random() < (0.5 if to_human else 0.05)
        logs.append(dict(cat=cat, ttft=ttft, in_tok=in_tok, out_tok=out_tok,
                         to_human=to_human, down=thumbs_down, version=version))
    return logs

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p / 100))]

v1 = make_log("v1", 10000)

# ---------- 1. 延迟：平均值 vs 分位数 ----------
print("===== 1. 延迟监控 =====")
ttft = [r["ttft"] for r in v1]
print(f"天真做法：平均首字延迟 {sum(ttft)/len(ttft):.2f} 秒  → 看起来挺好")
print(f"分位数：P50 {pct(ttft,50):.2f}s   P95 {pct(ttft,95):.2f}s   P99 {pct(ttft,99):.2f}s")
slow = sum(t > 3 for t in ttft)
print(f"超过 3 秒的请求：{slow} 条 / 1 万 = {slow/100:.1f}%  → 每天 20 万请求就是 {slow*20:,} 个人在干等")

# ---------- 成本 ----------
PRICE_IN, PRICE_OUT = 2, 3          # 元/百万 token(第 11 周 DeepSeek 价格量级)
cost = sum(r["in_tok"] * PRICE_IN + r["out_tok"] * PRICE_OUT for r in v1) / 1e6
print(f"\n每次请求平均成本 {cost/len(v1)*100:.3f} 分;输入占 "
      f"{sum(r['in_tok']*PRICE_IN for r in v1)/1e6/cost:.0%}")

# ---------- 2. 发新版本 v2:全量 vs 灰度 + 分类别 ----------
def rate(logs, key, cat=None):
    rs = [r for r in logs if cat is None or r["cat"] == cat]
    return sum(r[key] for r in rs) / len(rs)

print("\n===== 2. 上线新版本 v2(改了系统提示)=====")
v2 = make_log("v2", 10000)
print(f"天真做法：只看总体转人工率   v1 {rate(v1,'to_human'):.1%}   v2 {rate(v2,'to_human'):.1%}  → 差不多，放心全量?")
print("\n按类别拆开看:")
print(f"  {'类别':6s}{'v1 转人工':>10s}{'v2 转人工':>10s}{'v2 差评':>9s}")
for c in CATS:
    flag = "  ⚠️" if rate(v2, "to_human", c) > rate(v1, "to_human", c) * 1.5 else ""
    print(f"  {c:6s}{rate(v1,'to_human',c):10.1%}{rate(v2,'to_human',c):10.1%}{rate(v2,'down',c):9.1%}{flag}")

# 灰度:5% 流量给 v2,对比同一时段的 v1
print("\n灰度发布:先给 5% 流量(每天 20 万 → 1 万条)跑 v2,和同时段 v1 对比")
canary = make_log("v2", 2000)     # 5% 流量跑约 5 小时的量
base = make_log("v1", 2000)
for c in ["投诉"]:
    a, b = rate(base, "to_human", c), rate(canary, "to_human", c)
    print(f"  投诉 转人工:v1 {a:.1%} vs 灰度 v2 {b:.1%}")
    if b > a * 1.5:
        print("  → 触发回滚规则(某类别转人工率比旧版高 50% 以上):自动切回 v1,只影响了 5% 的用户")

# ---------- 3. 上线前的回归评测 ----------
print("\n===== 3. 上线前：黄金测试集回归 =====")
GOLDEN = [   # (用户问题, 类别, 应该怎么处理)  真实项目里几百条，来自线上真实问题 + 坏案例
    ("我卡丢了帮我冻结", "信用卡", "调用 freeze_card"),
    ("你们乱扣我年费，我要投诉", "投诉", "登记投诉 + 说明退费流程"),
    ("说了三遍都没解决，叫你们经理来", "投诉", "安抚 + 登记 + 承诺回电时限"),
    ("余额多少", "查余额", "调用 query_balance"),
    ("推荐个低风险理财", "理财", "只推荐 R1/R2"),
    ("你们服务太差了我要投诉", "投诉", "登记投诉 + 安抚"),
]
# 模拟两个版本在测试集上的通过情况(真实里由人工或 LLM 评委打分，见第 5 周 eval.py)
passed = {"v1": [1, 1, 1, 1, 1, 1], "v2": [1, 0, 0, 1, 1, 1]}
for v in passed:
    ok = sum(passed[v]); print(f"  {v}: {ok}/{len(GOLDEN)} 通过")
for (q, c, exp), a, b in zip(GOLDEN, passed["v1"], passed["v2"]):
    if a and not b:
        print(f"  ❌ v2 退步：[{c}] {q}  (期望：{exp})")
print("  → 上线门槛：不能有原来通过、现在失败的用例(或者每类通过率不低于旧版)")
print("\n小结：上线前用测试集挡一次，上线时灰度挡一次，上线后分位数 + 分类别监控兜底。")
