"""
第 11 周 L1:API 账本 —— 一次请求、一个月到底花多少钱
运行:python week11/tco_api.py
  第 1 部分会真调 DeepSeek(需要 DEEPSEEK_API_KEY);没网/没 key 会自动跳过,后面照样跑。

价格是示例(2026 年写课时的公开价,单位:元 / 百万 token)。真做方案时一定以官网当天价格为准。
"""
import os, time

# ---------- 价格表:输入(缓存命中)、输入(未命中)、输出 ----------
PRICES = {
    #                      命中    未命中   输出     (元 / 百万 token)
    "DeepSeek-V3.2":     (0.2,   2.0,    3.0),
    "某旗舰闭源模型(示例)": (2.2,   22.0,   108.0),   # 约等于 $3 / $15 一档
}

def cost(model, hit, miss, out):
    """一次请求的钱(元)。hit/miss/out 都是 token 数。"""
    p_hit, p_miss, p_out = PRICES[model]
    return (hit * p_hit + miss * p_miss + out * p_out) / 1e6

# ======================================================================
# 第 1 部分:真调两次,看 usage 和缓存
# ======================================================================
SYSTEM = ("你是 ShopX 电商客服。规则如下:\n" +
          "".join(f"{i}. 第 {i} 条服务规范:回答要礼貌、简洁,涉及退款先确认订单号,不承诺系统外的赔偿。\n"
                  for i in range(1, 61)))          # 一段较长、每次都一样的系统提示(约 1500+ token)

def part1():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("[第 1 部分] 没有 DEEPSEEK_API_KEY,跳过真调用\n"); return
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    for i, q in enumerate(["我想退款,怎么操作?", "优惠券过期了还能用吗?"], 1):
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q}],
            max_tokens=150)
        u = r.usage
        hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        miss = getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens - hit)
        c = cost("DeepSeek-V3.2", hit, miss, u.completion_tokens)
        print(f"[第 {i} 次] 输入 {u.prompt_tokens}(命中 {hit} / 未命中 {miss}),"
              f"输出 {u.completion_tokens} → {c*100:.4f} 分钱")
        time.sleep(3)   # 给缓存一点时间落盘
    print()

# ======================================================================
# 第 2 部分:一个月的账单(客服场景)
# ======================================================================
SCENARIO = dict(
    requests_per_day = 1_000_000,  # 每天 100 万次请求
    system_tokens    = 1500,       # 系统提示(每次一样 → 可以命中缓存)
    rag_tokens       = 2000,       # RAG 检索出来的段落(每次不一样)
    user_tokens      = 100,        # 用户这句话
    output_tokens    = 300,        # 模型回答
    cache_hit_rate   = 0.0,        # 系统提示命中缓存的比例,先当 0
)

def monthly(model, s):
    hit  = s["system_tokens"] * s["cache_hit_rate"]
    miss = s["system_tokens"] * (1 - s["cache_hit_rate"]) + s["rag_tokens"] + s["user_tokens"]
    out  = s["output_tokens"]
    p_hit, p_miss, p_out = PRICES[model]
    n = s["requests_per_day"] * 30
    parts = {"输入-命中": hit * p_hit * n / 1e6,
             "输入-未命中": miss * p_miss * n / 1e6,
             "输出": out * p_out * n / 1e6}
    return parts

def part2():
    print("[第 2 部分] 客服场景月账单(每天 100 万次)")
    for rate in (0.0, 0.9):
        SCENARIO["cache_hit_rate"] = rate
        print(f"  系统提示缓存命中率 {rate:.0%}:")
        for m in PRICES:
            parts = monthly(m, SCENARIO)
            total = sum(parts.values())
            detail = ",".join(f"{k} {v/1e4:.1f} 万" for k, v in parts.items())
            print(f"    {m:<14} 月费 {total/1e4:8.1f} 万元  ({detail})")
    SCENARIO["cache_hit_rate"] = 0.0
    print()

# ======================================================================
# 第 3 部分:多轮对话 —— 天真估算(故意有错)vs 真实账单
# ======================================================================
def part3(turns=10, sys_t=1500, user_t=100, out_t=300, model="DeepSeek-V3.2"):
    print(f"[第 3 部分] 一段 {turns} 轮对话的钱({model},不算缓存)")
    # 天真估算:每轮只算"这一轮的系统提示 + 用户这句 + 回答"
    naive = sum(cost(model, 0, sys_t + user_t, out_t) for _ in range(turns))
    # 真实:API 是无状态的(第 1 周学过),第 k 轮要把前面所有轮的问答重新发一遍
    real, history = 0.0, 0
    for k in range(1, turns + 1):
        inp = sys_t + history + user_t
        real += cost(model, 0, inp, out_t)
        if k in (1, 5, 10): print(f"  第 {k:>2} 轮输入 {inp:>6} token")
        history += user_t + out_t
    print(f"  天真估算 {naive*100:.2f} 分钱   真实 {real*100:.2f} 分钱   差 {real/naive:.1f} 倍\n")

if __name__ == "__main__":
    part1()
    part2()
    part3()
