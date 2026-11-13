"""第 9 周 L2.5:PPO —— rlhf.py 的"正式版"多了什么
运行:python week9/ppo.py

PPO = Proximal Policy Optimization(近端策略优化)
  Policy  = LLM(决定说什么)
  Proximal = 每次只挪一小步
比 rlhf.py 多两样东西:① clip 限速  ② 价值模型给每个字分功劳
"""
import numpy as np

ANSWERS = ["简洁准确", "详细准确", "敷衍一句", "胡编政策", "长篇废话"]
RM_SCORE = np.array([6.0, 6.5, 1.0, 0.0, 9.1])
TRUE_GOOD = np.array([9, 9, 3, 0, 2])
REF_P = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
BETA = 3.0

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

# =====================================================================
print("=" * 60)
print("第一部分:为什么要 clip(限速)")
print("=" * 60)
print("""
LLM 生成回答是整个训练里最慢的一步。所以 PPO 的做法是:
  先抽一批回答(比如 16 个) → 拿这同一批数据反复更新好几轮(比如 10 轮)
省了生成,但有风险:一批里碰巧某个回答拿了高分,反复推 10 轮,它的概率会被推得太猛。
""")

# ---- 最小例子:一个回答,算一次 clip ----
print("--- 最小例子 ---")
p_old, A = 0.25, +2.0
for p_new in [0.25, 0.28, 0.30, 0.40]:
    ratio = p_new / p_old
    clipped = min(max(ratio, 0.8), 1.2)
    obj = min(ratio * A, clipped * A)
    push = "继续推" if ratio * A <= clipped * A else "到顶了,不再推(梯度 = 0)"
    print(f"  旧概率 {p_old}  新概率 {p_new:.2f}  ratio {ratio:.2f}  "
          f"clip 后 {clipped:.2f}  目标 {obj:.2f}  → {push}")
print("  比上一版涨到 1.2 倍以后,这一批数据就不再推它了。A < 0 时同理,降到 0.8 倍就停。\n")

# ---- 实验:同一批数据反复用 10 轮,有没有 clip ----
def run(use_clip, seed, batches=150, batch=16, epochs=10, lr=0.2):
    rng = np.random.default_rng(seed)
    logits = np.log(REF_P).copy()
    jumps = []
    for _ in range(batches):
        p_old = softmax(logits)
        idx = rng.choice(5, size=batch, p=p_old)                     # 1. 抽一批(贵)
        r = RM_SCORE[idx] - BETA * np.log(p_old[idx] / REF_P[idx])   # 2. 打分 + KL 绳子
        adv = r - r.mean()                                           # 3. 和这批的平均比
        for _ in range(epochs):                                      # 4. 同一批反复用
            p = softmax(logits)
            grad = np.zeros(5)
            for i, a in zip(idx, adv):
                ratio = p[i] / p_old[i]
                if use_clip and ((a > 0 and ratio > 1.2) or (a < 0 and ratio < 0.8)):
                    continue                                         # 到顶了,这条不推
                g = -p.copy(); g[i] += 1
                grad += a * ratio * g
            logits += lr * grad / batch
        p_new = softmax(logits)
        jumps.append(np.max(np.maximum(p_new / p_old, p_old / p_new)))
    p = softmax(logits)
    return p, np.median(jumps), np.max(jumps)

for use_clip in [False, True]:
    name = "PPO(有 clip)" if use_clip else "不限速"
    sats, mean_j, max_j = [], [], []
    for seed in range(8):
        p, mj, xj = run(use_clip, seed)
        sats.append(p @ TRUE_GOOD); mean_j.append(mj); max_j.append(xj)
    xj = np.max(max_j)
    xj_s = f"{xj:.1f} 倍" if xj < 100 else "上百倍(数值都爆了)"
    crashed = sum(s_ < 5 for s_ in sats)
    print(f"{name:10s} 一批之后概率一般变 {np.median(mean_j):.2f} 倍,最猛一次变 {xj_s}")
    print(f"{'':10s} 8 次训练的满意度 {np.round(sats, 2)} → 崩了 {crashed} 次")
print("不限速:一批运气好的数据被反复用 10 轮,概率被推爆,有的训练直接崩成只说废话(满意度 2)。")
print("clip:每批最多改约 1.2 倍,结果稳定。\n")

# =====================================================================
print("=" * 60)
print("第二部分:价值模型 —— 给每个字分功劳")
print("=" * 60)
print("""
真实回答是一个字一个字生成的,裁判只在最后给一个总分。那这个分该算谁的功劳?
最小例子:回答只有 2 步。
  第 1 步 开场:  直接 / 客套(各 50%)
  第 2 步 内容:  准确 / 胡编(直接开场后 80% 准确;客套开场后容易跑题,只有 40% 准确)
""")
FINAL = {("直接", "准确"): 7, ("直接", "胡编"): 0, ("客套", "准确"): 6, ("客套", "胡编"): -1}
P_OPEN = {"直接": 0.5, "客套": 0.5}
P_OK = {"直接": 0.8, "客套": 0.4}

# 价值 V = "从这里接着写下去,平均能拿多少分"
V_after = {o: P_OK[o] * FINAL[(o, "准确")] + (1 - P_OK[o]) * FINAL[(o, "胡编")] for o in P_OPEN}
V_start = sum(P_OPEN[o] * V_after[o] for o in P_OPEN)
print(f"V(开始)        = {V_start:.1f}")
for o in P_OPEN:
    print(f"V(开场={o})   = {P_OK[o]}×{FINAL[(o,'准确')]} + {1-P_OK[o]:.1f}×{FINAL[(o,'胡编')]} = {V_after[o]:.1f}")

o, c = "客套", "准确"
R = FINAL[(o, c)]
print(f"\n这次抽到:{o} + {c},最终 {R} 分")
print(f"rlhf.py 的做法(只看总分):两步都拿 {R} − {V_start:.1f} = {R - V_start:+.1f} → '客套'也被表扬了")
print(f"PPO 价值模型的做法(看每一步让期望变了多少):")
print(f"  第 1 步 '{o}':V 从 {V_start:.1f} 变成 {V_after[o]:.1f} → 功劳 {V_after[o] - V_start:+.1f}(该批评)")
print(f"  第 2 步 '{c}':V 从 {V_after[o]:.1f} 变成 最终 {R} → 功劳 {R - V_after[o]:+.1f}(该表扬)")
print("真实的价值模型是另一个和 LLM 一样大的神经网络,训练它去预测这个 V。")
print("这就是 PPO 显存里要放 4 个模型的原因:LLM、SFT 参照、裁判、价值模型。")
