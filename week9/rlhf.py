"""第 9 周 L2:RLHF —— 拿奖励模型的分数推着 LLM 改
运行:python week9/rlhf.py

最小版本:一个问题"怎么退款?",LLM 只会从 5 个候选回答里选一个。
LLM = 5 个 logits(就是最后那层 wx+b 的输出)→ softmax → 按概率抽一个回答。
"""
import numpy as np

ANSWERS = ["简洁准确", "详细准确", "敷衍一句", "胡编政策", "长篇客套废话"]
RM_SCORE = np.array([6.0, 6.5, 1.0, 0.0, 9.1])   # L1 那个"偏爱长度"的奖励模型打的分
TRUE_GOOD = np.array([9, 9, 3, 0, 2])            # 人真正觉得的好坏(训练时看不到)
REF_P = np.array([0.30, 0.25, 0.20, 0.15, 0.10]) # SFT 之后的模型(参考模型)

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def show(name, p):
    bar = "  ".join(f"{a}{x:5.1%}" for a, x in zip(ANSWERS, p))
    print(f"{name:14s} {bar}")
    print(f"{'':14s} 奖励模型平均分 {p @ RM_SCORE:.2f}   人真正的满意度 {p @ TRUE_GOOD:.2f}")

def train(beta, steps=3000, lr=0.05, seed=0):
    """beta = KL 绳子的力度;0 表示不拴绳子"""
    rng = np.random.default_rng(seed)
    logits = np.log(REF_P).copy()          # 从 SFT 模型出发
    baseline = REF_P @ RM_SCORE            # 从 SFT 模型的平均分开始
    for _ in range(steps):
        p = softmax(logits)
        i = rng.choice(5, p=p)                       # 1. 生成一个回答
        r = RM_SCORE[i] - beta * np.log(p[i] / REF_P[i])   # 2. 打分,离 SFT 太远就扣分
        adv = r - baseline                           # 3. 比平时好多少?
        baseline += 0.05 * (r - baseline)            #    baseline = 最近的平均分
        grad = -p.copy(); grad[i] += 1               # 4. d log p[i] / d logits
        logits += lr * adv * grad                    # 5. 好就把这个回答的概率推高,差就压低
    return softmax(logits)

print("=== 第 0 步:一次更新长什么样 ===")
p = REF_P
print(f"抽到'详细准确',得分 6.5,平时平均 {p @ RM_SCORE:.2f} → 比平时好 → 推高它的概率")
print(f"抽到'敷衍一句',得分 1.0,比平时差 → 压低它的概率\n")

show("SFT 模型", REF_P)
print()
show("RLHF 不拴绳", train(beta=0.0))
print("  ↑ 模型发现'长篇废话'分最高,就几乎只说它了 —— 奖励作弊\n")
for beta in [1.0, 3.0, 10.0]:
    show(f"RLHF β={beta}", train(beta=beta))
    print()
print("KL 绳子:每次回答都按 log(p / p_SFT) 扣分,离 SFT 模型越远扣得越多。")
print("β 太小 → 作弊;β 太大 → 几乎没学到东西。")

# 彩蛋(L3 DPO 要用):拴绳子的 RLHF 最后会停在哪里,其实有现成公式
beta = 3.0
best = REF_P * np.exp(RM_SCORE / beta); best /= best.sum()
show("公式 β=3", best)
print("p_最优 ∝ p_SFT × e^(奖励/β) —— 和上面 β=3 训练出来的几乎一样")
