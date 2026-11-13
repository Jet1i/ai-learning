"""第 9 周 L1:奖励模型 —— 把"人更喜欢 A"变成一个分数
运行:python week9/reward.py

真实的奖励模型 = 一个 LLM + 最后一个 wx+b(输出一个数)。
这里把 LLM 省掉:每个回答直接用 4 个特征表示,奖励模型只剩那个 wx+b。
"""
import numpy as np

rng = np.random.default_rng(0)
FEATS = ["有用", "礼貌", "长度(百字)", "胡编"]

# 标注员心里的真实偏好(模型看不到,只能从"选 A 还是选 B"里猜)
TRUE_W = np.array([2.0, 1.0, 0.3, -3.0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------- 第 0 步:最小的例子 ----------
print("=== 第 0 步:两个回答,两个分数 ===")
rA, rB = 2.0, 0.5
print(f"A 得 {rA},B 得 {rB},差 {rA - rB}")
print(f"P(人选 A) = sigmoid({rA - rB}) = {sigmoid(rA - rB):.3f}")
print("分数本身没意义,只有'差'有意义:A、B 都加 100,概率不变\n")

# ---------- 造数据:标注员两两比较 ----------
def make_pairs(n):
    xa = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2, n),
                          rng.uniform(0, 5, n), rng.integers(0, 2, n)])
    xb = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2, n),
                          rng.uniform(0, 5, n), rng.integers(0, 2, n)])
    # 人也会犯错:分差越大越不容易选错
    p_a = sigmoid(xa @ TRUE_W - xb @ TRUE_W)
    a_wins = rng.random(n) < p_a
    chosen = np.where(a_wins[:, None], xa, xb)
    rejected = np.where(a_wins[:, None], xb, xa)
    return chosen, rejected

train_c, train_r = make_pairs(2000)
test_c, test_r = make_pairs(500)

# ---------- 训练奖励模型 r = x·w + b ----------
w = np.zeros(4)
b = 0.0
LR = 0.5
print("=== 训练:loss = -log sigmoid(r_chosen - r_rejected) ===")
for step in range(1001):
    diff = train_c @ w + b - (train_r @ w + b)     # b 在这里被减掉了
    p = sigmoid(diff)                              # 模型认为"人选 chosen"的概率
    loss = -np.log(p).mean()
    grad_w = -((1 - p)[:, None] * (train_c - train_r)).mean(axis=0)
    w -= LR * grad_w
    if step % 250 == 0:
        acc = (p > 0.5).mean()  # step 0 时 w=0,全是 0.5,所以是 0%
        print(f"step {step:3d}  loss {loss:.3f}  训练集上猜对 {acc:.1%}")

print("\n学到的 w vs 标注员心里的 w:")
for name, wl, wt in zip(FEATS, w, TRUE_W):
    print(f"  {name:8s} 学到 {wl:+.2f}   真实 {wt:+.2f}")

test_acc = ((test_c @ w - test_r @ w) > 0).mean()
ceiling = ((test_c @ TRUE_W - test_r @ TRUE_W) > 0).mean()
print(f"\n测试集:奖励模型和人一致 {test_acc:.1%}"
      f"(就算用真实 w 也只有 {ceiling:.1%},因为人自己会选错)")

# ---------- 用奖励模型给新回答打分 ----------
print("\n=== 给两个新回答打分 ===")
good = np.array([2.5, 0.5, 1.0, 0])   # 短、有用、不胡编
long_junk = np.array([0.3, 1.5, 10.0, 1])  # 很长、很客气、没用、还胡编
print(f"短而有用:   {good @ w:+.2f}")
print(f"长篇客套废话:{long_junk @ w:+.2f}")
