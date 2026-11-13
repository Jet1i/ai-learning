"""
第 8 周 L1:MoE(混合专家)最小例子
运行:python week8/moe.py

一句话:普通 FFN 每个 token 都要用到全部参数;
MoE 把 FFN 拆成好几个小 FFN(专家),再加一个"路由器"(就是一个 wx+b),
每个 token 只挑 top-k 个专家来算。参数总量大,但每个 token 只用一小部分。
"""
import numpy as np

np.random.seed(0)

D = 4          # 每个 token 的向量维度(真实模型 4096~7168)
H = 8          # 每个专家 FFN 中间层宽度
N_EXPERT = 4   # 专家个数
TOP_K = 2      # 每个 token 挑几个专家


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()          # 别忘了 ÷总和


# ---------- 1. 专家 = 一个普通的小 FFN:relu(x W1) W2 ----------
experts = [
    (np.random.randn(D, H) * 0.5, np.random.randn(H, D) * 0.5)
    for _ in range(N_EXPERT)
]

def run_expert(i, x):
    W1, W2 = experts[i]
    return np.maximum(0, x @ W1) @ W2


# ---------- 2. 路由器 = 一个 wx(没有 b),输出每个专家的分数 ----------
W_router = np.random.randn(D, N_EXPERT)


def moe_forward(x, verbose=False):
    scores = x @ W_router                    # 4 个分数,和 wx+b 一样
    probs = softmax(scores)                  # 变成概率
    top = np.argsort(probs)[::-1][:TOP_K]    # 挑最大的 TOP_K 个
    gate = probs[top] / probs[top].sum()     # 只在被选中的里重新归一化
    out = sum(g * run_expert(i, x) for g, i in zip(gate, top))
    if verbose:
        print(f"  路由分数 {np.round(scores, 2)}")
        print(f"  softmax  {np.round(probs, 2)}  (和 = {probs.sum():.2f})")
        print(f"  选中专家 {[int(i) for i in top]}  权重 {np.round(gate, 2)}")
        print(f"  输出     {np.round(out, 2)}")
    return out, top


# ---------- 3. 先看 3 个 token 各自怎么走 ----------
print("=== 第一步:3 个 token 各自被分给谁 ===")
tokens = np.random.randn(3, D)
for t, x in enumerate(tokens):
    print(f"token {t}: x = {np.round(x, 2)}")
    moe_forward(x, verbose=True)
    print()

# ---------- 4. 算账:参数总量 vs 每个 token 实际用到的参数 ----------
per_expert = D * H + H * D
total = N_EXPERT * per_expert + D * N_EXPERT     # 专家 + 路由器
active = TOP_K * per_expert + D * N_EXPERT
print("=== 第二步:算账 ===")
print(f"每个专家参数   {per_expert}")
print(f"总参数         {total}   (要放进显存的)")
print(f"每 token 激活  {active}   (每个 token 真正乘到的)")
print(f"激活比例       {active / total:.0%}")
print()

# ---------- 5. 1000 个 token 进来,每个专家各接了多少活? ----------
print("=== 第三步:1000 个 token,每个专家接了多少活 ===")
load = np.zeros(N_EXPERT, dtype=int)
for x in np.random.randn(1000, D):
    _, top = moe_forward(x)
    load[top] += 1
for i, n in enumerate(load):
    print(f"专家 {i}: {n:4d} 次  " + "#" * (n // 20))
print(f"理想情况每个专家 {1000 * TOP_K // N_EXPERT} 次")
print("现在很均匀,因为路由器是随机的。训练时会出现'强者越强':")
print("被选多的专家学得更好 -> 更容易被选 -> 其他专家饿死。(下一课的主题)")
