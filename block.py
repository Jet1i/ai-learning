# 第 3 周 第 2 课：多头注意力 + 一个完整的 Transformer block（numpy，只做前向）
import numpy as np

np.random.seed(0)
vocab = ["狗", "咬", "人", "我", "爱", "吃", "苹果"]
d, H = 16, 4                  # 向量维度 16，4 个头，每个头 16/4 = 4 维
dh = d // H
r = lambda *s: np.random.randn(*s) / np.sqrt(s[0])   # 随机初始化的参数

# ---- 参数（训练时全都要学）----
E   = r(len(vocab), d)        # 词向量表：一行一个 token
P   = r(10, d)                # 位置向量表：第 0,1,2... 个位置各一行
Wq, Wk, Wv, Wo = r(d, d), r(d, d), r(d, d), r(d, d)
W1, W2 = r(d, 4 * d), r(4 * d, d)   # 前馈网络：16 → 64 → 16

def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def layernorm(x):             # 把每个向量标准化成均值 0、方差 1，训练更稳
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)

def multi_head_attention(x, causal):
    T = len(x)
    # (T, d) → (H, T, dh)：把 16 维切成 4 份，每个头各管一份
    split = lambda m: m.reshape(T, H, dh).transpose(1, 0, 2)
    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)        # (H, T, T)：每个头一张注意力图
    if causal:
        scores = np.where(np.triu(np.ones((T, T)), 1) == 1, -np.inf, scores)
    w = softmax(scores)
    out = (w @ V).transpose(1, 0, 2).reshape(T, d)         # 4 个头的结果拼回 16 维
    return out @ Wo, w

def block(x, causal=True):
    a, w = multi_head_attention(layernorm(x), causal)
    x = x + a                                              # 残差：在原来的基础上“加点东西”
    f = np.maximum(0, layernorm(x) @ W1) @ W2              # 前馈网络：每个词自己想一想
    x = x + f                                              # 再一次残差
    return x, w

def embed(words, use_pos):
    x = E[[vocab.index(t) for t in words]]
    return x + P[:len(words)] if use_pos else x

# ---- 跑一句话，看形状 ----
words = ["我", "爱", "吃", "苹果"]
x = embed(words, use_pos=True)
y, w = block(x)
print("输入形状:", x.shape, " 输出形状:", y.shape, " 注意力图:", w.shape, "(头数, T, T)")
np.set_printoptions(precision=2, suppress=True)
for h in range(H):
    print(f"\n第 {h} 个头的注意力：\n{w[h]}")

n = sum(m.size for m in [Wq, Wk, Wv, Wo, W1, W2])
print(f"\n这个 block 有 {n} 个参数（不算词向量表）")
