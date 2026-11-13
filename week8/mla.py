"""
第 8 周 L3:MLA(多头潜在注意力)—— 把 KV cache 压扁
运行:python week8/mla.py

一句话:普通注意力每个 token 要给每个头存一份 k 和 v;
MLA 只存一个很短的"压缩向量" c(一个 wx 压下去),用的时候再展开成各个头的 k、v。
"""
import numpy as np

np.random.seed(0)
D, N_HEAD, D_HEAD, D_C = 8, 2, 4, 3   # 模型维度 8,2 个头,每头 4 维,压缩到 3 维
T = 5                                 # 已经有 5 个 token 在上下文里

X = np.random.randn(T, D)             # 5 个 token 的向量
q = np.random.randn(N_HEAD, D_HEAD)   # 当前 token 每个头的 query


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------- 1. 普通多头注意力(MHA):每个头的 k、v 都存 ----------
Wk = np.random.randn(N_HEAD, D, D_HEAD)
Wv = np.random.randn(N_HEAD, D, D_HEAD)
K_cache = np.stack([X @ Wk[h] for h in range(N_HEAD)])   # (2, 5, 4)
V_cache = np.stack([X @ Wv[h] for h in range(N_HEAD)])
print("=== 1. MHA ===")
print(f"每个 token 存 k+v:2 × {N_HEAD} 头 × {D_HEAD} 维 = {2 * N_HEAD * D_HEAD} 个数")

# ---------- 2. MLA:先压缩成 c,只存 c ----------
W_down = np.random.randn(D, D_C)                  # 8 -> 3,压缩
W_uk = np.random.randn(N_HEAD, D_C, D_HEAD)       # 3 -> 每头的 k
W_uv = np.random.randn(N_HEAD, D_C, D_HEAD)       # 3 -> 每头的 v
C_cache = X @ W_down                              # (5, 3)  ← 只存这个!
print("\n=== 2. MLA ===")
print(f"每个 token 只存 c:{D_C} 个数   (省到 {D_C / (2 * N_HEAD * D_HEAD):.0%})")
print(f"c 的缓存:\n{np.round(C_cache, 2)}")

# ---------- 3. 朴素用法:每一步把 c 展开成 k、v 再算 ----------
def mla_naive():
    outs = []
    for h in range(N_HEAD):
        K = C_cache @ W_uk[h]                # 现场展开 (5, 4)
        V = C_cache @ W_uv[h]
        w = softmax(K @ q[h] / np.sqrt(D_HEAD))
        outs.append(w @ V)
    return np.concatenate(outs)

# ---------- 4. 吸收技巧:不展开,把 W_uk 并到 q 上 ----------
#   q · k = q · (c W_uk) = (q W_ukᵀ) · c     矩阵乘法可以换个顺序结合
def mla_absorbed():
    outs = []
    for h in range(N_HEAD):
        q_c = W_uk[h] @ q[h]                 # 先把 q 变成 3 维 (只算一次)
        w = softmax(C_cache @ q_c / np.sqrt(D_HEAD))   # 直接和 c 点乘
        outs.append((w @ C_cache) @ W_uv[h]) # 先在 c 上加权,最后再展开一次
    return np.concatenate(outs)

a, b = mla_naive(), mla_absorbed()
print("\n=== 3/4. 朴素展开 vs 吸收技巧 ===")
print(f"朴素:{np.round(a, 3)}")
print(f"吸收:{np.round(b, 3)}")
print(f"最大差 {np.abs(a - b).max():.1e}  -> 结果一样,但吸收版不用把 5 个 token 都展开")

# ---------- 5. 真实账本:每个 token 每层存多少,整模型多少 ----------
print("\n=== 5. 真实模型:每个 token 的 KV cache(fp16 = 2 字节) ===")
def kb(n_numbers, layers):
    return n_numbers * layers * 2 / 1024

rows = [
    ("LLaMA-3 8B  GQA 8 个 KV 头", 2 * 8 * 128, 32),
    ("V3 如果用 MHA:128 头 × 128 维", 2 * 128 * 128, 61),
    ("V3 如果用 GQA:8 个 KV 头", 2 * 8 * 128, 61),
    ("V3 实际 MLA:c 512 + RoPE 64", 512 + 64, 61),
]
for name, n, L in rows:
    print(f"  {name:32s} 每层 {n:6d} 个数 × {L} 层 = {kb(n, L):8.0f} KB")
print("  128K 上下文时 MHA 版每个请求要 "
      f"{kb(2 * 128 * 128, 61) * 128 * 1024 / 1024 / 1024:.0f} GB,MLA 版 "
      f"{kb(576, 61) * 128 * 1024 / 1024 / 1024:.1f} GB")
