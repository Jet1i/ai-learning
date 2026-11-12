"""第 7 周 L2：FlashAttention 的核心 —— 分块 + online softmax（numpy 版）

运行：python week7/flash.py

FlashAttention 不改数学（结果和普通注意力一模一样），只改"算的顺序"：
  普通版：把 T×T 的分数表 S 整张写进显存 → 再读出来做 softmax → 再写 → 再读出来乘 V
  Flash ：K、V 切成小块，一块一块搬到片上 SRAM，边算边更新结果，T×T 的表从不落显存
难点：softmax 要先知道"整行的总和"才能相除，分块时还没看到后面的块怎么办？
答案：online softmax —— 先按目前看到的算，看到新块时把旧结果"补救缩放"一下。
"""
import numpy as np

np.random.seed(0)

# ===============================================================
# 第 1 部分：一维热身 —— online softmax
# 普通 softmax 三步：取 e → 求和 → 相除（还要先减最大值防溢出）
# ===============================================================
s = np.array([1.0, 3.0, 0.5, 4.0, 2.0, 2.5])     # 一行注意力分数
v = np.array([10., 20., 30., 40., 50., 60.])     # 对应的 v（一维方便看）

p = np.exp(s - s.max())
ref = (p / p.sum()) @ v
print("普通 softmax 加权和:", round(ref, 4))

# online：一个一个（或一块一块）地看，只记 3 个数
m = -np.inf   # 目前见过的最大分数
l = 0.0       # 目前的 e 之和（以 m 为基准）
o = 0.0       # 目前的"未除以总和"的加权和（以 m 为基准）
for i in range(len(s)):
    m_new = max(m, s[i])
    scale = np.exp(m - m_new)          # 最大值变了 → 旧的 e 都要乘这个系数补救
    l = l * scale + np.exp(s[i] - m_new)
    o = o * scale + np.exp(s[i] - m_new) * v[i]
    m = m_new
    print(f"  看到第 {i+1} 个: m={m:.1f}  l={l:.4f}  当前答案 o/l={o/l:.4f}")
print("online softmax 加权和:", round(o / l, 4), "← 和上面一样\n")


# ===============================================================
# 第 2 部分：真正的注意力，Q K V 都是矩阵
# ===============================================================
T, d = 256, 64          # 序列长度、头维度
Q = np.random.randn(T, d)
K = np.random.randn(T, d)
V = np.random.randn(T, d)

def standard_attention(Q, K, V):
    S = Q @ K.T / np.sqrt(d)                 # T×T，整张表写进显存
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)     # 再读、再写一张 T×T
    return P @ V                             # 再读一次

def flash_attention(Q, K, V, Bc=32):
    """K/V 按 Bc 行一块；每次只有 T×Bc 的分数块存在（在真 GPU 上它待在 SRAM 里）"""
    m = np.full((T, 1), -np.inf)             # 每一行（每个 query）自己的 m、l、o
    l = np.zeros((T, 1))
    O = np.zeros((T, d))
    for j in range(0, T, Bc):
        Kj, Vj = K[j:j+Bc], V[j:j+Bc]        # 从 HBM 搬一块 K、V 进 SRAM
        Sj = Q @ Kj.T / np.sqrt(d)           # T×Bc 小块分数
        m_new = np.maximum(m, Sj.max(axis=1, keepdims=True))
        scale = np.exp(m - m_new)            # 补救旧结果
        Pj = np.exp(Sj - m_new)
        l = l * scale + Pj.sum(axis=1, keepdims=True)
        O = O * scale + Pj @ Vj
        m = m_new
    return O / l                             # 最后才统一"除以总和"

O_std = standard_attention(Q, K, V)
O_fla = flash_attention(Q, K, V)
print(f"T={T}, d={d}: 两种结果最大差 = {np.abs(O_std - O_fla).max():.2e}  （浮点误差级别，等于完全一样）")

# ===============================================================
# 第 3 部分：省在哪 —— 中间表 S 有多大
# ===============================================================
print("\n每个头、每层，普通注意力要在显存里放的 T×T 分数表（fp16）：")
for T_ in [2_048, 8_192, 32_768, 131_072]:
    S_bytes = T_ * T_ * 2
    print(f"  T={T_:>7,}: {S_bytes/2**20:>10,.0f} MB   Flash 只需一个 Bc×Bc 小块（如 128×128 = 32 KB，在 SRAM 里）")

print("""
要点：
  1. 计算量没变（还是 ~4·T²·d），省的是"T×T 表在 HBM 里来回读写"的搬运
  2. 显存从 O(T²) 降到 O(T)：长上下文能跑起来，主要靠这个
  3. 关键技巧 = online softmax：先按目前最大值算，最大值变了就乘 e^(m旧-m新) 补救，最后统一除以 l
""")
