# 第 6 周 第 2 课：从 GPT-2 块到 LLaMA 块 —— RMSNorm、SwiGLU、去掉偏置
# numpy，只做前向，和第 3 周的 week3/block.py 对照着看。运行：python week6/llama_block.py
import numpy as np
np.random.seed(0)

d, T = 16, 5
x = np.random.randn(T, d) * 3 + 1          # 5 个 token，故意让均值≠0、方差≠1
r = lambda *s: np.random.randn(*s) / np.sqrt(s[0])

# ================= 改动 1：LayerNorm → RMSNorm =================
# LayerNorm：减均值，再除以标准差，再乘 g 加 b（两组可学参数）
def layernorm(x, g=1, b=0):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * g + b

# RMSNorm：不减均值，只除以“均方根”，只乘 g（一组参数），少一半计算
def rmsnorm(x, g=1):
    rms = np.sqrt((x**2).mean(-1, keepdims=True) + 1e-5)
    return x / rms * g

print("== 1. LayerNorm vs RMSNorm ==")
for name, f in [("LayerNorm", layernorm), ("RMSNorm", rmsnorm)]:
    y = f(x)
    print(f"{name:9s} 均值={y.mean(-1)[:3].round(2)}  均方根={np.sqrt((y**2).mean(-1))[:3].round(2)}")
print("→ RMSNorm 只管“大小”（均方根=1），不管均值。实验发现减均值这一步几乎没用，省掉更快。")

# ================= 改动 2：ReLU/GELU FFN → SwiGLU FFN =================
def gelu(z):  return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))
def silu(z):  return z / (1 + np.exp(-z))          # 也叫 Swish：z · sigmoid(z)

# GPT-2 的 FFN：两个矩阵  d → 4d → d
W1, W2 = r(d, 4 * d), r(4 * d, d)
def ffn_gpt2(x):
    return gelu(x @ W1) @ W2

# LLaMA 的 SwiGLU：三个矩阵。锚定 wx+b：
#   a = x·W_gate  → 过 silu，当“开关”：每一维开多大
#   b = x·W_up    → 真正要传的内容
#   (silu(a) * b) · W_down  → 开关乘内容，再压回 d 维
# 为了参数量和 GPT-2 一样，中间维度从 4d 缩到 (8/3)d：3 × (8/3)d² = 8d² = 2 × 4d²
h = int(8 * d / 3)
Wg, Wu, Wd = r(d, h), r(d, h), r(h, d)
def ffn_swiglu(x):
    return (silu(x @ Wg) * (x @ Wu)) @ Wd

print("\n== 2. FFN：GELU vs SwiGLU ==")
print(f"GPT-2  FFN 参数：2 × {d}×{4*d}       = {W1.size + W2.size}")
print(f"SwiGLU FFN 参数：3 × {d}×{h}        = {Wg.size + Wu.size + Wd.size}  （几乎一样）")
z = np.array([-3, -1, -0.5, 0, 0.5, 1, 3.])
print("z      :", z)
print("ReLU(z):", np.maximum(0, z))
print("GELU(z):", gelu(z).round(2))
print("SiLU(z):", silu(z).round(2), " ← 负数不是直接变 0，有一点点负值，梯度更平滑")

# ================= 改动 3：所有线性层去掉偏置 b =================
# GPT-2 的 x@W 后面都有 +b；LLaMA 全部去掉，只剩 x@W。参数少一点，训练更稳，效果不掉。

# ================= 组装：LLaMA 块 =================
H = 4; dh = d // H
Wq, Wk, Wv, Wo = r(d, d), r(d, d), r(d, d), r(d, d)
def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attention(x):
    T = len(x)
    split = lambda m: m.reshape(T, H, dh).transpose(1, 0, 2)
    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    # （L3 会在这里给 Q、K 加 RoPE 旋转；L4 会让多个头共享 K、V）
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.triu(np.ones((T, T)), 1) == 1, -np.inf, s)
    return (softmax(s) @ V).transpose(1, 0, 2).reshape(T, d) @ Wo

def llama_block(x):
    x = x + attention(rmsnorm(x))     # Pre-Norm：先归一化再进子层，残差走“干净”的主干
    x = x + ffn_swiglu(rmsnorm(x))
    return x

y = llama_block(x)
print("\n== 3. LLaMA 块前向 ==")
print("输入", x.shape, "→ 输出", y.shape, "（形状不变，所以可以叠 32 层）")

# ================= 参数量：LLaMA-2 7B 真的是 7B 吗？ =================
def llama_params(L, d, hidden, V):
    attn = 4 * d * d                 # Wq Wk Wv Wo，无偏置
    ffn  = 3 * d * hidden            # Wgate Wup Wdown
    norm = 2 * d                     # 两个 RMSNorm 的 g
    return L * (attn + ffn + norm) + V * d * 2 + d   # 词表 + 输出头（LLaMA 不共享）+ 最后一个 norm

print("\n== 4. LLaMA-2 7B 参数量 ==")
N = llama_params(L=32, d=4096, hidden=11008, V=32000)
print(f"L=32, d=4096, FFN hidden=11008, V=32000  →  N = {N/1e9:.2f}B")
