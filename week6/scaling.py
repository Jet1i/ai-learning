# 第 6 周 第 1 课：Scaling Laws —— 给定算力预算，模型该多大、数据该多少？
# 纯 Python + numpy，不用 GPU，不调 API。运行：python week6/scaling.py
import numpy as np

# ---------- 1. 参数量：第 3 周的公式 ----------
# 每层：注意力 4 个 d×d 矩阵 (Wq Wk Wv Wo) = 4d²；FFN 两个 d×4d 矩阵 = 8d²  → 12d²
# 再加词表 V×d（输出层和词表共享）。偏置、LayerNorm 很小，忽略。
def params(L, d, V=50257):
    return 12 * L * d**2 + V * d

print("== 1. 参数量 N ≈ 12·L·d² + V·d ==")
for name, L, d in [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024), ("GPT-2 XL", 48, 1600)]:
    print(f"{name:13s} L={L:2d} d={d:4d}  N ≈ {params(L, d)/1e6:6.0f}M")

# ---------- 2. 训练算力：C ≈ 6·N·D ----------
# 锚定 wx+b：每个 token 经过一个权重 w，要做 1 次乘法 + 1 次加法 = 2 FLOPs（前向）。
# 反向传播要算两份梯度（对 x 的、对 w 的），大约是前向的 2 倍 = 4 FLOPs。
# 所以：每个参数 × 每个 token ≈ 6 FLOPs。
def train_flops(N, D):
    return 6 * N * D

def gpu_days(C, tflops=312, mfu=0.4):
    # A100 bf16 峰值 312 TFLOPs，实际利用率(MFU)一般 30%~50%
    return C / (tflops * 1e12 * mfu) / 86400

print("\n== 2. 训练算力 C ≈ 6·N·D ==")
N, D = params(12, 768), 10e9          # GPT-2 small 训 100 亿 token
C = train_flops(N, D)
print(f"GPT-2 small × 10B token: C = {C:.2e} FLOPs ≈ {gpu_days(C):.1f} A100·天")

# ---------- 3. Chinchilla 定律：损失怎么随 N、D 下降 ----------
# Hoffmann et al. 2022 拟合出来的公式（常数直接用论文里的）：
#   Loss(N, D) = E + A / N^α + B / D^β
#   E   = 1.69 → 无论多大都去不掉的“语言本身的不确定性”
#   A/N^α     → 模型太小造成的损失
#   B/D^β     → 数据太少造成的损失
E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
def loss(N, D):
    return E + A / N**alpha + B / D**beta

print("\n== 3. 同样的算力，怎么分？(isoFLOP) ==")
C = 1e21                              # 固定预算
print(f"预算 C = {C:.0e} FLOPs（≈ {gpu_days(C):.0f} A100·天）")
print(f"{'N(参数)':>10s} {'D(token)':>10s} {'D/N':>7s} {'loss':>6s}")
Ns = np.logspace(7.5, 10.5, 13)       # 3000 万 ~ 300 亿参数
for N in Ns:
    D = C / (6 * N)                    # 预算固定：模型大了，数据就少了
    print(f"{N/1e9:9.2f}B {D/1e9:9.1f}B {D/N:7.1f} {loss(N, D):6.3f}")

# 精确找最优点
Ns = np.logspace(7, 11, 2000)
Ls = loss(Ns, C / (6 * Ns))
Nbest = Ns[Ls.argmin()]
print(f"→ 最优：N ≈ {Nbest/1e9:.2f}B，D ≈ {C/(6*Nbest)/1e9:.0f}B，D/N ≈ {C/(6*Nbest)/Nbest:.0f}")
print("  注意：用这组拟合常数算出来是几十；论文用另外两种方法得到约 20。")
print("  业界记的经验法则：每个参数配约 20 个 token（Chinchilla 比例）。重点是数量级：几十，不是 2，也不是 2000。")

# ---------- 4. 真实模型为什么“不按规矩来”？ ----------
print("\n== 4. 真实模型的 D/N ==")
for name, N, D in [("GPT-3 175B (2020)", 175e9, 300e9),
                   ("Chinchilla 70B",    70e9,  1.4e12),
                   ("LLaMA-1 7B",        6.7e9, 1.0e12),
                   ("LLaMA-3 8B",        8.0e9, 15e12)]:
    print(f"{name:18s} D/N = {D/N:6.0f}   loss≈{loss(N, D):.3f}   训练 ≈ {gpu_days(train_flops(N, D)):8.0f} A100·天")
# 思考：LLaMA-3 8B 的 D/N 远超 20，训练“不划算”，Meta 为什么还这么干？（见练习 3）
