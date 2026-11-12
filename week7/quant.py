"""第 7 周 L4：量化 —— 用 numpy 手写 int8 / int4，看误差和省下的显存

运行：python week7/quant.py

三部分：
  1. 最小例子：4 个权重压成 int8、int4
  2. 异常值问题：整体一个刻度 vs 每 128 个一组
  3. 真实大小：4096×4096 的权重矩阵，量化后 x @ W 的误差
"""
import numpy as np

np.random.seed(0)


def quantize(w, bits, group=None):
    """对称量化：返回 (整数, 刻度)。group=None 表示整个张量共用一个刻度"""
    qmax = 2 ** (bits - 1) - 1                       # int8 → 127，int4 → 7
    if group is None:
        scale = np.abs(w).max() / qmax
        q = np.clip(np.round(w / scale), -qmax, qmax)
        return q, scale
    g = w.reshape(-1, group)                         # 每 group 个一组
    scale = np.abs(g).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(g / scale), -qmax, qmax)
    return q, scale


def dequantize(q, scale, shape):
    return (q * scale).reshape(shape)


# ---------------------------------------------------------------
print("=" * 60, "\n第 1 部分：最小例子\n")
w = np.array([0.12, -0.53, 0.98, -0.09])
for bits in [8, 4]:
    q, s = quantize(w, bits)
    back = dequantize(q, s, w.shape)
    print(f"int{bits}: 刻度 = {s:.4f}  整数 = {q.astype(int)}  乘回来 = {np.round(back, 4)}  最大误差 = {np.abs(back - w).max():.4f}")

# ---------------------------------------------------------------
print("\n" + "=" * 60, "\n第 2 部分：混进一个异常值 10.0\n")
w2 = np.concatenate([w, [10.0]])
for bits in [8, 4]:
    q, s = quantize(w2, bits)
    back = dequantize(q, s, w2.shape)
    print(f"int{bits} 整体一个刻度：刻度 = {s:.3f}  整数 = {q.astype(int)}  乘回来 = {np.round(back, 3)}")
print("→ int4 时 0.12 和 -0.09 都变成 0 了：一个异常值让刻度变得太粗")

# ---------------------------------------------------------------
print("\n" + "=" * 60, "\n第 3 部分：真实大小的一层 4096×4096\n")
d = 4096
W = np.random.randn(d, d).astype(np.float32) * 0.02          # 真实权重大多是很小的数
idx = np.random.choice(W.size, 200, replace=False)            # 混进 200 个异常值
W.flat[idx] = np.random.choice([-1, 1], 200) * np.random.uniform(0.2, 0.3, 200)   # 比普通权重大 10~15 倍
x = np.random.randn(8, d).astype(np.float32)                  # 8 个 token
y_ref = x @ W


def rel_err(Wq):
    y = x @ Wq
    return np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)


print(f"{'方案':<22}{'每参数 bit':>10}{'这层大小':>10}{'输出相对误差':>14}")
print(f"{'fp16（原版）':<22}{16:>10}{d*d*2/2**20:>8.0f}MB{0:>13.2%}")
for bits in [8, 4]:
    for group in [None, 128]:
        q, s = quantize(W, bits, group)
        Wq = dequantize(q, s, W.shape)
        eff = bits + (16 / group if group else 0)              # 每组多存一个 fp16 刻度
        name = f"int{bits} " + ("整体一个刻度" if group is None else f"每 {group} 个一组")
        print(f"{name:<22}{eff:>10.3f}{d*d*eff/8/2**20:>8.0f}MB{rel_err(Wq):>13.2%}")

print("""
看什么：
  - int8：整体一个刻度误差几个 %，分组后不到 1% → 基本无损
  - int4 整体一个刻度：刻度被异常值撑得太粗，大部分小权重直接变成 0，误差爆炸
  - int4 分组：误差降到 10% 左右，大小只多了 3%（4.125 bit vs 4 bit）→ 所以 GPTQ/AWQ/GGUF 都分组
  - 10% 听起来很大，但这是随机权重 + 最简单的四舍五入；真实模型比较"耐噪声"，
    GPTQ/AWQ 还会把误差挪到不重要的地方，实际评测 int4 一般只掉 1~2 个点
""")

# ---------------------------------------------------------------
print("=" * 60, "\n速算：70B 模型放一张 80GB 卡\n")
for name, b in [("fp16", 16), ("int8", 8), ("int4 分组", 4.125)]:
    gb = 70e9 * b / 8 / 1e9
    left = 80 - gb - 4
    print(f"  {name:<10} 权重 {gb:>6.1f} GB → {'放不下' if left < 0 else f'剩 {left:.0f} GB 给 KV Cache'}")
