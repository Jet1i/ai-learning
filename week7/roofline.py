"""第 7 周 L1：为什么大模型推理是"搬运受限"（memory-bound）

运行：python week7/roofline.py

两部分：
  1. 真实实验：在你的 Mac 上测 x @ W，batch 从 1 涨到 256，看时间怎么变
  2. 纸面计算：A100 上 7B 模型，每秒最多生成多少 token（roofline 屋顶线）
"""
import time
import numpy as np

# ---------------------------------------------------------------
# 第 1 部分：真实实验
# W 是一层的权重（d×d），x 是 batch 个 token 的向量（batch×d）
# 每次 x @ W：
#   计算量 FLOPs = 2 * batch * d * d     （每个 w 一乘一加，锚定 wx+b）
#   搬运量 Bytes ≈ d * d * 4             （W 必须从内存读一遍，x 很小可忽略）
# ---------------------------------------------------------------
d = 4096
W = np.random.randn(d, d).astype(np.float32)   # 64 MB，比 CPU 缓存大得多

print(f"W: {d}x{d} float32 = {W.nbytes / 1e6:.0f} MB\n")
print(f"{'batch':>6} {'耗时ms':>8} {'每token ms':>11} {'GFLOP/s':>9} {'强度 FLOP/B':>12}")

for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    x = np.random.randn(batch, d).astype(np.float32)
    x @ W                                         # 预热
    times = []
    for _ in range(30):                           # 测 30 次取最快，减少抖动
        t0 = time.perf_counter()
        y = x @ W
        times.append(time.perf_counter() - t0)
    t = min(times)
    flops = 2 * batch * d * d
    intensity = flops / W.nbytes                  # 每搬 1 字节做几次运算
    print(f"{batch:>6} {t*1e3:>8.2f} {t*1e3/batch:>11.3f} {flops/t/1e9:>9.1f} {intensity:>12.1f}")

print("""
看什么：
  - batch 从 1 到 8 左右，总耗时几乎不涨 → 时间都花在"把 W 读进来"，不在算
  - 所以每个 token 的成本随 batch 变大而直线下降
  - batch 很大后耗时开始跟着 batch 涨 → 这时才真正"算不过来"（compute-bound）
""")

# ---------------------------------------------------------------
# 第 2 部分：屋顶线（roofline），纸面算 GPU
# ---------------------------------------------------------------
GPU = {                      # 公开规格（dense，不含稀疏加速）
    "A100 80GB": dict(tflops=312, bw_TBs=2.0, mem=80),
    "H100 SXM":  dict(tflops=989, bw_TBs=3.35, mem=80),
    "RTX 4090":  dict(tflops=165, bw_TBs=1.0, mem=24),
}

N = 7e9            # 参数量
bytes_per_w = 2    # fp16

print("=" * 60)
print(f"7B 模型 fp16，权重 {N*bytes_per_w/1e9:.0f} GB")
for name, g in GPU.items():
    ridge = g["tflops"] * 1e12 / (g["bw_TBs"] * 1e12)   # 屋脊点：强度超过它才是算力受限
    # decode：每生成 1 个 token，所有权重都要读一遍
    t_mem = N * bytes_per_w / (g["bw_TBs"] * 1e12)      # 读权重的时间
    t_cmp = 2 * N / (g["tflops"] * 1e12)                 # 算的时间（batch=1）
    print(f"\n{name}: 屋脊点 = {ridge:.0f} FLOP/字节")
    print(f"  batch=1 每个 token：读权重 {t_mem*1e3:.2f} ms vs 计算 {t_cmp*1e3:.3f} ms"
          f" → 上限约 {1/t_mem:.0f} token/s（算力利用率 {t_cmp/t_mem:.1%}）")
    # batch=B 时：读一遍权重，服务 B 个 token；强度 ≈ B（fp16 下 2B FLOP / 2 字节）
    B_star = ridge                                         # 强度=B，B 到屋脊点才吃满算力
    print(f"  batch 约 {B_star:.0f} 时才吃满算力 → 总吞吐约 {B_star/max(t_mem, B_star*t_cmp):.0f} token/s")
