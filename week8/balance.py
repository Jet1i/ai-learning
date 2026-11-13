"""
第 8 周 L2:负载均衡 —— 专家为什么会"饿死",DeepSeek-V3 怎么不用辅助损失就解决
运行:python week8/balance.py

玩具设定(故意简化,只保留"强者越强"这一个机制):
  路由分数 = x·W(随机,每个 token 不同) + skill[i](专家 i 的"本事")
  被选中的专家会被训练 → skill 涨一点 → 下次更容易被选中
"""
import numpy as np

N_EXPERT, TOP_K, D = 8, 2, 16
STEPS, BATCH = 200, 256
GAMMA = 0.05      # DeepSeek-V3 偏置的调整步长


def bar(load):
    ideal = BATCH * TOP_K / N_EXPERT
    for i, n in enumerate(load):
        print(f"  专家 {i}: {int(n):4d} " + "#" * int(n / ideal * 10))
    print(f"  (理想每个 {ideal:.0f};最忙的是理想的 {load.max() / ideal:.1f} 倍)")


def train(method, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(D, N_EXPERT)) * 0.3
    skill = np.zeros(N_EXPERT)       # "本事",被训练就涨
    bias = np.zeros(N_EXPERT)        # DeepSeek-V3 的偏置,只管挑人
    for step in range(STEPS):
        x = rng.normal(size=(BATCH, D))
        score = x @ W + skill                       # (256, 8)
        pick_score = score + bias if method == "bias" else score
        top = np.argsort(-pick_score, axis=1)[:, :TOP_K]
        load = np.bincount(top.ravel(), minlength=N_EXPERT).astype(float)

        # 强者越强:接活越多的专家,本事涨得越多
        skill += 0.05 * load / BATCH

        if method == "bias":
            # 忙的专家偏置降一点,闲的升一点(步长 GAMMA)
            ideal = BATCH * TOP_K / N_EXPERT
            bias -= GAMMA * np.sign(load - ideal)
    return load, skill, bias


print("=== 1. 不管它:强者越强 ===")
load, skill, _ = train("none")
bar(load)
print(f"  skill = {np.round(skill, 2)}")
print("  -> 少数专家包揽所有活,其余专家参数白占显存;")
print("     多卡部署时(每张卡放几个专家),最忙那张卡决定整步耗时。\n")

print("=== 2. DeepSeek-V3:无辅助损失的偏置 ===")
load, skill, bias = train("bias")
bar(load)
print(f"  bias = {np.round(bias, 2)}")
print("  -> 偏置只用来'挑谁',不进加权相加的权重,所以不改变输出的数学含义。\n")

# ---------- 3. 细粒度专家:组合数爆炸 ----------
from math import comb
print("=== 3. DeepSeekMoE 的细粒度专家:同样的计算量,组合多得多 ===")
for n, k, name in [(8, 2, "Mixtral 式:8 个大专家选 2"),
                   (64, 16, "切细 8 倍:64 个小专家选 16"),
                   (256, 8, "DeepSeek-V3:256 个路由专家选 8")]:
    print(f"  {name:28s} 组合数 C({n},{k}) = {comb(n, k):.3g}")
print("  组合越多,每个 token 能拼出越精细的'专家团队'。")
print("  另外 V3 还有 1 个共享专家:每个 token 都走,存通用知识,路由专家就不用重复学。")
