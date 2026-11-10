# 第 6 周 第 3 课：RoPE 旋转位置编码 —— 用“转角度”代替“加位置表”
# numpy。运行：python week6/rope.py
import numpy as np
np.random.seed(0)

# ================= 1. 二维直觉：把向量转一个角度 =================
def rot2(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

q = np.array([1.0, 0.5])      # “咬”这个词的 query（假装只有 2 维）
k = np.array([0.8, -0.3])     # “狗”这个词的 key
theta = 0.3                   # 每往后一个位置，多转 0.3 弧度

print("== 1. 二维：第 m 个位置的 q 转 m·θ，第 n 个位置的 k 转 n·θ ==")
for m, n in [(2, 5), (10, 13), (100, 103), (2, 9)]:
    score = rot2(q, m * theta) @ rot2(k, n * theta)
    print(f"q 在位置 {m:3d}，k 在位置 {n:3d}，距离 {n-m}：score = {score:.4f}")
print("→ 距离一样，分数就一样！点积只看两个向量的“夹角差”，绝对位置被抵消了。")

# 对比：GPT-2 的做法 —— 把位置向量“加”上去
P = np.random.randn(200, 2) * 0.5      # 一张随机的位置表（训练前）
print("\n对比 GPT-2 加位置表：")
for m, n in [(2, 5), (10, 13), (100, 103)]:
    score = (q + P[m]) @ (k + P[n])
    print(f"q 在位置 {m:3d}，k 在位置 {n:3d}，距离 {n-m}：score = {score:.4f}")
print("→ 距离一样，分数却不一样：模型得自己从数据里“学会”相对距离。")

# ================= 2. 高维：每两维一组，各转各的速度 =================
# d 维向量切成 d/2 对，第 i 对的转速 θ_i = base^(-2i/d)
# 前几对转得飞快（像秒针），后几对转得很慢（像时针）：
#   秒针分辨“隔 1 个词还是 2 个词”，时针分辨“隔 100 个词还是 1000 个词”
def rope(x, pos, base=10000):
    d = x.shape[-1]
    i = np.arange(d // 2)
    theta = base ** (-2 * i / d)              # (d/2,) 每一对的转速
    ang = pos[:, None] * theta[None, :]       # (T, d/2) 每个位置每一对转多少
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]       # 偶数维、奇数维配对
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

d = 64
print(f"\n== 2. d={d}：{d//2} 对“指针”的转速 ==")
theta = 10000 ** (-2 * np.arange(d // 2) / d)
for i in [0, 1, 8, 16, 31]:
    print(f"第 {i:2d} 对：每个位置转 {theta[i]:.5f} 弧度，转一圈要 {2*np.pi/theta[i]:9.0f} 个位置")

# ================= 3. 放进注意力：只转 Q 和 K =================
T = 8
Q, K = np.random.randn(T, d), np.random.randn(T, d)
pos = np.arange(T)
S1 = rope(Q, pos) @ rope(K, pos).T
S2 = rope(Q, pos + 1000) @ rope(K, pos + 1000).T       # 整句话往后挪 1000 个位置
print("\n== 3. 整句往后挪 1000 个位置，注意力分数变了吗？ ==")
print("最大差异：", np.abs(S1 - S2).max().round(8), " → 没变（只是浮点误差）")
print("所以 RoPE 不需要位置表、没有可学参数，也没有“表只有 1024 行”这种硬上限。")
print("（但超过训练时见过的长度，效果还是会变差；长上下文模型会用“位置插值”、YaRN 等技巧把角度压缩回训练范围。）")
print("（V 不转：V 是“要传的内容”，位置只影响“看谁”，不影响“拿什么”。）")

# ================= 4. 远处的词，影响自然变小 =================
q1 = np.ones(d); k1 = np.ones(d)        # 两个完全相同的向量
print("\n== 4. 同一对向量，距离越远，RoPE 后的点积 ==")
for dist in [0, 1, 4, 16, 64, 256, 1024]:
    s = rope(q1[None], np.array([0]))[0] @ rope(k1[None], np.array([dist]))[0]
    print(f"距离 {dist:5d}：score = {s:7.2f}")
print("→ 整体趋势是越远越小（有波动）：RoPE 天然带一点“近的词更重要”的偏好。")
