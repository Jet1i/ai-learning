"""
第 12 周 L1-a:图片是怎么变成 token 的(纯 numpy,不用下载任何模型)

一句话：VLM = 视觉编码器(把图切块 → 每块变一个向量) + 一个普通 LLM。
图片块向量和文字 token 向量排在同一个序列里，后面的 Transformer 完全不知道谁是图谁是字。

步骤(和第 3 周的 GPT 对照):
  文字：  "猫"  → 查 embedding 表 → 一个 d 维向量
  图片：  一块像素 → 拉平 → 乘 W 加 b(就是 wx+b)→ 一个 d 维向量
"""
import numpy as np
np.random.seed(0)

# ---------- 1. 最小例子：一张 4×4 的灰度图 ----------
img = np.array([
    [0, 0, 9, 9],
    [0, 0, 9, 9],
    [9, 9, 0, 0],
    [9, 9, 0, 0],
], dtype=float)          # 左上黑、右上白、左下白、右下黑
P = 2                    # patch 大小 2×2
print("原图 4×4:\n", img)

# ---------- 2. 切块(patch)----------
patches = []
for r in range(0, 4, P):
    for c in range(0, 4, P):
        patches.append(img[r:r+P, c:c+P].flatten())   # 2×2 → 4 个数
patches = np.array(patches)                            # 4 块 × 4 个数
print("\n切成", len(patches), "块，每块拉平成", patches.shape[1], "个数:\n", patches)

# ---------- 3. 每块做 wx+b,变成和文字一样宽的向量 ----------
d = 8                                  # 假设 LLM 的隐藏维度 d=8
W = np.random.randn(P * P, d) * 0.1    # 训练出来的投影矩阵(这里随机)
b = np.zeros(d)
img_tokens = patches @ W + b           # 4 块 → 4 个"图片 token",每个 8 维
print("\n图片 token 形状:", img_tokens.shape, "(4 个 token,每个 d=8)")

# ---------- 4. 和文字 token 拼成一个序列 ----------
vocab = {"这": 0, "张": 1, "图": 2, "里": 3, "有": 4, "什": 5, "么": 6}
emb = np.random.randn(len(vocab), d) * 0.1      # 文字 embedding 表
text = ["这", "张", "图", "里", "有", "什", "么"]
text_tokens = emb[[vocab[t] for t in text]]
seq = np.concatenate([img_tokens, text_tokens])  # [图1 图2 图3 图4 这 张 图 里 有 什 么]
print("拼起来的序列形状:", seq.shape, "→ 送进普通 Transformer(第 3 周那个)")
print("注意：两块颜色一样的 patch(块0 和 块3 都是黑)得到的向量完全相同:",
      np.allclose(img_tokens[0], img_tokens[3]))
print("  → 光有内容分不清位置，所以图片 token 也要加位置编码(和文字一样)")

# ---------- 5. 放回真实模型：一张图要多少 token?----------
def image_tokens(h, w, patch=14, merge=2):
    """ViT 式切块;很多 VLM(如 Qwen2-VL)再把相邻 2×2 块合并成 1 个 token"""
    n = (h // patch) * (w // patch)
    return n // (merge * merge)

print("\n真实规模(patch=14,2×2 合并):")
for h, w, name in [(224, 224, "缩略图"), (448, 448, "小图"),
                   (1024, 1024, "手机截图"), (1344, 1792, "A4 扫描件(高清)")]:
    n = image_tokens(h, w)
    print(f"  {name:12s} {h}×{w:<5d} → {n:5d} 个 token")

print("\n对比：一句 20 字的中文问题 ≈ 15~20 个 token。")
print("一张高清扫描件 ≈ 3000 个 token,和上周客服一轮的全部输入(约 3000)一样多。")
print("售前一句话：多模态的成本和延迟主要看【图片分辨率】,先问客户图多大、一次传几张。")
