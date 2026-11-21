"""
第 12 周 L1-b:用真模型 CLIP 让"图片"和"文字"算相似度

CLIP = 两个编码器:
  图片编码器(ViT,就是 L1-a 那套切块 + Transformer)→ 一个向量
  文字编码器(Transformer)                         → 一个向量
训练目标：配对的(图, 描述)向量靠近，不配对的推远(对比学习)。
训完以后，图和字落在同一个向量空间里，可以直接算余弦相似度——和第 5 周 bge 检索一样。

运行：python week12/clip1.py   (第一次会下载约 600MB 模型)
国内网络：先 export HF_ENDPOINT=https://hf-mirror.com
"""
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

# ---------- 1. 自己画 3 张图(不用找图片)----------
def draw(shape, color):
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([40, 40, 184, 184], fill=color)
    elif shape == "square":
        d.rectangle([40, 40, 184, 184], fill=color)
    elif shape == "triangle":
        d.polygon([(112, 30), (194, 190), (30, 190)], fill=color)
    return img

images = [draw("circle", "red"), draw("square", "blue"), draw("triangle", "green")]
names = ["红圆", "蓝方", "绿三角"]

# ---------- 2. 候选文字 ----------
texts_en = ["a red circle", "a blue square", "a green triangle", "a photo of a cat"]
texts_zh = ["一个红色的圆", "一个蓝色的正方形", "一个绿色的三角形", "一张猫的照片"]

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
proc = CLIPProcessor.from_pretrained(model_id)

def scores(texts):
    inputs = proc(text=texts, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    # 图片向量、文字向量(都已归一化)
    img_v = out.image_embeds          # [3, 512]
    txt_v = out.text_embeds           # [4, 512]
    cos = img_v @ txt_v.T             # 余弦相似度 [3 张图, 4 句话]
    prob = out.logits_per_image.softmax(dim=-1)   # 乘温度后 softmax
    return cos, prob

for label, texts in [("英文描述", texts_en), ("中文描述", texts_zh)]:
    cos, prob = scores(texts)
    print(f"\n===== {label} =====")
    print("每张图对每句话的概率(softmax 后，每行加起来 = 1):")
    print(" " * 8 + "".join(f"{t[:14]:>16s}" for t in texts))
    for i, n in enumerate(names):
        row = "".join(f"{p:16.2f}" for p in prob[i].tolist())
        pick = texts[prob[i].argmax()]
        print(f"{n:6s}{row}   → 选了: {pick}")
    print("原始余弦相似度(注意都挤在 0.2~0.3 附近):")
    print(cos.round(decimals=3))

print("\n图片向量维度:", model.config.projection_dim)
print("CLIP 的 ViT-B/32:224×224 切成 32×32 的块 → 7×7 = 49 个 patch token(+1 个 CLS)")
