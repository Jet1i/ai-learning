"""
第 5 周 L1：Embedding —— 把一句话变成一个向量，用"夹角"衡量意思像不像
运行:  python week5/embed1.py
第一次运行会下载 BAAI/bge-small-zh-v1.5（约 95MB，中文小模型，Mac CPU 就能跑）。
这里默认走国内镜像 hf-mirror.com（和第 3 周 gpt2.py 一样）。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 一个迷你"知识库"：5 句话
docs = [
    "退款申请提交后，3 到 5 个工作日原路退回。",
    "会员可以在设置页面修改绑定的手机号。",
    "订单发货后一般 48 小时内能看到物流信息。",
    "密码忘了可以用手机验证码重置。",
    "今天北京天气晴，最高气温 18 度。",
]
# 用户的问题：故意和知识库用词不一样
queries = ["钱多久能退回来？", "我登录不上了怎么办", "年假还有多久"]


def cosine(a, b):
    """余弦相似度 = 点积 / (长度 × 长度)，范围 -1~1，越大越像"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------- 方法 1：字袋向量（数每个字出现了几次）----------
vocab = sorted(set("".join(docs + queries)))


def char_vec(s):
    v = np.zeros(len(vocab))
    for ch in s:
        v[vocab.index(ch)] += 1
    return v


# ---------- 方法 2：Embedding 模型（一个小 Transformer）----------
name = "BAAI/bge-small-zh-v1.5"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()


def embed(texts):
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = model(**batch).last_hidden_state  # (句子数, token 数, 512)：每个 token 一个向量
    v = h[:, 0]                               # 只取第 0 个位置 [CLS] 的向量，代表整句
    v = torch.nn.functional.normalize(v, dim=-1)  # 把长度缩放成 1
    return v.numpy()


D = embed(docs)
print(f"知识库 {len(docs)} 句话 → 向量矩阵形状 {D.shape}（每句 {D.shape[1]} 个数字）\n")
print(tok.convert_ids_to_tokens(tok("快递到哪了")["input_ids"]))
for q in queries:
    qv = embed([q])[0]
    s_bge = D @ qv  # 长度都是 1，所以点积 = 余弦相似度
    s_char = [cosine(char_vec(q), char_vec(d)) for d in docs]
    print(f"问题：{q}")
    print("   bge   字袋   文档")
    for i in np.argsort(-s_bge):
        print(f"  {s_bge[i]:.3f}  {s_char[i]:.3f}  {docs[i]}")
    print()
