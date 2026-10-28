# 第 3 周 第 1 课：自注意力（self-attention），只用 numpy
import numpy as np

np.random.seed(0)
words = ["我", "爱", "吃", "苹果"]
T, d = len(words), 8          # T 个词，每个词用 d 维向量表示

# 1) 词向量：真实模型里是学出来的，这里先用随机数代替
X = np.random.randn(T, d)     # 形状 (T, d)，一行一个词

# 2) 三个可学习的矩阵：把每个词变成 Query / Key / Value
Wq = np.random.randn(d, d) / np.sqrt(d)
Wk = np.random.randn(d, d) / np.sqrt(d)
Wv = np.random.randn(d, d) / np.sqrt(d)
Q = X @ Wq    # 我想找什么
K = X @ Wk    # 我是什么（给别人匹配用）
V = X @ Wv    # 我真正要交出去的内容

# 3) 打分：每个词的 Q 和所有词的 K 做点积，越像分越高
scores = Q @ K.T / np.sqrt(d)  # (T, T)，除以 √d 防止分数太大

# 4) softmax：把每一行分数变成加起来 = 1 的权重
def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)

# 5) 输出：每个词 = 所有词 V 的加权平均
out = weights @ V              # (T, d)

np.set_printoptions(precision=2, suppress=True)
print("注意力权重（行 = 谁在看，列 = 看谁）:")
print("      " + "  ".join(f"{w:>4}" for w in words))
for w, row in zip(words, weights):
    print(f"{w:>4}  " + "  ".join(f"{p:4.2f}" for p in row))
print("\n每行之和:", weights.sum(axis=1))
print("输出形状:", out.shape, "（和输入一样，每个词还是 d 维，但已经“混入”了别的词的信息）")
