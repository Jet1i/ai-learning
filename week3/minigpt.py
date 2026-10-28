# 第 3 周 第 3 课：用交叉熵训练一个迷你 GPT（PyTorch）
# 任务：读入一些简单的中文句子，学会“猜下一个字”，然后自己写句子
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); random.seed(0)

# ---- 1) 造语料：规则很简单，但模型事先不知道 ----
people = ["小明", "小红", "老师", "妈妈", "我", "他"]
foods  = ["苹果", "面条", "米饭", "西瓜"]
drinks = ["水", "茶", "牛奶"]
places = ["学校", "公园", "超市"]
def sentence():
    p = random.choice(people)
    return random.choice([
        f"{p}吃{random.choice(foods)}。",
        f"{p}喝{random.choice(drinks)}。",
        f"{p}去{random.choice(places)}。",
        f"{p}在{random.choice(places)}吃{random.choice(foods)}。",
    ])
text = "".join(sentence() for _ in range(3000))
chars = sorted(set(text))
V = len(chars)                                # 词表大小（这里一个字 = 一个 token）
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(encode(text))
print(f"语料 {len(text)} 字，词表 {V} 个字：{''.join(chars)}")

# ---- 2) 模型：就是 block.py 的 PyTorch 版本 ----
T, d, H, L = 16, 64, 4, 2                     # 上下文长度、维度、头数、层数

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)        # Wq、Wk、Wv 合成一个矩阵
        self.proj = nn.Linear(d, d)           # Wo
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.att = None                       # 存一下注意力，方便看
    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        split = lambda m: m.view(B, t, H, d // H).transpose(1, 2)   # (B, H, t, dh)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // H)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
        w = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        self.att = w.detach()
        a = (w @ v).transpose(1, 2).reshape(B, t, d)
        x = x + self.proj(a)                  # 残差
        return x + self.ffn(self.ln2(x))      # 残差

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, d)         # 词向量表
        self.pos = nn.Embedding(T, d)         # 位置向量表
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)           # 把 d 维向量变成 V 个分数（logits）
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln(self.blocks(x)))   # (B, t, V)

model = MiniGPT()
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

# ---- 3) 训练数据：输入是一段字，答案是“往后挪一位”的同一段字 ----
def batch(B=32):
    i = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[j:j + T] for j in i])
    y = torch.stack([data[j + 1:j + T + 1] for j in i])
    return x, y

@torch.no_grad()
def generate(start="小", n=40, temperature=1.0):
    idx = torch.tensor([encode(start)])
    for _ in range(n):
        logits = model(idx[:, -T:])[:, -1, :] / temperature   # 只要最后一个位置的分数
        probs = F.softmax(logits, dim=-1)                     # 分数 → 概率
        nxt = torch.multinomial(probs, 1)                     # 按概率抽一个字
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())

valid = {f"{p}{s}" for p in people for s in
         [f"吃{f}。" for f in foods] + [f"喝{x}。" for x in drinks] + [f"去{x}。" for x in places] +
         [f"在{x}吃{f}。" for x in places for f in foods]}
def score(sample):                            # 生成的句子里，有几成是“合法”的
    ss = [s + "。" for s in sample.split("。")[1:-1]]
    return sum(s in valid for s in ss) / max(1, len(ss))

x, y = batch()
print(f"\n训练前 loss = {F.cross_entropy(model(x).view(-1, V), y.view(-1)).item():.2f}"
      f"（瞎猜的话应该是 ln({V}) = {math.log(V):.2f}）")
print("训练前生成：", generate())

# ---- 4) 训练循环：还是那 4 步 ----
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(1, 801):
    x, y = batch()
    logits = model(x)                                         # ① 前向
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))    #   交叉熵 = −log(正确答案的概率)
    opt.zero_grad()                                           # ② 梯度清零
    loss.backward()                                           # ③ 反向传播
    opt.step()                                                # ④ 更新参数
    if step in (1, 10, 30, 60) or step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}  正确答案平均概率≈{math.exp(-loss.item()):.2f}")

s = generate(n=200)
print("\n训练后生成：", s[:80])
print(f"合法句子比例：{score(s):.0%}")

# ---- 5) 看一眼：模型觉得“小明吃”后面是什么？ ----
with torch.no_grad():
    probs = F.softmax(model(torch.tensor([encode("小明吃")]))[0, -1], dim=-1)
top = probs.topk(5)
print("\n“小明吃”的下一个字：", "  ".join(f"{chars[i]} {p:.2f}" for p, i in zip(top.values, top.indices)))
