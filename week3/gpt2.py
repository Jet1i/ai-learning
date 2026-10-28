# 第 3 周 第 4 课：加载真的 GPT-2（1.24 亿参数），看它怎么分词、怎么生成
# 第一次运行会下载约 500MB 模型；下载慢的话先执行：export HF_ENDPOINT=https://hf-mirror.com
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")  # eager 才能取出注意力
model.eval()

# ---- 1) 分词器：文字 → token id ----
for s in ["I love eating apples.", "我爱吃苹果"]:
    ids = tok.encode(s)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{s!r}: {len(ids)} 个 token")
    print("   id   :", ids)
    print("   片段 :", pieces)
print("词表大小:", tok.vocab_size)

# ---- 2) 模型有多大 ----
n = sum(p.numel() for p in model.parameters())
cfg = model.config
print(f"\n参数量: {n:,}   层数 L={cfg.n_layer}  维度 d={cfg.n_embd}  头数 H={cfg.n_head}  上下文 T={cfg.n_positions}")

# ---- 3) 下一个 token 的概率：和 minigpt.py 里“小明吃”一样 ----
prompt = "The capital of France is"
ids = torch.tensor([tok.encode(prompt)])
with torch.no_grad():
    logits = model(ids).logits[0, -1]          # 最后一个位置对 50257 个 token 的分数
probs = F.softmax(logits, dim=-1)
top = probs.topk(8)
print(f"\n“{prompt}” 的下一个 token：")
for p, i in zip(top.values, top.indices):
    print(f"   {tok.decode([int(i)])!r:12} {p:.3f}")

# ---- 4) 自己写生成循环：每次只预测一个 token，接到末尾，再预测 ----
def generate(prompt, n=30, temperature=0.8):
    ids = torch.tensor([tok.encode(prompt)])
    for _ in range(n):
        with torch.no_grad():
            logits = model(ids).logits[0, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt[None]], dim=1)
    return tok.decode(ids[0])

torch.manual_seed(0)
print("\n生成：", generate("Once upon a time, there was a little robot who"))

# ---- 5) 注意力：句子里的 "it" 在看谁？ ----
s = "The animal didn't cross the street because it was too tired"
ids = torch.tensor([tok.encode(s)])
words = [tok.decode([i]) for i in ids[0]]
pos = words.index(" it")
with torch.no_grad():
    att = model(ids, output_attentions=True).attentions   # 12 层，每层 (1, 12 头, T, T)
# 注意：模型内部 12 个头是“拼接”后交给 Wo 的，这里只是把注意力权重拿出来看
a_i, s_i = words.index(" animal"), words.index(" street")
print(f"\n“it” 在每一层里，最关注 animal 的那个头：")
for layer, a in enumerate(att):
    w = a[0, :, pos]                                       # (12 个头, T)：每个头里 it 对每个词的权重
    h = int(w[:, a_i].argmax())
    print(f"   第 {layer:2d} 层 头 {h:2d}: animal {w[h, a_i]:.2f}   street {w[h, s_i]:.2f}   it 自己 {w[h, pos]:.2f}")
