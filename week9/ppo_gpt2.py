"""第 9 周 L2.5 实验:真实 LLM 上手写 PPO(GPT-2 学会写好评)
运行:python week9/ppo_gpt2.py        (Mac CPU 上约 5~15 分钟,第一次会下载约 1 GB 模型)

和真实 RLHF 一样的 4 个模型:
  策略 π      = lvwerra/gpt2-imdb      (在影评上训练过的 GPT-2,当作"SFT 模型"),要训练
  参照 π_ref  = 同一个模型复制一份,冻住
  奖励模型 RM = lvwerra/distilbert-imdb (用人工标注的影评训练的情感分类器,"正面"分数当奖励),冻住
  价值模型 V  = GPT-2 主干 + 一个输出 1 个数的 wx+b,每个 token 位置给一个 V,要训练
唯一的简化:RM 是"正面/负面"分类器,不是用 A≻B 偏好训练的;其他步骤和 InstructGPT 的 PPO 一样。
目标:给一个影评开头,让 GPT-2 写出更正面的评论,同时不要离原模型太远。
"""
import os, random, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像;想用官方源就先 export HF_ENDPOINT
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModel,
                          AutoModelForSequenceClassification)

POLICY_NAME = os.environ.get("POLICY_NAME", "lvwerra/gpt2-imdb")
RM_NAME = os.environ.get("RM_NAME", "lvwerra/distilbert-imdb")
DEVICE = os.environ.get("DEVICE", "cpu")

ITERS = int(os.environ.get("ITERS", 50))   # 一共几轮(每轮:生成 → 打分 → 更新)
BATCH = 16          # 每轮几条提示词(真实系统几百条)
GEN_LEN = 20        # 每条生成几个 token(真实系统几百个)
K_EPOCHS = 4        # 同一批数据更新几轮
MINI = 4            # 每个小批几条(16 条分成 4 个小批,每轮更新 4×4 = 16 次)
LR = 1.4e-5
BETA = 0.1          # 绳子力度
CLIP = 0.2          # ratio 限制在 [0.8, 1.2]
GAMMA, LAM = 1.0, 0.95

PROMPTS = ["This movie", "The film", "I watched this", "The acting", "The plot of this movie",
           "Honestly, this film", "The ending", "My friends and I saw this movie", "The director",
           "This was", "I went to see this", "The story"]

torch.manual_seed(0); random.seed(0)

# ---------------- 准备 4 个模型 ----------------
print("加载模型(第一次会下载)…")
tok = AutoTokenizer.from_pretrained(POLICY_NAME)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
policy = AutoModelForCausalLM.from_pretrained(POLICY_NAME).to(DEVICE)
ref = AutoModelForCausalLM.from_pretrained(POLICY_NAME).to(DEVICE)
for p in ref.parameters():
    p.requires_grad_(False)

class ValueModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.trunk = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.trunk.config.n_embd, 1)       # 输出 1 个数的 wx+b
    def forward(self, ids, mask, pos):
        h = self.trunk(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
        return self.head(h).squeeze(-1)                          # (B, L):每个位置一个 V
value = ValueModel(POLICY_NAME).to(DEVICE)

rm_tok = AutoTokenizer.from_pretrained(RM_NAME)
rm = AutoModelForSequenceClassification.from_pretrained(RM_NAME).to(DEVICE)
for p in rm.parameters():
    p.requires_grad_(False)
labels = {v.upper(): int(k) for k, v in rm.config.id2label.items()}
POS = labels.get("POSITIVE", 1)

for m in (policy, ref, value, rm):
    m.eval()        # 关掉 dropout,否则同样的权重算两次概率也不一样,ratio 就不准了
opt_pi = torch.optim.Adam(policy.parameters(), lr=LR)
opt_v = torch.optim.Adam(value.parameters(), lr=LR)

def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
print(f"策略 {n_params(policy):.0f}M · 参照 {n_params(ref):.0f}M · 奖励 {n_params(rm):.0f}M · 价值 {n_params(value):.0f}M 参数")

# ---------------- 工具函数 ----------------
def positions(mask):
    return (mask.cumsum(-1) - 1).clamp(min=0)     # 左边补齐时,位置编号从真正的第一个字算起

def token_logprobs(model, ids, mask, P):
    """每个生成 token 的 log 概率,形状 (B, T)"""
    logits = model(input_ids=ids, attention_mask=mask, position_ids=positions(mask)).logits
    logp = torch.log_softmax(logits[:, P - 1:-1].float(), dim=-1)    # 第 P-1 个位置预测第 1 个回答 token
    return logp.gather(-1, ids[:, P:].unsqueeze(-1)).squeeze(-1)

def token_values(ids, mask, P):
    return value(ids, mask, positions(mask))[:, P - 1:-1]           # V_t = 写第 t 个 token 之前的预期

@torch.no_grad()
def reward_score(texts):
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    return rm(**enc).logits[:, POS]                                  # "正面"的分数

@torch.no_grad()
def generate(model, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    out = model.generate(**enc, do_sample=True, top_k=0, top_p=1.0, max_new_tokens=GEN_LEN,
                         min_new_tokens=GEN_LEN, pad_token_id=tok.eos_token_id)
    mask = torch.cat([enc.attention_mask, torch.ones_like(out[:, enc.input_ids.shape[1]:])], dim=1)
    return out, mask, enc.input_ids.shape[1]

def decode(ids, P):
    return [tok.decode(x[P:], skip_special_tokens=True) for x in ids]

# ---------------- 训练前看一眼 ----------------
test_prompts = ["This movie", "The acting", "The ending"]
torch.manual_seed(1)
ids0, m0, P0 = generate(ref, test_prompts)
before = decode(ids0, P0)
print("\n=== 训练前(SFT 模型)写的 ===")
for p_, t in zip(test_prompts, before):
    print(f"  {p_}{t}")

# ---------------- PPO 主循环 ----------------
print(f"\n=== 开始 PPO:{ITERS} 轮,每轮 {BATCH} 条 × {GEN_LEN} 个 token ===")
print(" 轮   奖励(正面分)  KL(离 SFT 多远)  clip 比例   用时")
t0 = time.time()
for it in range(1, ITERS + 1):
    prompts = random.choices(PROMPTS, k=BATCH)

    # ① 生成
    ids, mask, P = generate(policy, prompts)
    texts = [p_ + r for p_, r in zip(prompts, decode(ids, P))]

    # ② 四次前向(不算梯度)
    with torch.no_grad():
        logp_old = token_logprobs(policy, ids, mask, P)     # (B, T)
        logp_ref = token_logprobs(ref, ids, mask, P)        # (B, T)
        values = token_values(ids, mask, P)                 # (B, T)
        R = reward_score(texts)                             # (B,)

    # ③ 每个 token 的奖励:绳子按 token 扣,最后一个 token 加 RM 分
    kl = logp_old - logp_ref
    rewards = -BETA * kl
    rewards[:, -1] += R

    # ④ GAE 分功劳
    T = rewards.shape[1]
    adv = torch.zeros_like(rewards)
    last = torch.zeros(BATCH, device=DEVICE)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t < T - 1 else torch.zeros(BATCH, device=DEVICE)
        delta = rewards[:, t] + GAMMA * next_v - values[:, t]     # 这个 token 让预期涨了多少
        last = delta + GAMMA * LAM * last
        adv[:, t] = last
    returns = adv + values                                        # 价值模型的学习目标 G_t
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)                 # 标准化

    if it == 1:
        print(f"     (第 1 轮的数据形状:logp_old {tuple(logp_old.shape)} · logp_ref {tuple(logp_ref.shape)}"
              f" · V {tuple(values.shape)} · R {tuple(R.shape)} · 优势 A {tuple(adv.shape)})")

    # ⑤ 同一批数据更新 K 轮,每轮分小批
    clipped = []
    for _ in range(K_EPOCHS):
        for idx in torch.randperm(BATCH).split(MINI):
            new_lp = token_logprobs(policy, ids[idx], mask[idx], P)
            ratio = torch.exp(new_lp - logp_old[idx])
            A = adv[idx]
            pg_loss = -torch.min(ratio * A, ratio.clamp(1 - CLIP, 1 + CLIP) * A).mean()
            v_new = token_values(ids[idx], mask[idx], P)
            v_loss = 0.5 * ((v_new - returns[idx]) ** 2).mean()
            opt_pi.zero_grad(); pg_loss.backward(); opt_pi.step()
            opt_v.zero_grad(); v_loss.backward(); opt_v.step()
            clipped.append(((ratio - 1).abs() > CLIP).float().mean().item())

    # ⑥ 这批数据扔掉,下一轮重新生成
    print(f"{it:3d}   {R.mean().item():+8.2f}     {kl.sum(1).mean().item():8.2f}"
          f"        {sum(clipped) / len(clipped):6.1%}   {time.time() - t0:5.0f}s")

# ---------------- 训练后对比 ----------------
torch.manual_seed(1)
ids1, m1, P1 = generate(policy, test_prompts)
after = decode(ids1, P1)
print("\n=== 同样的开头,训练前 vs 训练后 ===")
sb = reward_score([p_ + t for p_, t in zip(test_prompts, before)])
sa = reward_score([p_ + t for p_, t in zip(test_prompts, after)])
for p_, b, a, x, y in zip(test_prompts, before, after, sb, sa):
    print(f"  前[{x:+.1f}] {p_}{b}")
    print(f"  后[{y:+.1f}] {p_}{a}\n")
