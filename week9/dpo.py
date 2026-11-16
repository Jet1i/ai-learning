"""第 9 周 L3:DPO —— 不要奖励模型、不要 critic、不用边训边生成,直接拿偏好对训练
运行:先跑 python week9/sft.py(最后会把 SFT 模型存到 week9/sft_model),再 python week9/dpo.py
     Mac CPU 约 2~4 分钟(真 GPT-2 实测:纯 DPO 拒答 0/6;加 SFT loss 后拒答 5/6,网店题保持 4/6)

场景:L0 的 SFT 客服什么问题都往网店答案上套(问"法国首都是哪"它也答退款流程)。
     现在给它偏好数据,教它两件事:
       无关问题 → 礼貌拒答(chosen) 好于 SFT 模型自己乱套的网店答案(rejected)
       网店问题 → 正确答案(chosen) 好于 答成别的类别(rejected),也好于 拒答(rejected)—— 该答的别拒

真实 DPO 就这 5 步:
  ① 偏好数据:(问题, chosen, rejected) 三元组 —— 和 L1 训练奖励模型用的数据长得一样
  ② 两个模型:π(要训练的,从 SFT 开始)+ π_ref(SFT 的冻结副本)。没有奖励模型,没有 critic
  ③ 每对数据算 4 个数:π 和 π_ref 各算一次 chosen、rejected 的 log 概率
     (一条回答的 log 概率 = 回答部分每个 token 的 log 概率加起来;问题部分不算,和 SFT 的 -100 一个意思)
  ④ 隐含奖励 r = β × (logπ − logπ_ref);loss = −ln σ(r_chosen − r_rejected) —— 就是 L1 奖励模型那条 loss
  ⑤ 反向传播只改 π;π_ref 一直不动(它就是 PPO 里那根 KL 绳子,藏在 r 的减号里)

实测踩到的坑(真实项目里也常见):纯 DPO 的 loss 能降到 0.002,但拒答还是 0/6。
  DPO 只管"chosen 比 rejected 高多少",不管 chosen 能不能高到被生成出来;
  压低 rejected 让出来的概率流到了第三种回答(新的胡编)。
  修法:loss 再加 NLL_ALPHA × chosen 的 SFT loss(Llama 3 做 DPO 时也加了这一项)。
"""
import os, random, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.environ.get("SFT_DIR", os.path.join(HERE, "sft_model"))
BETA = 0.1             # 绳子松紧;练习里会改
LR = 2e-5
EPOCHS = 8
BATCH = 4              # 每批 4 对 = 8 条回答
ONLY_OFFTOPIC = False  # 练习 3:改成 True,只给"该拒答"的数据
NLL_ALPHA = 1.0        # 再给 chosen 加一项 SFT loss(Llama 3 也这么做);0 = 纯 DPO(你第一次跑的版本)
torch.manual_seed(0); random.seed(0)

if not os.path.isdir(SFT_DIR):
    raise SystemExit(f"找不到 {SFT_DIR}\n先运行 python week9/sft.py,它最后会把 SFT 模型存到这个文件夹。")

# ---------------- ② 两个模型,都从同一个 SFT 模型加载 ----------------
tok = AutoTokenizer.from_pretrained(SFT_DIR)
policy = AutoModelForCausalLM.from_pretrained(SFT_DIR)   # π:要训练的
ref = AutoModelForCausalLM.from_pretrained(SFT_DIR)      # π_ref:冻结,不训练
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval(); ref.eval()   # 关掉 dropout:这样训练前 π 和 π_ref 算出来完全一样(TRL 默认也这么做)
USER, ASSIST, END = (tok.convert_tokens_to_ids(t) for t in ["<|user|>", "<|assistant|>", "<|end|>"])

# ---------------- ① 偏好数据 ----------------
ANSWERS = {   # 和 sft.py 里一模一样
    "refund":   "Go to My Orders, pick the order and tap Request Refund. The money returns in 3-5 business days.",
    "shipping": "Standard shipping takes 2-4 days. You can track it in My Orders.",
    "password": "Tap Forgot Password on the login page and follow the email we send you.",
    "cancel":   "You can cancel in My Orders before the order ships. After that, please request a refund.",
    "address":  "Go to Settings, then Addresses, and edit or add a new address.",
    "contact":  "You can chat with us in the app 24/7, or email help@shopx.com.",
}
KINDS = list(ANSWERS)
REFUSE = "Sorry, I can only help with ShopX orders, shipping and accounts."

ON_TRAIN = {   # 网店问题(SFT 训练时见过的问法里各取 2 个)
    "refund": ["How do I get a refund?", "What is the refund process?"],
    "shipping": ["How long does shipping take?", "When will my package arrive?"],
    "password": ["I forgot my password.", "How do I reset my password?"],
    "cancel": ["How do I cancel my order?", "Can I cancel an order?"],
    "address": ["How do I change my address?", "Where do I add a new address?"],
    "contact": ["How do I contact support?", "Is there live chat?"],
}
ON_TEST = {    # 网店问题,SFT 和 DPO 都没见过的问法
    "refund": "How do I ask for my money back?", "shipping": "When does my parcel get here?",
    "password": "I don't remember my password.", "cancel": "Can I undo my order?",
    "address": "How can I update where my order goes?", "contact": "How do I get in touch with your team?",
}
OFF_TRAIN = ["What's the weather like today?", "Tell me a joke.", "Who won the last World Cup?",
             "How do I cook rice?", "What is 15 times 7?", "Write a short poem about cats.",
             "Who is the richest person in the world?", "Can you recommend a good movie?",
             "How tall is Mount Everest?", "Can you help me with my math homework?",
             "What time is it in Tokyo?", "How do I learn to play guitar?"]
OFF_TEST = ["What is the capital of France?", "How do I fix a flat bike tire?", "Who wrote Romeo and Juliet?",
            "What's a good recipe for pizza?", "Explain how rainbows form.", "How many planets are in the solar system?"]

def prompt_ids(q):
    return [USER] + tok.encode(q) + [ASSIST]

def ans_ids(text):
    return tok.encode(" " + text) + [END]

@torch.no_grad()
def generate(model, q, n=40):
    """贪心生成,遇到 <|end|> 停。返回 (回答的 token ids 含 END, 文字)"""
    ids = torch.tensor([prompt_ids(q)])
    out = []
    for _ in range(n):
        nxt = model(ids).logits[0, -1].argmax().item()
        out.append(nxt)
        if nxt == END:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    if out[-1] != END:
        out.append(END)
    return out, tok.decode(out[:-1]).strip()

def make_pairs(on, off):
    pairs = []
    for k, qs in on.items():
        for q in ([qs] if isinstance(qs, str) else qs):
            wrong = KINDS[(KINDS.index(k) + 1) % len(KINDS)]            # 答成下一个类别
            pairs.append(dict(q=q, kind="网店", chosen=ans_ids(ANSWERS[k]), rejected=ans_ids(ANSWERS[wrong])))
            pairs.append(dict(q=q, kind="网店", chosen=ans_ids(ANSWERS[k]), rejected=ans_ids(REFUSE)))  # 该答的别拒
    for q in off:
        rej, text = generate(ref, q)                                     # rejected = SFT 模型自己的回答
        if not text or "only help" in text:
            rej = ans_ids(random.choice(list(ANSWERS.values())))
        pairs.append(dict(q=q, kind="无关", chosen=ans_ids(REFUSE), rejected=rej))
    return pairs

print("准备偏好数据(无关问题的 rejected 用 SFT 模型自己的回答,和真实做法一样)...")
train = make_pairs(ON_TRAIN, OFF_TRAIN)
if ONLY_OFFTOPIC:
    train = [d for d in train if d["kind"] == "无关"]
test = make_pairs(ON_TEST, OFF_TEST)
print(f"训练 {len(train)} 对,测试 {len(test)} 对(测试的问题训练时没出现过)")

ex = next(d for d in train if d["kind"] == "无关")
print(f"\n=== ① 一条偏好数据 ===\n  问题    : {ex['q']}\n  chosen  : {tok.decode(ex['chosen'][:-1]).strip()}"
      f"\n  rejected: {tok.decode(ex['rejected'][:-1]).strip()}   ← SFT 模型自己说的")

# ---------------- ③ 一条回答的 log 概率 = 回答部分 token 的 log 概率之和 ----------------
def seq_logp(model, items):
    """items: [(问题, 回答 ids)] → 每条回答的总 log 概率 (B,)"""
    rows = [(prompt_ids(q), a) for q, a in items]
    L = max(len(p) + len(a) for p, a in rows)
    ids = torch.full((len(rows), L), tok.eos_token_id)
    attn = torch.zeros(len(rows), L, dtype=torch.long)
    is_ans = torch.zeros(len(rows), L)
    for i, (p, a) in enumerate(rows):
        ids[i, :len(p) + len(a)] = torch.tensor(p + a)
        attn[i, :len(p) + len(a)] = 1
        is_ans[i, len(p):len(p) + len(a)] = 1                 # 只有回答部分算
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]   # 第 t 位预测第 t+1 个 token
    logp = F.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * is_ans[:, 1:]).sum(-1)

def both(model, ds):
    """一次算 chosen 和 rejected:返回 (logp_chosen, logp_rejected)"""
    lp = seq_logp(model, [(d["q"], d["chosen"]) for d in ds] + [(d["q"], d["rejected"]) for d in ds])
    return lp[:len(ds)], lp[len(ds):]

# π_ref 冻结不变,所以它的两个 log 概率可以提前一次算好存起来
with torch.no_grad():
    for ds in (train, test):
        c, r = both(ref, ds)
        for d, a, b in zip(ds, c.tolist(), r.tolist()):
            d["ref_c"], d["ref_r"] = a, b

# ---------------- ④ 隐含奖励 + DPO loss ----------------
def dpo(ds, pol_c, pol_r):
    ref_c = torch.tensor([d["ref_c"] for d in ds]); ref_r = torch.tensor([d["ref_r"] for d in ds])
    r_c = BETA * (pol_c - ref_c)                 # chosen 的隐含奖励
    r_r = BETA * (pol_r - ref_r)                 # rejected 的隐含奖励
    loss = -F.logsigmoid(r_c - r_r).mean()       # = −ln σ(r_c − r_r),L1 那条
    return loss, r_c, r_r

@torch.no_grad()
def evaluate(ds):
    c, r = both(policy, ds)
    loss, r_c, r_r = dpo(ds, c, r)
    return loss.item(), (r_c > r_r).float().mean().item(), r_c, r_r, c, r

def show_pair(d, title):
    with torch.no_grad():
        c, r = both(policy, [d])
    loss, r_c, r_r = dpo([d], c, r)
    print(f"\n=== {title}:「{d['q']}」===")
    print(f"             logπ(现在)  logπ_ref(SFT)   隐含奖励 β×(差)")
    print(f"  chosen   {c.item():10.2f}  {d['ref_c']:12.2f}   {r_c.item():+8.2f}")
    print(f"  rejected {r.item():10.2f}  {d['ref_r']:12.2f}   {r_r.item():+8.2f}")
    print(f"  差距 r_c − r_r = {(r_c - r_r).item():+.2f}   loss = −ln σ(差距) = {loss.item():.3f}")

focus = next(d for d in test if d["q"] == "What is the capital of France?")
show_pair(focus, "③④ 训练前,一对测试数据")
print("  训练前 π 和 π_ref 一模一样 → 隐含奖励全是 0 → loss = ln 2 = 0.693(和 PPO 第 1 轮 KL = 0 一个道理)")

def answer_report(model):
    on_ok, off_ok, lines = 0, 0, []
    for k, q in ON_TEST.items():
        _, t = generate(model, q); ok = t.startswith(ANSWERS[k][:25]); on_ok += ok
        lines.append(f"  {'✅' if ok else '❌'} [网店] {q}\n       → {t}")
    for q in OFF_TEST:
        _, t = generate(model, q); ok = "only help" in t; off_ok += ok
        lines.append(f"  {'✅' if ok else '❌'} [无关] {q}\n       → {t}")
    return on_ok, off_ok, lines

before = answer_report(ref)

# ---------------- ⑤ 训练:只改 π ----------------
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
print(f"\n=== ⑤ 开始 DPO:{EPOCHS} 个 epoch,每批 {BATCH} 对,β {BETA},lr {LR},NLL_ALPHA {NLL_ALPHA} ===")
print("  epoch  训练loss  训练准确率  测试准确率  chosen平均奖励  rejected平均奖励")
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    random.shuffle(train)
    for i in range(0, len(train), BATCH):
        ds = train[i:i + BATCH]
        c, r = both(policy, ds)                  # π 的 2 次前向(带梯度)
        loss, _, _ = dpo(ds, c, r)
        if NLL_ALPHA > 0:                        # chosen 的 SFT loss:每个 token 的 −log p 取平均
            n_c = torch.tensor([len(d["chosen"]) for d in ds], dtype=torch.float)
            loss = loss + NLL_ALPHA * (-(c / n_c).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    tl, ta, rc, rr, _, _ = evaluate(train)
    _, va, _, _, _, _ = evaluate(test)
    print(f"  {ep:5d}  {tl:8.3f}  {ta:10.0%}  {va:10.0%}  {rc.mean().item():+14.2f}  {rr.mean().item():+16.2f}   {time.time()-t0:.0f}s")

show_pair(focus, "训练后,同一对测试数据")

# chosen 和 rejected 的 log 概率各自变了多少(练习 1)
_, _, _, _, c, r = evaluate(train)
dc = (c - torch.tensor([d["ref_c"] for d in train])).mean().item()
dr = (r - torch.tensor([d["ref_r"] for d in train])).mean().item()
print(f"\n训练数据平均:chosen 的 logπ 变了 {dc:+.2f},rejected 的 logπ 变了 {dr:+.2f}")

after = answer_report(policy)
print("\n=== 训练前(SFT 模型)的回答 ===")
print("\n".join(before[2]))
print("\n=== 训练后(DPO 模型)的回答 ===")
print("\n".join(after[2]))
print(f"\n网店问题答对:{before[0]}/6 → {after[0]}/6      无关问题拒答:{before[1]}/6 → {after[1]}/6")
