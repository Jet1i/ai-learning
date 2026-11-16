"""第 9 周 L4:GRPO —— DeepSeek-R1 用的强化学习:不要 critic、不要奖励模型,一个问题采一组回答,组内比高低
运行:python week9/grpo.py   (需要 L0 sft.py 存好的 week9/sft_model;Mac CPU 约 3~5 分钟)

场景:接着 L0 的 SFT 客服。它有两个毛病:
  ① 有些问法答错类别(取消、改地址答成退款)
  ② 答完不停,接着啰嗦 "After that, please request a refund..."
我们手里有每个问题的标准答案,所以能用**规则**打分 —— 和 R1 用规则检查数学答案一样。

GRPO 每一轮就 5 步:
  ① 取几个问题,每个问题用当前模型采样 G 个回答(温度 1,要有探索)
  ② 规则打分(见 reward 函数)
  ③ 组内优势:A_i = (r_i − 这组的平均分) / 这组的标准差。同一个回答里每个 token 都用这个 A
     —— 用「同组其他回答的平均分」代替 critic 的 V,所以不需要价值模型
  ④ loss = −A × logπ(token) + β × KL(π‖π_ref)(逐 token,KL 用 GRPO 论文的估计式)
  ⑤ 反向传播只改 π;π_ref 是冻结的 SFT 模型(KL 绳子)

REWARD_MODE:
  "acc"     朴素版:只看答没答对类别(+1)
  "acc+fmt" 再加格式分:答完正好停住、不多说一个字(+0.5)
"""
import os, random, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.environ.get("SFT_DIR", os.path.join(HERE, "sft_model"))
REWARD_MODE = "acc"    # 先跑朴素版;练习里改成 "acc+fmt"
G = 8                  # 每个问题采几个回答(一组)
PROMPTS_PER_ITER = 4   # 每轮几个问题 → 每轮 4 × 8 = 32 个回答
ITERS = 20
LR = 1e-5
BETA = 0.04            # KL 系数(GRPO 论文用 0.04)
MAX_NEW = 40
torch.manual_seed(0); random.seed(0)

if not os.path.isdir(SFT_DIR):
    raise SystemExit(f"找不到 {SFT_DIR}\n先运行 python week9/sft.py,它最后会把 SFT 模型存到这个文件夹。")

tok = AutoTokenizer.from_pretrained(SFT_DIR)
policy = AutoModelForCausalLM.from_pretrained(SFT_DIR)   # π:要训练的
ref = AutoModelForCausalLM.from_pretrained(SFT_DIR)      # π_ref:冻结
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval(); ref.eval()   # 关掉 dropout
USER, ASSIST, END = (tok.convert_tokens_to_ids(t) for t in ["<|user|>", "<|assistant|>", "<|end|>"])
PAD = tok.eos_token_id

ANSWERS = {   # 和 sft.py 一样 = 每类问题的「标准答案」
    "refund":   "Go to My Orders, pick the order and tap Request Refund. The money returns in 3-5 business days.",
    "shipping": "Standard shipping takes 2-4 days. You can track it in My Orders.",
    "password": "Tap Forgot Password on the login page and follow the email we send you.",
    "cancel":   "You can cancel in My Orders before the order ships. After that, please request a refund.",
    "address":  "Go to Settings, then Addresses, and edit or add a new address.",
    "contact":  "You can chat with us in the app 24/7, or email help@shopx.com.",
}
RL_PROMPTS = {   # 强化学习用的新问法(SFT 没见过)
    "refund": ["Can I get a refund for a broken item?", "How long until I get my refund?", "The item was damaged, I want my money.",
               "How do I return an order for money back?", "Refund for my last order?"],
    "shipping": ["How long is delivery to my city?", "My package is late, how long does it take?", "What is your delivery time?",
                 "How soon will it ship and arrive?", "When can I expect my order?"],
    "password": ["My password doesn't work.", "I need a new password.", "How can I recover my account password?",
                 "Login says wrong password, help.", "Reset password please."],
    "cancel": ["I changed my mind, can I cancel?", "Please cancel the order I just made.", "Can I stop my order before it ships?",
               "How to cancel a purchase?", "I ordered by mistake, cancel it."],
    "address": ["I need to change my delivery address.", "Can I ship to a different address?", "My address is wrong in my account.",
                "How do I update my home address?", "Add another shipping address?"],
    "contact": ["Can I speak to a human?", "What is your support email?", "How do I chat with an agent?",
                "Is customer support open now?", "I want to talk to customer service."],
}
TEST = {   # 测试:L0 留出的 6 个问法 + 6 个新问法,训练时都没出现过
    "refund": ["How do I ask for my money back?", "I'd like my money returned for this order."],
    "shipping": ["When does my parcel get here?", "How many days until delivery?"],
    "password": ["I don't remember my password.", "My login password is not working."],
    "cancel": ["Can I undo my order?", "I don't want this order anymore, cancel please."],
    "address": ["How can I update where my order goes?", "I need to fix the address on my account."],
    "contact": ["How do I get in touch with your team?", "Can I email your support team?"],
}
pool = [(k, q) for k, qs in RL_PROMPTS.items() for q in qs]

def prompt_ids(q):
    return [USER] + tok.encode(q) + [ASSIST]

def split(ids):
    """生成的 token → (文字, 有没有停住)"""
    ids = list(ids)
    if END in ids:
        return tok.decode(ids[:ids.index(END)]).strip(), True
    return tok.decode([i for i in ids if i != PAD]).strip(), False

# ---------------- ② 规则打分 ----------------
def reward(kind, ids):
    text, stopped = split(ids)
    acc = 1.0 if text.startswith(ANSWERS[kind][:30]) else 0.0           # 答对类别
    if REWARD_MODE == "acc":
        return acc
    fmt = 0.5 if (stopped and text == ANSWERS[kind]) else 0.0           # 答完正好停住
    return acc + fmt

@torch.no_grad()
def sample(q, n):
    p = prompt_ids(q)
    x = torch.tensor([p] * n)
    out = policy.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                          max_new_tokens=MAX_NEW, eos_token_id=END, pad_token_id=PAD)
    return p, out[:, len(p):]

def token_logp(model, p, ans):
    """每个回答 token 的 log 概率 (n, T)"""
    ids = torch.cat([torch.tensor([p] * len(ans)), ans], dim=1)
    logits = model(ids).logits[:, len(p) - 1:-1]
    return F.log_softmax(logits, -1).gather(-1, ans.unsqueeze(-1)).squeeze(-1)

def answer_mask(ans):
    """回答部分:从第一个 token 到 <|end|>(含);停住以后补的 pad 不算"""
    m = torch.zeros(ans.shape)
    for i, row in enumerate(ans.tolist()):
        L = row.index(END) + 1 if END in row else len(row)
        m[i, :L] = 1
    return m

@torch.no_grad()
def greedy(q):
    p = prompt_ids(q)
    x = torch.tensor([p])
    out = policy.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=MAX_NEW,
                          eos_token_id=END, pad_token_id=PAD)
    return split(out[0, len(p):].tolist())

def report(title):
    acc = stop = 0
    lines = []
    for k, qs in TEST.items():
        for q in qs:
            text, stopped = greedy(q)
            ok = text.startswith(ANSWERS[k][:30]); clean = stopped and text == ANSWERS[k]
            acc += ok; stop += clean
            lines.append(f"  {'✅' if ok else '❌'}{'🛑' if clean else '  '} {q}\n        → {text}")
    print(f"\n=== {title} ===")
    print("\n".join(lines))
    print(f"  答对类别 {acc}/12    答完正好停住 {stop}/12")
    return acc, stop

print(f"奖励模式 REWARD_MODE = {REWARD_MODE!r},每轮 {PROMPTS_PER_ITER} 个问题 × G={G} 个回答,共 {ITERS} 轮")
before = report("训练前(SFT 模型)贪心回答   ✅=答对类别  🛑=答完正好停住")

# ---------------- 看一组:采样 → 打分 → 组内优势 ----------------
k0, q0 = "cancel", "Can I stop my order before it ships?"
p0, a0 = sample(q0, G)
r0 = torch.tensor([reward(k0, a) for a in a0.tolist()])
A0 = (r0 - r0.mean()) / (r0.std() + 1e-4)
print(f"\n=== ①②③ 一组示例:「{q0}」采 {G} 个回答 ===")
for a, r, adv in zip(a0.tolist(), r0.tolist(), A0.tolist()):
    print(f"  分 {r:.1f}  优势 {adv:+.2f}  {split(a)[0][:70]}")
print(f"  组平均 {r0.mean():.2f},标准差 {r0.std():.2f} → 高于平均的回答优势为正(推高),低于的为负(压低)")

# ---------------- 训练 ----------------
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
print(f"\n=== ④⑤ 开始 GRPO:lr {LR},β {BETA} ===")
print("  轮   平均分   分全一样的组   KL      用时")
t0 = time.time()
for it in range(1, ITERS + 1):
    batch = random.sample(pool, PROMPTS_PER_ITER)
    loss_total, rs, flat, kls = 0.0, [], 0, []
    opt.zero_grad()
    for kind, q in batch:
        p, ans = sample(q, G)                                              # ① 采一组
        r = torch.tensor([reward(kind, a) for a in ans.tolist()])          # ② 打分
        rs += r.tolist()
        if r.std() < 1e-6:                                                 # 全对或全错:优势全是 0,这组学不到东西
            flat += 1
        A = (r - r.mean()) / (r.std() + 1e-4)                              # ③ 组内优势
        mask = answer_mask(ans)
        lp = token_logp(policy, p, ans)
        with torch.no_grad():
            lp_ref = token_logp(ref, p, ans)
        kl = torch.exp(lp_ref - lp) - (lp_ref - lp) - 1                    # 每个 token 的 KL 估计(≥0)
        per_tok = -A.unsqueeze(1) * lp + BETA * kl                         # ④ 每个 token 的 loss
        loss = ((per_tok * mask).sum(1) / mask.sum(1)).mean()
        (loss / PROMPTS_PER_ITER).backward()
        kls.append(((kl * mask).sum() / mask.sum()).item())
    opt.step()                                                             # ⑤ 更新
    print(f"  {it:3d}   {sum(rs)/len(rs):.2f}     {flat}/{PROMPTS_PER_ITER}          {sum(kls)/len(kls):.3f}   {time.time()-t0:.0f}s")

after = report("训练后(GRPO 模型)贪心回答")
print(f"\n答对类别 {before[0]}/12 → {after[0]}/12      答完正好停住 {before[1]}/12 → {after[1]}/12")
