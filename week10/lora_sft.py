"""第 10 周 L2:用 LoRA 给 ShopX 客服模型"打补丁"
运行:pip install peft      (第一次)
      python week10/lora_sft.py                 默认 MODE="lora"
      MODE=full python week10/lora_sft.py       对照组:全参微调(和第 9 周 sft.py 一样)

起点:第 9 周 sft.py 存下来的 week9/sft_model(已经会 6 类客服问题,会用 <|user|><|assistant|><|end|>)
新需求:ShopX 上线了 2 个新业务 —— 保修(warranty)和礼品卡(gift card),客服要会答。
我们只用 2 类 × 6 条 = 12 条新数据去训练,然后看三件事:
  ① 新问题学会了没有(没见过的问法)
  ② 旧的 6 类问题忘了没有
  ③ 训练了多少参数、存下来多大

⚠️ 这个版本故意留了一个问题:第一次跑 LoRA 你会发现它"几乎没学会"。
   先别看答案,对照 L1 想想是哪个设置不对(提示:这个值是从 sft.py 直接抄来的)。
   修好之后还有第二关:看看"旧问题"那一栏。到时候再用 REPLAY。
"""
import os, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODE = os.environ.get("MODE", "lora")    # "lora" 或 "full"
EPOCHS = 10
BATCH = 4
LR = 5e-5          # 从 sft.py 抄来的
R = 8              # LoRA 的秩
ALPHA = 16         # 缩放 = ALPHA / R = 2
REPLAY = 0         # 第二关再用:每类旧问题混进几条一起训练(0 = 只用新数据)
torch.manual_seed(0); random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.join(HERE, "..", "week9", "sft_model")

# ---------------- 新业务数据:每类 6 种问法训练,2 种留着测试 ----------------
NEW_ANSWERS = {
    "warranty": "All electronics have a 1-year warranty. Open My Orders and tap Warranty Claim.",
    "giftcard": "Enter the gift card code at checkout under Payment. Gift cards never expire.",
}
NEW_Q = {
    "warranty": ["Is there a warranty?", "My headphones broke, is it covered?", "How long is the warranty?",
                 "How do I make a warranty claim?", "Does my laptop have a guarantee?", "Warranty info please.",
                 "My phone stopped working after 3 months, can you fix it?", "Do you cover broken electronics?"],
    "giftcard": ["How do I use a gift card?", "Where do I enter my gift card code?", "Do gift cards expire?",
                 "Can I pay with a gift card?", "I got a gift card, how to redeem it?", "Gift card help.",
                 "How can I spend my gift card?", "Is my gift card still valid next year?"],
}
# 第 9 周的 6 类旧问题(用 sft.py 的测试问法:训练时没见过)
OLD_TEST = [
    ("How do I ask for my money back?", "Go to My Orders, pick the order and tap Request Refund."),
    ("When does my parcel get here?", "Standard shipping takes 2-4 days."),
    ("I don't remember my password.", "Tap Forgot Password on the login page"),
    ("Can I undo my order?", "You can cancel in My Orders before the order ships."),
    ("How can I update where my order goes?", "Go to Settings, then Addresses"),
    ("How do I get in touch with your team?", "You can chat with us in the app 24/7"),
]
# 第 9 周 sft.py 里的旧训练数据(每类前 2 种问法),REPLAY>0 时混进来
OLD_TRAIN = {
    "Go to My Orders, pick the order and tap Request Refund. The money returns in 3-5 business days.":
        ["How do I get a refund?", "How can I get my money back?"],
    "Standard shipping takes 2-4 days. You can track it in My Orders.":
        ["How long does shipping take?", "When will my package arrive?"],
    "Tap Forgot Password on the login page and follow the email we send you.":
        ["I forgot my password.", "How do I reset my password?"],
    "You can cancel in My Orders before the order ships. After that, please request a refund.":
        ["How do I cancel my order?", "Can I cancel an order?"],
    "Go to Settings, then Addresses, and edit or add a new address.":
        ["How do I change my address?", "I moved, how to update my address?"],
    "You can chat with us in the app 24/7, or email help@shopx.com.":
        ["How do I contact support?", "I need to talk to someone."],
}
train = [(q, NEW_ANSWERS[k]) for k in NEW_Q for q in NEW_Q[k][:6]]          # 12 条
train += [(q, a) for a, qs in OLD_TRAIN.items() for q in qs[:REPLAY]]
new_test = [(q, NEW_ANSWERS[k]) for k in NEW_Q for q in NEW_Q[k][6:]]       # 4 条

# ---------------- 加载 SFT 模型 ----------------
tok = AutoTokenizer.from_pretrained(SFT_DIR)
model = AutoModelForCausalLM.from_pretrained(SFT_DIR)
USER, ASSIST, END = (tok.convert_tokens_to_ids(t) for t in ["<|user|>", "<|assistant|>", "<|end|>"])
total = sum(p.numel() for p in model.parameters())

if MODE == "lora":
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                     target_modules=["c_attn"],     # 每层注意力里算 q,k,v 的那个 768→2304 Linear
                     fan_in_fan_out=True)           # GPT-2 的 Linear 是转置存的(Conv1D),要告诉 peft
    model = get_peft_model(model, cfg)              # 冻住全部原参数,只在 c_attn 旁边挂 A、B
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"训练数据 {len(train)} 条(新 12 + 旧 {len(train)-12})")
print(f"MODE={MODE}  总参数 {total:,}  训练参数 {trainable:,}({trainable/total:.2%})  LR={LR}")


@torch.no_grad()
def answer(q, n=30):
    model.eval()
    ids = torch.tensor([[USER] + tok.encode(q) + [ASSIST]])
    out = []
    for _ in range(n):
        nxt = model(ids).logits[0, -1].argmax().item()
        if nxt == END:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out).strip()


def score(pairs, show=False):
    ok = 0
    for q, a in pairs:
        out = answer(q)
        hit = out.startswith(a.split(".")[0])            # 第一句对上就算对
        ok += hit
        if show:
            print(f"  {'✅' if hit else '❌'} {q}\n     → {out}")
    return ok


def batchify(pairs):
    rows = []
    for q, a in pairs:
        prompt = [USER] + tok.encode(q) + [ASSIST]
        ans = tok.encode(" " + a) + [END]
        rows.append((prompt + ans, [-100] * len(prompt) + ans))
    L = max(len(r[0]) for r in rows)
    x = torch.tensor([r[0] + [tok.eos_token_id] * (L - len(r[0])) for r in rows])
    y = torch.tensor([r[1] + [-100] * (L - len(r[1])) for r in rows])
    m = torch.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rows])
    return x, y, m


print(f"\n=== 训练前 ===  新业务 {score(new_test)}/4   旧问题 {score(OLD_TEST)}/6")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); random.shuffle(train); losses = []
    for i in range(0, len(train), BATCH):
        x, y, m = batchify(train[i:i + BATCH])
        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep:2}  loss {sum(losses)/len(losses):.2f}   {time.time()-t0:.0f}s")

print("\n=== 训练后:新业务(没见过的问法)===")
n_new = score(new_test, show=True)
print("\n=== 训练后:旧问题(忘了没有?)===")
n_old = score(OLD_TEST, show=True)
print(f"\n汇总 MODE={MODE}:新业务 {n_new}/4,旧问题 {n_old}/6,训练参数 {trainable:,}")

# ---------------- 存下来:LoRA 只存 adapter ----------------
save = os.path.join(HERE, f"shopx_{MODE}")
model.save_pretrained(save)
size = sum(os.path.getsize(os.path.join(save, f)) for f in os.listdir(save))
print(f"已保存到 {save}:{size/1e6:.1f} MB")

if MODE == "lora":
    print("\n=== 同一个底座,拔掉补丁 = 回到原来的 SFT 模型 ===")
    with model.disable_adapter():
        print(f"  拔掉 adapter:{NEW_Q['warranty'][6]} → {answer(NEW_Q['warranty'][6])}")
    print(f"  插上 adapter:{NEW_Q['warranty'][6]} → {answer(NEW_Q['warranty'][6])}")
