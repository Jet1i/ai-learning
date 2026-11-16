"""第 9 周 L0:SFT(监督微调)—— 把只会续写的 GPT-2 教成"客服助手"
运行:python week9/sft.py      (用第 3 周已经下载好的 gpt2,124M;Mac CPU 约 1~3 分钟)

对应完整流程图里 SFT 的 3 步:
  ① 示范数据 + 对话模板(新加 3 个特殊 token)
  ② 只给 assistant 部分算 loss(其余 label = -100)
  ③ 训练循环 → 得到 SFT 模型,对比训练前后
GPT-2 只在英文上训练过,所以数据用英文;一家虚构网店 ShopX 的客服问答。
"""
import os, random, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
MASK_USER = True      # 练习 2:改成 False,让问题部分也算 loss
EPOCHS = 10            # 第一次跑 5 轮,回答停不下来(<|end|> 还没学会),改成 10
BATCH = 4
LR = 5e-5
torch.manual_seed(0); random.seed(0)

# ---------------- ① 示范数据:6 类问题,每类 6 种问法训练 + 1 种问法留着测试 ----------------
ANSWERS = {
    "refund":   "Go to My Orders, pick the order and tap Request Refund. The money returns in 3-5 business days.",
    "shipping": "Standard shipping takes 2-4 days. You can track it in My Orders.",
    "password": "Tap Forgot Password on the login page and follow the email we send you.",
    "cancel":   "You can cancel in My Orders before the order ships. After that, please request a refund.",
    "address":  "Go to Settings, then Addresses, and edit or add a new address.",
    "contact":  "You can chat with us in the app 24/7, or email help@shopx.com.",
}
QUESTIONS = {
    "refund":   ["How do I get a refund?", "How can I get my money back?", "I want a refund.",
                 "Can I return this and get refunded?", "What is the refund process?", "Refund please, how?",
                 "How do I ask for my money back?"],
    "shipping": ["How long does shipping take?", "When will my package arrive?", "How fast is delivery?",
                 "How many days for shipping?", "Where is my order?", "Is shipping slow?",
                 "When does my parcel get here?"],
    "password": ["I forgot my password.", "How do I reset my password?", "I can't log in.",
                 "How to change my password?", "Lost my password, help.", "Password reset?",
                 "I don't remember my password."],
    "cancel":   ["How do I cancel my order?", "Can I cancel an order?", "I want to cancel.",
                 "Cancel my purchase please.", "How to stop my order?", "Is it too late to cancel?",
                 "Can I undo my order?"],
    "address":  ["How do I change my address?", "I moved, how to update my address?", "Edit shipping address?",
                 "Wrong address, how to fix?", "Where do I add a new address?", "Change delivery address.",
                 "How can I update where my order goes?"],
    "contact":  ["How do I contact support?", "I need to talk to someone.", "Customer service phone?",
                 "How to reach you?", "Is there live chat?", "Where can I get help?",
                 "How do I get in touch with your team?"],
}
train = [(q, ANSWERS[k]) for k in QUESTIONS for q in QUESTIONS[k][:6]]   # 36 条
test = [(QUESTIONS[k][6], ANSWERS[k]) for k in QUESTIONS]               # 6 条没见过的问法
print(f"训练数据 {len(train)} 条,测试 {len(test)} 条(测试的问法训练时没出现过)")

# ---------------- 加载预训练模型 ----------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
old_vocab = model.get_input_embeddings().weight.shape[0]

@torch.no_grad()
def gen(prompt_ids, n=40, stop_id=None):
    model.eval()
    ids = torch.tensor([prompt_ids])
    for _ in range(n):
        nxt = model(ids).logits[0, -1].argmax().item()      # 贪心:每次取概率最大的
        if nxt == stop_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(ids[0, len(prompt_ids):])

print("\n=== 训练前:预训练 GPT-2 只会续写 ===")
for q in ["How do I get a refund?", "I forgot my password."]:
    print(f"  {q} →{gen(tok.encode(q), n=25)!r}")

# ---------------- 对话模板:加 3 个特殊 token ----------------
SPECIAL = ["<|user|>", "<|assistant|>", "<|end|>"]
tok.add_special_tokens({"additional_special_tokens": SPECIAL})
model.resize_token_embeddings(len(tok))              # 词向量表多 3 行(随机初始化,要靠训练学会)
USER, ASSIST, END = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
print(f"\n词表 {old_vocab} → {model.get_input_embeddings().weight.shape[0]}(新加 {SPECIAL})")

def build(q, a):
    """拼成 <|user|>问题<|assistant|>回答<|end|>,并给出 labels(不算的位置 = -100)"""
    prompt = [USER] + tok.encode(q) + [ASSIST]
    answer = tok.encode(" " + a) + [END]
    ids = prompt + answer
    labels = ([-100] * len(prompt) if MASK_USER else prompt[:]) + answer
    return ids, labels

# ---------------- ② 看一条数据:哪些 token 算 loss ----------------
ids, labels = build(*train[0])
print(f"\n=== 第 1 条数据拼好后({len(ids)} 个 token)===")
print("  token                 算不算 loss")
for i, l in zip(ids, labels):
    print(f"  {tok.decode([i])!r:22} {'✅' if l != -100 else '—'}")
counted = sum(l != -100 for l in labels)
print(f"  共 {len(ids)} 个 token,其中 {counted} 个算 loss")
print("  (模型内部会把 labels 往左错一位:第 t 个位置的输出去预测第 t+1 个 token)")

def batchify(pairs):
    rows = [build(q, a) for q, a in pairs]
    L = max(len(r[0]) for r in rows)
    pad = tok.eos_token_id
    input_ids = torch.tensor([r[0] + [pad] * (L - len(r[0])) for r in rows])
    labels = torch.tensor([r[1] + [-100] * (L - len(r[1])) for r in rows])   # 补齐的位置也不算
    attn = torch.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rows])
    return input_ids, labels, attn

@torch.no_grad()
def eval_loss(pairs):
    model.eval()
    x, y, m = batchify(pairs)
    return model(input_ids=x, attention_mask=m, labels=y).loss.item()

print(f"\n测试集 loss(只算回答部分):训练前 {eval_loss(test):.2f}")

# ---------------- ③ 训练循环 ----------------
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"\n=== 开始 SFT:{EPOCHS} 个 epoch,每批 {BATCH} 条,lr {LR},全参微调 ===")
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train)
    losses = []
    for i in range(0, len(train), BATCH):
        x, y, m = batchify(train[i:i + BATCH])
        loss = model(input_ids=x, attention_mask=m, labels=y).loss   # 交叉熵,自动跳过 -100
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep}  训练 loss {sum(losses)/len(losses):.2f}   测试 loss {eval_loss(test):.2f}   {time.time()-t0:.0f}s")

# ---------------- 训练后:SFT 模型 ----------------
print("\n=== 训练后:用没见过的问法提问 ===")
for q, a in test:
    out = gen([USER] + tok.encode(q) + [ASSIST], n=40, stop_id=END)
    print(f"  问:{q}\n  答:{out.strip()}\n")

print("=== 问一个和网店无关的问题 ===")
q = "What is the capital of France?"
print(f"  问:{q}\n  答:{gen([USER] + tok.encode(q) + [ASSIST], n=40, stop_id=END).strip()}")

# ---------------- 存下 SFT 模型:L3 dpo.py 从这里接着训练 ----------------
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_model")
model.save_pretrained(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)
print(f"\nSFT 模型已保存到 {SAVE_DIR}(约 500MB;L3 的 dpo.py 会从这里开始)")
