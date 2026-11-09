"""
第 5 周 L2：第一个 RAG = 切块 → 向量化 → 检索 → 把资料塞进 prompt → 生成
运行:  python week5/rag1.py      然后直接提问，输入 q 退出
知识库: week5/kb/*.md（虚构的"云帆云"帮助文档，可以自己改、自己加）

RAG 两个阶段：
  离线（建索引）：文档 → 切成小块 → 每块用 bge 变成向量 → 存起来
  在线（回答）  ：问题 → 变成向量 → 和所有块算相似度 → 取前 k 块 → 拼进 prompt → DeepSeek 回答
"""
import glob
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "kb")
TOP_K = 3          # 每次取最像的几块
THRESHOLD = 0.55   # 最高分低于它就当"知识库里没有"。L4 eval.py 门槛扫描定的：0.55 放行 10/12、挡住 4/4


# ---------- 1. 切块 ----------
def load_chunks(kb_dir=KB_DIR, max_len=300, overlap=50):
    """按 '## 小标题' 切；一节太长再按 max_len 个字硬切，相邻两块重叠 overlap 个字"""
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        fname = os.path.basename(path)
        title = ""
        for sec in open(path, encoding="utf-8").read().split("\n## "):
            lines = sec.strip().splitlines()
            if lines[0].startswith("# "):  # 文件开头的大标题，记下来
                title = lines[0][2:]
                continue
            head, body = lines[0], "".join(lines[1:]).strip()
            start = 0
            while True:
                piece = body[start:start + max_len]
                # 小技巧：每块前面带上"文档标题 - 小标题"，这样单独拿出一块也知道它在讲什么
                chunks.append({"source": f"{fname} > {head}", "text": f"{title} - {head}：{piece}"})
                if start + max_len >= len(body):
                    break
                start += max_len - overlap
    return chunks


# ---------- 2. Embedding（和 L1 一样）----------
_tok = _model = None


def embed(texts):
    global _tok, _model
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _model is None:
        name = "BAAI/bge-small-zh-v1.5"
        _tok = AutoTokenizer.from_pretrained(name)
        _model = AutoModel.from_pretrained(name).eval()
    batch = _tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        v = _model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1).numpy()


# ---------- 3. 建索引（知识库很小，每次启动现算；大了就要存进向量数据库）----------
_chunks = _vecs = None


def get_index():
    global _chunks, _vecs
    if _chunks is None:
        _chunks = load_chunks()
        _vecs = embed([c["text"] for c in _chunks])
    return _chunks, _vecs


# ---------- 4. 检索 ----------
def search(query, k=TOP_K):
    chunks, vecs = get_index()
    scores = vecs @ embed([query])[0]   # 一次算出和所有块的余弦相似度
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


# ---------- 5. 生成 ----------
SYSTEM = (
    "你是云帆云的客服助手。只根据【参考资料】回答，在用到的句子后面用 [1] [2] 标出处。"
    "如果资料里没有答案，就回答「知识库里没有相关信息」，不要自己编。"
)
_client = None


def answer(query, verbose=True):
    global _client
    hits = search(query)
    if verbose:
        print("  检索结果：")
        for s, c in hits:
            print(f"    {s:.3f}  {c['source']}")
    good = [(s, c) for s, c in hits if s >= THRESHOLD]
    if not good:
        return "知识库里没有找到相关信息。（最高分低于门槛，没有调用大模型）"

    context = "\n".join(f"[{i + 1}]（{c['source']}）{c['text']}" for i, (s, c) in enumerate(good))
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    resp = _client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{query}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    chunks, _ = get_index()
    print(f"知识库切成了 {len(chunks)} 块。提问吧（q 退出）\n")
    while (q := input("你：").strip()) != "q":
        if q:
            print("助手：" + answer(q) + "\n")
