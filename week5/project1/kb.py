"""
项目 1 · 知识库核心：读笔记 → 切块 → 两路检索（向量 + 关键词）→ 融合
被 server.py / agent.py / eval.py 共用。

和 L2 rag1.py 相比的升级：
  1. 切块懂 Markdown：跳过 YAML 头、不把代码块里的 # 当标题、按段落装箱，每块带"面包屑"出处
  2. 混合检索：bge 向量（懂意思）+ BM25 关键词（认得 AccessKey、GPT-2 这种专有名词），用 RRF 融合排名
  3. 向量缓存：笔记没变就不重算，启动快
  4. 小块检索、大块返回：read_section() 可以取回命中块所在的整个小节
"""
import glob
import hashlib
import math
import os
import re
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # ~/ai-learning
SOURCES = ["notes/week*.md", "week*/qa-log.md"]               # 知识库 = 每周笔记 + 每周问答记录
CACHE = os.path.join(HERE, ".cache_vectors.npz")
MAX_LEN = 400        # 一块最多多少字
THRESHOLD = 0.55     # 向量最高分低于它 → 当作"笔记里没有"（L4 评测定的，换了知识库要重新量）


# ---------- 1. 读文件 + 按 Markdown 结构切块 ----------
def split_sections(text):
    """返回 [(标题路径, 正文)]。跳过 YAML 头；``` 代码块里的 # 不算标题"""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
    sections, path, buf, in_code = [], [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((" > ".join(path), "\n".join(buf).strip()))
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2).strip()]
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((" > ".join(path), "\n".join(buf).strip()))
    return sections


def pack(body, max_len=MAX_LEN):
    """按空行分段（太长的段再按行拆），再把相邻段落装进不超过 max_len 的箱子；单段超长就硬切"""
    paras = []
    for p in re.split(r"\n\s*\n", body):
        # 太长的段落（比如一长串列表）再按行拆开，尽量不在句子中间切
        paras += [p] if len(p) <= max_len else p.splitlines()
    paras = [p.strip() for p in paras if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > max_len:
            out.append(p[:max_len])
            p = p[max_len - 50:]          # 硬切时留 50 字重叠
        if len(cur) + len(p) + 1 > max_len and cur:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        out.append(cur)
    return out


def load_chunks():
    chunks, sections = [], {}
    for pattern in SOURCES:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(path, ROOT)
            for head, body in split_sections(open(path, encoding="utf-8").read()):
                src = f"{rel} > {head}" if head else rel
                sections[src] = body
                for piece in pack(body):
                    chunks.append({"source": src, "text": f"[{head}] {piece}"})
    return chunks, sections


# ---------- 2. 向量检索（bge，和 L1/L2 一样）----------
_tok = _model = None


def embed(texts, batch=32):
    global _tok, _model
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
        _model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").eval()
    out = []
    for i in range(0, len(texts), batch):
        b = _tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            v = _model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(out)


# ---------- 3. 关键词检索（BM25）----------
def tokenize(s):
    """中文按"相邻两个字"切（没装分词器的简易做法），英文/数字按整词，统一小写"""
    s = s.lower()
    words = re.findall(r"[a-z0-9_.\-]+", s)
    han = re.findall(r"[一-鿿]+", s)
    bigrams = [h[i:i + 2] for h in han for i in range(max(1, len(h) - 1))]
    return words + bigrams


class BM25:
    """经典关键词打分：词在这块里出现越多、在整个库里越稀有，分越高；长块会被适当打折"""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return s


# ---------- 4. 索引 + 混合检索 ----------
class Index:
    def __init__(self):
        self.chunks, self.sections = load_chunks()
        texts = [c["text"] for c in self.chunks]
        key = hashlib.md5("\n".join(texts).encode()).hexdigest()
        if os.path.exists(CACHE) and str(np.load(CACHE)["key"]) == key:
            self.vecs = np.load(CACHE)["vecs"]           # 笔记没变：直接读缓存
        else:
            self.vecs = embed(texts)
            np.savez(CACHE, key=key, vecs=self.vecs)
        self.bm25 = BM25(texts)

    def search(self, query, k=4, mode="hybrid"):
        """返回 [(chunk, 向量相似度)]，按 mode 排序。mode: vector / bm25 / hybrid"""
        cos = self.vecs @ embed([query])[0]
        if mode == "vector":
            order = np.argsort(-cos)
        elif mode == "bm25":
            order = np.argsort(-self.bm25.scores(query))
        else:
            # RRF（倒数排名融合）：不比分数（两种分数单位不同），只比名次。名次越靠前，1/(60+名次) 越大
            rrf = np.zeros(len(self.chunks))
            for s in (cos, self.bm25.scores(query)):
                for rank, i in enumerate(np.argsort(-s)):
                    rrf[i] += 1 / (60 + rank + 1)
            order = np.argsort(-rrf)
        return [(self.chunks[i], float(cos[i])) for i in order[:k]]

    def best_cos(self, query):
        return float((self.vecs @ embed([query])[0]).max())


_index = None


def get_index():
    global _index
    if _index is None:
        _index = Index()
    return _index
