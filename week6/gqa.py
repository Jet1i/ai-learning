# 第 6 周 第 4 课：KV Cache 与 GQA —— 为什么推理时显存不够用？
# numpy。运行：python week6/gqa.py
import numpy as np
np.random.seed(0)

# ================= 1. 什么是 KV Cache =================
# 生成第 t 个 token 时，新 token 的 q 要和前面所有 token 的 k 做点积，再加权所有 v。
# 前面 token 的 k、v 每一步都一样（x@Wk、x@Wv 不会变），每次重算太浪费 → 存起来，叫 KV Cache。
# 代价：每生成一个 token，每一层都要多存一份 k 和 v。

def kv_cache_bytes(layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    # 2 = K 和 V 各一份；bytes_per=2 表示 fp16/bf16
    return 2 * layers * n_kv_heads * head_dim * seq_len * batch * bytes_per

GB = 1024**3
print("== 1. 一个用户、一条对话的 KV Cache 有多大？ ==")
for name, L, n_heads, n_kv, dh in [
        ("LLaMA-2 7B  (MHA，32 头各自有 K V)", 32, 32, 32, 128),
        ("LLaMA-3 8B  (GQA，32 个 Q 头共享 8 组 K V)", 32, 32, 8, 128),
        ("LLaMA-2 70B 假如用 MHA（64 组 K V）", 80, 64, 64, 128),
        ("LLaMA-2 70B 实际 GQA（8 组 K V）", 80, 64, 8, 128)]:
    for T in [4096, 32768]:
        print(f"{name:40s} 上下文 {T:6d}：{kv_cache_bytes(L, n_kv, dh, T)/GB:6.2f} GB")

# ================= 2. MHA / GQA / MQA：多少个头共享一组 K V =================
d, H, T = 32, 8, 6        # 8 个 Q 头，每头 4 维
dh = d // H
x = np.random.randn(T, d)
def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def attention(x, n_kv):
    # Q 还是 8 个头；K、V 只有 n_kv 组，每组被 H/n_kv 个 Q 头共用
    Wq = np.random.randn(d, H * dh) / np.sqrt(d)
    Wk = np.random.randn(d, n_kv * dh) / np.sqrt(d)    # ← 变小了！
    Wv = np.random.randn(d, n_kv * dh) / np.sqrt(d)
    Q = (x @ Wq).reshape(T, H, dh).transpose(1, 0, 2)      # (8, T, dh)
    K = (x @ Wk).reshape(T, n_kv, dh).transpose(1, 0, 2)   # (n_kv, T, dh) ← 要缓存的
    V = (x @ Wv).reshape(T, n_kv, dh).transpose(1, 0, 2)
    rep = H // n_kv
    K_all, V_all = np.repeat(K, rep, 0), np.repeat(V, rep, 0)   # 复制给同组的 Q 头用（不占额外缓存）
    s = Q @ K_all.transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.triu(np.ones((T, T)), 1) == 1, -np.inf, s)
    out = (softmax(s) @ V_all).transpose(1, 0, 2).reshape(T, d)
    return out, K.size + V.size, Wk.size + Wv.size

print("\n== 2. 8 个 Q 头，K V 分几组？ ==")
for name, n_kv in [("MHA 每头一组", 8), ("GQA 两头一组", 4), ("GQA 四头一组", 2), ("MQA 全部共用一组", 1)]:
    out, cache, p = attention(x, n_kv)
    print(f"{name:16s} n_kv={n_kv}：输出 {out.shape}，每层缓存 {cache:4d} 个数（{cache/(2*T*d):.3f} 倍），Wk+Wv 参数 {p}")
print("→ 输出形状完全一样，下一层照常接；缓存按 n_kv/H 等比例缩小。")

# ================= 3. 售前视角：一张卡能同时服务多少人？ =================
print("\n== 3. 一张 80GB A100 部署 8B 模型（fp16），能同时服务多少条 4K 对话？ ==")
weights = 8e9 * 2 / GB                  # 80 亿参数 × 2 字节
free = 80 - weights - 4                 # 再留 4GB 给中间计算
for name, n_kv in [("MHA（32 组）", 32), ("GQA（8 组）", 8), ("MQA（1 组）", 1)]:
    per = kv_cache_bytes(32, n_kv, 128, 4096) / GB
    print(f"{name:10s} 每条 {per:5.2f} GB → 约 {int(free // per):4d} 条并发")
print(f"（模型权重约 {weights:.1f} GB，剩下 {free:.0f} GB 给 KV Cache）")
print("→ 并发数直接决定每张卡能接多少用户 = 每个用户的成本。GQA 是 MHA 和 MQA 的折中：省显存，效果几乎不掉。")
