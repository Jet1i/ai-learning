"""
第 8 周 L4:DeepSeek-V3 全景 —— 把 671B 一块一块算出来
运行:python week8/v3.py

只用一条规则:一个矩阵的参数 = 行 × 列。
配置来自 DeepSeek-V3 公开的 config.json。
"""
d = 7168            # 每个 token 的向量维度
L = 61              # 层数:前 3 层普通 FFN,后 58 层 MoE
L_dense, L_moe = 3, 58
V = 129280          # 词表大小
H, d_h, d_rope = 128, 128, 64      # 头数、每头维度、RoPE 小 k 维度
q_c, kv_c = 1536, 512              # q 和 kv 的压缩维度
ffn_dense = 18432                  # 前 3 层普通 FFN 中间层
ffn_expert = 2048                  # 每个专家中间层(切得很细)
n_routed, n_shared, top_k = 256, 1, 8

B = 1e9
def show(name, n):
    if n < B:
        print(f"  {name:34s} {n / 1e6:8.1f} M")
    else:
        print(f"  {name:34s} {n / B:8.2f} B")
    return n

print("=== 1. 一层注意力(MLA) ===")
att = 0
att += show("q 压缩 W_dq   7168 × 1536", d * q_c)
att += show("q 展开 W_uq   1536 × 128×(128+64)", q_c * H * (d_h + d_rope))
att += show("kv 压缩 W_dkv 7168 × (512+64)", d * (kv_c + d_rope))
att += show("kv 展开 W_ukv 512 × 128×(128+128)", kv_c * H * (d_h + d_h))
att += show("输出 W_o      128×128 × 7168", H * d_h * d)
show("一层注意力合计", att)

print("\n=== 2. FFN ===")
dense_ffn = 3 * d * ffn_dense      # SwiGLU 3 个矩阵
expert = 3 * d * ffn_expert
show("普通 FFN(前 3 层用)", dense_ffn)
show("一个专家", expert)
moe_layer = (n_routed + n_shared) * expert + d * n_routed   # 加路由器
show("一层 MoE:257 个专家 + 路由器", moe_layer)

print("\n=== 3. 全模型 ===")
total = 0
total += show("注意力 × 61 层", att * L)
total += show("普通 FFN × 3 层", dense_ffn * L_dense)
total += show("MoE × 58 层", moe_layer * L_moe)
total += show("词表 embedding + 输出头(不共享)", 2 * V * d)
show("总参数(官方 671B)", total)

print("\n=== 4. 每个 token 激活多少 ===")
active = att * L + dense_ffn * L_dense + (top_k + n_shared) * expert * L_moe + 2 * V * d
show("激活参数(官方 37B)", active)
print(f"  激活比例 {active / total:.1%}  -> 每个 token 只算 {active / total:.0%} 的参数")

print("\n=== 5. 售前账本 ===")
w_fp8 = total * 1 / B
print(f"  权重 FP8(每参数 1 字节)       {w_fp8:6.0f} GB")
print(f"  8 × H100 80GB = 640 GB        {'放得下' if w_fp8 < 640 else '放不下 -> 要 2 台机器或 8 × H200'}")
print(f"  8 × H200 141GB = 1128 GB      剩 {1128 - w_fp8:.0f} GB 给 KV cache")
kv_tok_kb = (kv_c + d_rope) * L * 2 / 1024      # MLA 缓存按 bf16 算
print(f"  每 token KV cache(MLA)       {kv_tok_kb:.0f} KB")
print(f"  剩余显存能放多少 token        {(1128 - w_fp8) * 1e9 / (kv_tok_kb * 1024) / 1e6:.1f} M ≈ "
      f"{(1128 - w_fp8) * 1e9 / (kv_tok_kb * 1024) / 131072:.0f} 个 128K 的请求")

print("\n=== 6. 训练账本(6ND,第 6 周) ===")
D_tok = 14.8e12
flops = 6 * active * D_tok
print(f"  6 × {active / B:.0f}B × 14.8T token = {flops:.1e} 次运算")
print(f"  如果是 671B 稠密模型:{6 * total * D_tok:.1e},贵 {total / active:.0f} 倍")
print("  DeepSeek 报告:约 278.8 万 H800 GPU 小时,按 $2/小时 ≈ $5.6M(只算最后一次训练)")
