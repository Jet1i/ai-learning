"""第 10 周 L3:合并 LoRA + QLoRA 显存估算
运行:python week10/merge_qlora.py
前提:先跑过 L2 的 lora_sft.py(MODE=lora),有 week10/shopx_lora 这个 adapter

  ① 合并:W' = W + (alpha/r)·B·A,合完就是一个普通模型,推理不多算一步
  ② 验证:合并前后输出一样吗?速度差多少?
  ③ 显存估算:全参 / LoRA / QLoRA 训练 7B、14B、70B 各要多少(售前用)
"""
import os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.join(HERE, "..", "week9", "sft_model")
ADAPTER = os.path.join(HERE, "shopx_lora")

tok = AutoTokenizer.from_pretrained(SFT_DIR)
base = AutoModelForCausalLM.from_pretrained(SFT_DIR)
model = PeftModel.from_pretrained(base, ADAPTER)      # 底座 + 补丁(没合并)
model.eval()
USER, ASSIST = tok.convert_tokens_to_ids("<|user|>"), tok.convert_tokens_to_ids("<|assistant|>")
q = "Is my gift card still valid next year?"
ids = torch.tensor([[USER] + tok.encode(q) + [ASSIST]])

# ---------------- ① 看一层里的 W、A、B ----------------
layer = model.base_model.model.transformer.h[0].attn.c_attn
A = layer.lora_A["default"].weight      # r × 768
B = layer.lora_B["default"].weight      # 2304 × r
scale = layer.scaling["default"]
print("=== ① 第 0 层 c_attn ===")
print(f"W {tuple(layer.base_layer.weight.shape)}   A {tuple(A.shape)}   B {tuple(B.shape)}   scale = alpha/r = {scale}")
print(f"B 已经不是 0 了:|B| 平均 {B.abs().mean().item():.4f}(L2 训练长出来的补丁)")


def timed_logits(m, n=20):
    with torch.no_grad():
        m(ids)                                # 热身
        t0 = time.time()
        for _ in range(n):
            out = m(ids).logits[0, -1]
        return out, (time.time() - t0) / n * 1000


# ---------------- ② 合并前后对比 ----------------
before, t_before = timed_logits(model)
merged = model.merge_and_unload()             # 每个挂了 LoRA 的层:W ← W + scale·(B·A),然后扔掉 A、B
after, t_after = timed_logits(merged)
print("\n=== ② 合并前后 ===")
print(f"下一个 token 的 logits 最大差 {(before - after).abs().max().item():.6f}(≈0 → 行为一样)")
print(f"一次前向:合并前 {t_before:.1f} ms   合并后 {t_after:.1f} ms")
print(f"合并后还有 LoRA 层吗?{any('lora' in n for n, _ in merged.named_parameters())}")
print("代价:合并后补丁'焊死'在 W 里,拔不下来了;要存的话又是一个完整的 500 MB 模型")

# ---------------- ③ 显存估算 ----------------
print("\n=== ③ 训练显存估算(GB,粗算,不含框架开销)===")
GB = 1e9
def lora_params(n_layers, d, r=16, mods=2):        # 给 q_proj、v_proj 挂 LoRA
    return n_layers * mods * r * (d + d)
MODELS = {"7B": (7e9, 32, 4096), "14B": (14e9, 40, 5120), "70B": (70e9, 80, 8192)}
ACT = 4   # 激活值:小 batch + gradient checkpointing,粗估 ~4 GB(序列越长越大)
print(f"{'模型':6}{'全参(16B/参数)':>16}{'LoRA(W fp16)':>16}{'QLoRA(W 4bit)':>16}")
for name, (N, L, d) in MODELS.items():
    n = lora_params(L, d)
    full = 16 * N / GB
    lora = 2 * N / GB + 16 * n / GB + ACT
    qlora = 0.5 * N / GB + 16 * n / GB + ACT
    print(f"{name:6}{full:>14.0f}  {lora:>14.0f}  {qlora:>14.0f}")
print("""
每一列在算什么:
  全参  = 每个参数 16 字节(权重 2 + 梯度 2 + Adam m、v 各 4 + fp32 主权重 4)
  LoRA  = W 只存 fp16 权重(2 字节,不训练所以没有梯度和 Adam)+ LoRA 参数×16 + 激活
  QLoRA = W 压成 4 bit(0.5 字节,第 7 周的分组量化)+ LoRA 参数×16 + 激活
          前向时把 W 临时解压回 bf16 算,LoRA 的 A、B 一直是 bf16 训练
""")
