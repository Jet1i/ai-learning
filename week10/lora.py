"""第 10 周 L1:LoRA 原理 —— 不改原来的大矩阵 W,只学一个"小补丁" B·A
运行:python week10/lora.py      (只用 torch,CPU 几十秒)

还是第 3 周那个 y = x·W(Linear 层)。微调本来要改 W 的全部数字;
LoRA 说:把 W 冻住,另外学 ΔW = B·A,其中 B 是 d×r、A 是 r×d,r 很小(1~64)。
  ① 最小例子:4×4 的矩阵,秩 1 的补丁 = 一列 × 一行
  ② 为什么 B 要初始化成 0
  ③ 真实尺寸 768×768:全量 ΔW vs LoRA r=1/2/4/8/16,参数量和效果
  ④ 放回真实模型:GPT-2 和 7B 模型 LoRA 训练多少参数
"""
import torch
torch.manual_seed(0)

TRUE_RANK = 4          # 练习 1:改成 768(真实的改动不是低秩),再看 ③ 的结果
INIT_A_ZERO = False    # 练习 2:改成 True(A 也初始化成 0),看 ② ③ 会怎样

# ---------------- ① 最小例子:秩 1 = 一列 × 一行 ----------------
print("=== ① 最小例子:d=4, r=1 ===")
B = torch.tensor([[1.], [2.], [0.], [-1.]])      # 4×1
A = torch.tensor([[0.5, 0., 1., 2.]])             # 1×4
print("B (4×1) =", B.flatten().tolist())
print("A (1×4) =", A.flatten().tolist())
print("B·A (4×4) =\n", B @ A)
print("存 B 和 A 只要 4+4 = 8 个数,却能表示 16 个数的矩阵(每一行都是 A 的倍数)")
print("d=4 省得不多;d=4096 时:完整 ΔW 16,777,216 个数,r=8 的 B·A 只要 65,536 个(0.4%)\n")


class LoRALinear(torch.nn.Module):
    """y = x·W + x·A^T·B^T · (alpha/r)      W 冻住,只训练 A、B"""
    def __init__(self, W, r, alpha=None):
        super().__init__()
        d_in, d_out = W.shape
        self.W = W                                              # 冻住:不是 Parameter,不会被优化器改
        self.A = torch.nn.Parameter(torch.randn(r, d_in) / d_in ** 0.5)   # A 随机
        self.B = torch.nn.Parameter(torch.zeros(d_out, r))                # B = 0
        if INIT_A_ZERO:
            self.A.data.zero_()
        self.scale = (alpha or r) / r                           # 真实库里 alpha 常取 2r,这里先 = 1

    def forward(self, x):
        return x @ self.W + (x @ self.A.T @ self.B.T) * self.scale


# ---------------- ② B=0 → 训练第 0 步,模型和原来一模一样 ----------------
d = 768
W = torch.randn(d, d) / d ** 0.5                 # "预训练好的" 权重,冻住
layer = LoRALinear(W, r=8)
x = torch.randn(32, d)
print("=== ② B 初始化为 0 ===")
print(f"加上 LoRA 后,输出和原模型最大差 = {(layer(x) - x @ W).abs().max().item():.6f}")
print("→ 训练开始时模型行为完全没变,之后一点点长出补丁(不会一上来就把 SFT 好的模型打乱)\n")

# ---------------- ③ 真实尺寸:要学的改动 ΔW_true ----------------
# 假设"微调真正需要的改动"是低秩的(LoRA 论文的核心假设),再加一点噪声
U = torch.randn(d, TRUE_RANK); V = torch.randn(TRUE_RANK, d)
dW_true = (U @ V) / (d * TRUE_RANK) ** 0.5 + 0.001 * torch.randn(d, d)
X = torch.randn(2048, d)
Y = X @ (W + dW_true)                            # 目标:微调后的模型该有的输出


def train(module, params, steps=300, lr=1e-2):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = ((module(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


class FullFT(torch.nn.Module):                   # 对照组:直接学完整的 ΔW(d×d 全部数字)
    def __init__(self):
        super().__init__()
        self.dW = torch.nn.Parameter(torch.zeros(d, d))
    def forward(self, x):
        return x @ (W + self.dW)


base_loss = ((X @ W - Y) ** 2).mean().item()
print(f"=== ③ d={d},真实改动的秩 TRUE_RANK={TRUE_RANK} ===")
print(f"不微调的误差 {base_loss:.5f}")
print(f"{'方法':14}{'训练参数':>12}{'占比':>8}{'训练后误差':>12}")
full = FullFT()
print(f"{'全量 ΔW':14}{d*d:>12,}{'100%':>8}{train(full, [full.dW]):>12.5f}")
for r in [1, 2, 4, 8, 16]:
    m = LoRALinear(W, r)
    n = r * d * 2
    print(f"{'LoRA r=' + str(r):14}{n:>12,}{n/(d*d):>8.1%}{train(m, [m.A, m.B]):>12.5f}")
print("看:r 达到真实改动的秩之后,误差就和全量差不多了,但参数只要 1%~4%\n")

# ---------------- ④ 放回真实模型 ----------------
print("=== ④ 放回真实模型:每一层的 Linear 都可以挂一个 LoRA ===")
# GPT-2 124M:12 层,每层注意力的 c_attn 是 768→2304(一次算出 q,k,v)
r = 8
gpt2 = 12 * r * (768 + 2304)
print(f"GPT-2(12 层,只给 c_attn 挂 r=8):{gpt2:,} 个参数 = 124M 的 {gpt2/124.4e6:.2%}")
# LLaMA-2 7B:32 层,d=4096,常见做法给 q_proj 和 v_proj 挂 r=16
r = 16
llama = 32 * 2 * r * (4096 + 4096)
print(f"LLaMA-2 7B(32 层,q_proj+v_proj,r=16):{llama:,} 个参数 = 6.74B 的 {llama/6.74e9:.2%}")
print("训练完只要存这几 MB 的 A、B(叫 adapter),原模型一个字节都没改")
