# 第 7 周：推理优化 —— 屋顶线、FlashAttention、vLLM、量化

日期：2026-11-11 晚 ~ 11-12 凌晨 · 代码：`week7/` · 周测：约 1.8/5（凌晨 5 点作答，第 2 题跳过、第 5 题未作答）

## 一句话总结
同一块 GPU 怎么跑得更快、装得更多。**decode 是搬运受限**：每出一个 token 都要把全部权重从显存搬一遍，算力利用率不到 1%。三条优化路线都从这里出发：**FlashAttention** 让注意力的 T×T 中间表不进显存；**vLLM** 用分页管理 KV Cache、用连续批处理让每一步坐满人，batch 变大、吞吐翻几倍；**量化**让每个参数从 2 字节变成 0.5 字节，显存省 3/4、搬得少所以也更快。

## 核心概念

### 1. 屋顶线：为什么推理是"搬运受限"（roofline.py）
- 锚定 wx+b：生成 1 个 token，每个 w 从显存（HBM）搬到计算单元一次（fp16 = 2 字节），只做 1 乘 1 加。
- **屋脊点 = 算力 ÷ 带宽**：A100 = 312 万亿次/秒 ÷ 2 万亿字节/秒 = **156 次运算/字节**。强度低于它 = 搬运受限，高于它 = 算力受限。
- **batch** = 同一步里一起算的 token 数（decode 时 ≈ 同时在聊的用户数）。几个人的向量叠成矩阵一起乘 W，W 只搬一次（公交车：开一趟成本固定，坐 1 人还是 50 人都要开）。batch < 屋脊点时，加人几乎不花额外时间。
- 单用户上限 ≈ 带宽 ÷ 权重大小：7B fp16 在 A100 上 2000 ÷ 14 ≈ **143 token/s**；H100 ≈ 239；4090 ≈ 71。
- **prefill 算力受限，decode 搬运受限**：提示词几千个 token 共用一次搬运（强度高）；decode 每人一次只出 1 个（强度 ≈ 1）。对应指标 **TTFT**（首字延迟，看 prefill）和 **TPOT**（每个字的间隔，看 decode）。
- H100 算力是 A100 的 3.2 倍，但带宽只有 1.7 倍 → 单用户吐字只快 1.7 倍，3 倍算力要靠高并发才吃得满。
- 4090 屋脊点高是因为带宽小（1 TB/s），不是更强；显存 24 GB、没有 NVLink，适合个人开发和小并发。

### 2. FlashAttention（flash.py）
- **结果和普通注意力一模一样，计算量也一样，只改计算顺序**：T×T 分数表不写进显存。
- GPU 里两种存储：**HBM 显存**（仓库：80 GB，远）和**片上 SRAM**（工作台：每个 SM 约 192 KB，快）。Tensor Core 只能算桌上的数。
- 核心思想（GPA 比喻）：输出 = Σ(e^分数 × v) ÷ Σe^分数。**分子 o 和分母 l 都是加法，可以分批加，最后除一次**。
- **补救**：e¹⁴ ≈ 120 万，超过 fp16 能表示的 65504，所以要先减最大值；但 Flash 边看边算，只能用"目前最大值 m"当尺子，尺子变大时旧的 l、o 乘 e^(m旧−m新) 换算。
- 桌上一直只有：当前这批 k、v（每块 Bc 个 token，常用 64~128）+ 每行的 m、l、o。分数生了就用、用了就扔。
- 两种切法："**竖着切是为了省显存，横着切是为了并行**"。按 K/V 切成竖条，每个工人在自己的横条（一块 Q）里从左往右一块一块走；j < i 整块算、j = i 对角块遮一半、j > i 整块跳过。
- prefill 按 Q（行）分工，每个工人自己扫完整行，不用合并（softmax 按行做、行与行独立）；decode 只有 1 个 q，只能把 KV Cache 按列切给多个工人，最后用同一个补救公式合并（Flash-Decoding）。
- **不减少 KV Cache**。QKV 投影、Wo、FFN 是普通矩阵乘法，不叫 FlashAttention。
- 效果：注意力快 2~4 倍；32K 上下文时每个头省下 2 GB 的分数表，长上下文才跑得动。

### 3. vLLM（paged.py）
- **PagedAttention**：KV Cache 原来按最大长度给每个请求预留一整段连续显存（不知道回答多长），60~80% 空着。分页后每页 16 个 token，用满再给下一页，页不用连续，页表记录位置 → 浪费 < 4%。
- **抢占**：分页先让大家住进来，房间满了有人还要加页时，把最后进来的请求请出去重新排队（丢 KV 重算或暂存 CPU）。朴素预留一开始就留够，所以不会抢占。
- 页大小 16 是折中：页太小 KV 碎片化、GPU 搬运效率低、页表大；页太大最后一页浪费多。
- **连续批处理**：谁生成完了，下一步马上补人进来（静态批处理要等最慢的人）。
- **chunked prefill**：长提示词的 prefill 是一大块算力受限的计算（7B、8000 token 约 0.5 秒），会让其他人卡住；切成小段和别人的 decode 混着算。代价：长文档用户 TTFT 变慢，切块期间别人出字间隔变长但不卡顿。参数 `--enable-chunked-prefill`、`--max-num-batched-tokens`。

### 4. 量化（quant.py）
- 对称量化：刻度 = 最大绝对值 ÷ 127（int8）或 ÷ 7（int4）；存 round(w ÷ 刻度) 的整数 + 一个刻度；用时乘回来。
- 位数越少刻度越粗：int8 基本无损；int4 掉 1~2 个点；int3/int2 明显变差。
- **异常值问题**：一个特别大的数把刻度撑粗，小权重全被压成 0 → **分组量化**（每 128 个一组各用各的刻度，每参数多 0.125 bit）。GPTQ、AWQ 在此基础上再保护重要权重。
- 变快的原因是**搬得少**，不是算得少。
- 格式：GPU 用 AWQ / GPTQ / FP8（vLLM 支持）；Mac / Ollama 用 GGUF（如 Q4_K_M）。KV Cache 也能量化成 FP8，显存再省一半。

## 关键代码
```python
ridge = tflops * 1e12 / (bw_TBs * 1e12)              # 屋脊点
t_token = N * bytes_per_w / bandwidth                 # decode 每 token 下限
# online softmax（每来一块 K/V）
m_new = np.maximum(m, Sj.max(1, keepdims=True)); scale = np.exp(m - m_new)
Pj = np.exp(Sj - m_new); l = l * scale + Pj.sum(1, keepdims=True); O = O * scale + Pj @ Vj
out = O / l                                           # 最后才除
need = -(-n_tokens // PAGE) * PAGE                    # 分页：向上取整到整页
scale = np.abs(g).max(1, keepdims=True) / (2**(bits-1) - 1)   # 分组量化
q = np.clip(np.round(g / scale), -qmax, qmax)
```

## 实验结果
- Mac 上测 x @ W（4096×4096）：batch 2 → 64 总耗时 7.02 → 7.80 ms（+11%），每 token 便宜约 29 倍。
- flash.py：分块结果与普通注意力最大差 4e-16；Bc = 1/16/256 结果都一样；去掉补救误差 0.36。
- paged.py（13B、80GB、500 请求）：朴素 15 并发、显存真用率 24%；分页平均 55 / 最多 108 并发、91%，**快 3.7 倍**。PAGE=256 时降到 50 并发、82%。
- quant.py（一层 4096×4096，混入异常值）：int8 整体 3.4% / 分组 0.7%；int4 整体 62% / 分组 12%。

## Demo
- [屋顶线](../week7/roofline-demo.html) · https://claude.ai/artifact/CyQbGqCY1FKha5S7rQXsHP
- [FlashAttention：分块 + online softmax](../week7/flash-demo.html) · https://claude.ai/artifact/QsSBeKbrMcje6gNw8TxV6z
- [一步一步看：Q K V 怎么搬（prefill + decode）](../week7/flash-walkthrough.html) · https://claude.ai/artifact/GjFT4JC4KqdkfLM2udvX62
- [从输入到第一个词：prefill 全过程（真实数字）](../week7/prefill-walkthrough.html) · https://claude.ai/artifact/2jwhZAXTE3qWFVNA7rWFL6
- [FlashAttention 拆到最细：只看一行](../week7/flash-one-row.html) · https://claude.ai/artifact/BtrJFR2bGJX88zPLNJZRi6 ← **最推荐先看这个**
- [动画：竖着切、往右走](../week7/flash-tiles-anim.html) · https://claude.ai/artifact/SGUv7cpAp665dNjNJQr5se
- [vLLM 酒店：预留整段 vs 分页](../week7/paged-hotel.html) · https://claude.ai/artifact/T8Ae6Mfq7t2eUvEyLnNeXb
- [连续批处理 + chunked prefill](../week7/batching-demo.html) · https://claude.ai/artifact/4fth3B2w61ME7i98UcTMLP
- [量化：把权重吸到刻度上](../week7/quant-demo.html) · https://claude.ai/artifact/WrAtr9WfdrsuRMBUZQaS5B

## 我的提问与解答
完整记录见 [`week7/qa-log.md`](../week7/qa-log.md)（约 30 个问题）。精选：
- **搬运是什么意思？** 把参数从显存搬到计算单元旁边的 SRAM；每个 token 都要把每层 W 搬一遍，因为 SRAM 放不下。
- **屋脊点是最多能处理的 token 吗？** 是"免费的上限"：低于它加人几乎不花时间，高于它还能处理，只是每步变慢。
- **4090 为什么这么厉害？** 算力高，但推理吃带宽和显存，它都差 A100 一截。
- **Bc 是什么？** 每一块里几个 token 的 K、V；所有 token 都做 Flash，Bc 只决定一口吃多少。
- **v 是干嘛的？** q·k 只决定看谁，v 才是拿走的内容；K、V 同一批搬上来，分数一出生就去加权 v。
- **KV Cache 在 SRAM 吗？** 在 HBM。"缓存" = 存起来以后用；SRAM 是临时工作台。
- **token 5 不用把 1~4 拿回来看吗？** 要看（读 k、v），但不用重算。
- **Flash-Decoding 里工人 2 要等工人 1 吗？** 不用，各算各的 (m, l, o)，最后合并。
- **prefill 为啥不用合并？** softmax 按行做，每个工人自己扫完整行。
- **长上下文时一行 T 个分数都在 SRAM 吗？** 不存，每次只有一批 128 个流过桌面。
- **算完怎么知道注意力大小？** 推理不需要；训练反向时用 m、l 重算。
- **我的总结**："竖着切下来往右移动"——对。

## 易错点
- softmax **最后要除以总和**（复习第 1 题又漏了；输出应落在 v 的范围内）。
- 参数量要**算**，不要凭上周印象猜（复习题把 13B 猜成 8B）。
- 量化变快是因为**搬得少**，不是算得少（L1 答错，L4 改对）。
- **÷√d 和 softmax 的 ÷总和**是两次不同的除法。
- 补救是为了防 **fp16 数值溢出**，和 SRAM 容量无关；SRAM 容量决定 Bc。
- FlashAttention **不减少 KV Cache**。
- **连续批处理**解决"座位空着"，**chunked prefill**解决"长 prefill 卡住别人"，别混。
- 抢占 = 把在住的请求请出去，不是拒绝新请求。
- 其他人的出字速度叫 **TPOT**，不叫 TTFT。
- 售前题**一定先算数字**（周测第 1、5 题都丢在这）；4090 + 8B + 20×8K 的坑：权重量化了 KV Cache 仍是大头。

## 复习题
1. A100 的屋脊点怎么算？为什么 batch 从 1 涨到 32 每步时间几乎不变？7B fp16 单用户最多多少 token/s？
2. 手算：第 1 块分数 [0, 0]、v [2, 4]，第 2 块分数 [ln2]、v [8]，写出每块后的 l、o 和最终输出。
3. 用一句话说 FlashAttention 省了什么、为什么 softmax 能分块、补救为什么存在。
4. 为什么 decode 时 Flash 要把 KV 切给多个工人最后合并，而 prefill 不用？（周测替换题）
5. PagedAttention、连续批处理、chunked prefill 各解决什么问题？代价各是什么？
6. 一张 4090 跑 LLaMA-3 8B，20 人 × 8K：fp16 / int4 / int4 + FP8 KV 分别要多少显存？单路速度各多少？
