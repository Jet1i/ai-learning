# 第 6 周：Scaling Laws、LLaMA 架构、RoPE、GQA 与 KV Cache

日期：2026-11-10 · 代码：`week6/` · 周测：约 1/5（深夜作答，4、5 题看答案学）

## 一句话总结
从 GPT-2 到 LLaMA，骨架没变（归一化 → 注意力 → 残差 → 归一化 → FFN → 残差），改的每一处都是**同样的算力榨出更多效果，或者让推理更便宜**：RMSNorm、SwiGLU、去偏置让训练更省更好；RoPE 让位置不再占一张表、能扩展到长上下文；GQA 让推理时的 KV Cache 变成 1/4，一张卡能服务的人数翻几倍。Scaling Law 告诉我们钱该怎么分：N 和 D 一起涨，约 20 token/参数，但推理便宜更重要时会故意“喂撑”小模型。

## GPT-2 vs LLaMA 对比（本周产出）

| 部件 | GPT-2 (2019) | LLaMA (2023 起) | 解决什么 | 代价 |
|---|---|---|---|---|
| 位置信息 | 学一张位置表 P（1024×d），加到词向量上 | 无位置表；注意力里把 q、k 按位置**旋转**（RoPE） | 分数只看相对距离；没有“1024 行”硬上限；可插值扩展长上下文 | 每层多一点旋转计算；超过训练长度仍会变差 |
| 归一化 | LayerNorm：(x−均值)/标准差 × g + b | RMSNorm：x/均方根 × g | 少算均值、少一组参数，更快，效果不掉 | 几乎没有 |
| 归一化位置 | Pre-Norm | Pre-Norm（沿用） | 残差主干干净，深层训练稳 | — |
| FFN | GELU，d→4d→d，2 个矩阵 | SwiGLU，3 个矩阵，中间 (8/3)d（7B 用 11008） | 同参数量下效果更好（一个 wx 当开关、一个当内容） | 多一次矩阵乘法调度 |
| 偏置 | x@W + b | 全部去掉 b | 参数略少、训练更稳 | — |
| K/V | MHA：每个头一组 K/V | LLaMA-2 70B / LLaMA-3：GQA（如 32 个 Q 头共享 8 组） | KV Cache 变 1/4，并发翻几倍 | 表达力略降（远好于 MQA） |
| 输出头 | 复用词表 E（共享） | 单独一个 V×d 矩阵 | 输入/输出可以学得不同 | 多 V·d 参数 |
| 词表 | 50257 | LLaMA-2 32000 → LLaMA-3 128256 | 中文/代码切得更省（更便宜、同上下文装更多） | V·d 两份参数变大（8B 里约 10 亿） |
| 规模（代表） | small：12 层，d=768，124M | 7B：32 层，d=4096 | — | — |

## 核心概念

### 1. Scaling Laws（scaling.py）
- 参数量 N ≈ 12·L·d² + V·d；训练算力 **C ≈ 6·N·D**（锚定 wx+b：每个 w 每个 token 前向 1 乘 1 加 = 2，反向约 4）。
- 两个发现：① Kaplan 2020：loss 随 N、D、C 按幂律**可预测地**下降 → 先训小模型拟合曲线，就能预测大模型；② Chinchilla 2022：预算固定时 N 和 D **一起涨**，约 20 token/参数。算力 ×10 → N、D 各 ×√10≈3。
- Loss = E + A/N^α + B/D^β：E 是地板（语言本身的不确定性），后两项是“模型太小”“数据太少”，短板效应 → 钱花在最长的那块上。
- GPT-3 175B 只配 300B token（D/N≈2）是**故意的**：当时信 Kaplan（偏向做大模型，学习率计划没按训练长度调导致低估数据的作用）。按 Chinchilla 重分约 50B + 1T token 更好。
- LLaMA-3 8B 训 15T（D/N≈1875）：和 Chinchilla 70B 训练算力差不多、loss 差不多，但推理便宜约 9 倍。**训练付一次，推理天天付。**
- 模型能力 ≈ 算力 × 数据质量 × 算法效率。数据墙：高质量文本只有几十 T token；去重、过滤、合成数据。DeepSeek-V3 约 280 万 H800 GPU 小时，靠效率。

### 2. LLaMA 块（llama_block.py）
- RMSNorm vs LayerNorm：例 x=[2,4,6,8]，LN → [−1.34,−0.45,0.45,1.34]；RMS → [0.37,0.73,1.10,1.46]。都把大小缩到 1，LN 还挪中心。
- 归一化的作用：残差一路往上加，数值会漂；每个子层入口“稳压”，否则 softmax 太尖、梯度爆炸/消失。
- 参数计数规则：a×b 矩阵 = a·b 个参数。注意力 Wq Wk Wv Wo 各 d×d = 4d²（Wo 混合各头；多头是把 d 切开，不加参数）；GPT-2 FFN 2×4d² = 8d²；SwiGLU 3·d·h，h=4d 时 12d²（多 50%），h=(8/3)d 时 8d²（公平对比）。
- LLaMA-2 7B：32×(4d² + 3·d·11008 + 2d) + 2·V·d + d = **6.74B**。用 12Ld²+Vd 估是 6.57B，差的主要是输出头不共享（+V·d）。

### 3. RoPE（rope.py）
- 位置 m 的 q 转 m·θ，位置 n 的 k 转 n·θ，点积只取决于夹角差 (n−m)θ → **分数只看相对距离**。(2,5)(10,13)(100,103) 分数都是 0.9524；加位置表则各不相同。
- 公式现算，无表无参数。每两维一组，转速 θᵢ = base^(−2i/d)：快的像秒针管近距离，慢的像时针管远距离（d=64、base 10000 时最慢一圈 4.7 万位置；LLaMA-3 base 500000 → 208 万）。
- 只转 q、k，不转 v（位置影响“看谁”，不影响“拿什么”）；转的是结果 q，不是参数 Wq。
- 天然偏向近的词（距离越远分数越小）。
- 长上下文三道坎：① 超过训练长度，慢指针转到没见过的角度 → 崩，用位置插值/YaRN + 长文本继续训练；② T² 计算 + KV Cache 显存；③ lost in the middle（U 形：开头结尾记得好，中间差）。
- 为什么不能“把所有角度都训完”：慢指针必须慢（快了会撞车），覆盖它要很长文本；位置是所有指针的组合。实际：主体 4K~8K，最后加长上下文阶段。

### 4. KV Cache 与 GQA（gqa.py）
- 注意力 = T×T 分数表（每格 q·k，一个数）；q、k、v 是向量（x@W）。整句一起算是 T×T，生成时每个新 token 只算一行，前面的 k、v 从缓存拿 = **KV Cache**。
- 每层计算：参数矩阵约 24·T·d²，注意力约 4·T²·d，比值 T/6d（d=4096 时约 2.4 万 token 是分界）。
- KV Cache = 2(K,V) × 层数 × K/V 组数 × 每组维度 × token 数 × 字节。LLaMA-3 8B：**128 KB/token**（8K → 1 GB，128K → 16 GB，比权重还大）；MHA 是 512 KB/token。
- MHA / GQA / MQA：多少个 Q 头共享一组 K/V。这是**训练前定的结构**，Wk、Wv 本身变小；MQA 一份“简历”给 32 个面试官，质量降、训练不稳；GQA 8 份够用。
- 显存速算：fp16 显存 GB ≈ 参数(B) × 2（8B → 16 GB；70B → 140 GB；int4 约 1/4）。一张 80GB 卡部署 8B、4K 上下文：MHA 约 30 路、GQA 约 122 路。

## 关键代码
```python
C = 6 * N * D                                        # 训练算力
loss = E + A / N**alpha + B / D**beta                # Chinchilla
rms = np.sqrt((x**2).mean(-1, keepdims=True) + 1e-5); y = x / rms * g      # RMSNorm
ffn = (silu(x @ Wg) * (x @ Wu)) @ Wd                 # SwiGLU
ang = pos[:, None] * base ** (-2 * np.arange(d//2) / d)                     # RoPE 角度
out[..., 0::2] = x1 * cos - x2 * sin; out[..., 1::2] = x1 * sin + x2 * cos
K_all = np.repeat(K, H // n_kv, axis=0)              # GQA：一组 K/V 给多个 Q 头用
kv_bytes = 2 * layers * n_kv * head_dim * seq_len * 2
```

## 实验结果
- isoFLOP（C=1e21）：最优 N≈1.8B、D≈91B（这组拟合常数给 D/N≈50；论文另两种方法约 20）。
- LLaMA-2 7B 参数复算 6.74B ✓。
- RoPE：整句平移 1000 位，注意力分数最大差 0.0。
- KV Cache：LLaMA-2 70B 若用 MHA，32K 上下文一条 80 GB；GQA 10 GB。

## Demo
- [Scaling Laws：算力该怎么分](scaling-demo.html) · https://claude.ai/artifact/4bZCywxk27bGFKwzSnDrea
- [GPT-2 块 vs LLaMA 块（含 LayerNorm/RMSNorm 一步步算）](gpt2-vs-llama.html) · https://claude.ai/artifact/ULQjofFRyHUZ1ZbAy8emHW
- [RoPE：用转角度表示位置](rope-demo.html) · https://claude.ai/artifact/EZ2HZwLfcK3fwmswVLkmDi
- [KV Cache 与 GQA：显存都去哪了](gqa-demo.html) · https://claude.ai/artifact/6VTSrhwc2XW7zMDsTtFn8C

## 我的提问与解答
完整记录见 [`week6/qa-log.md`](../week6/qa-log.md)。精选：
- **D 和 D/N 是什么？** D = 训练 token 总数；D/N = 每个参数分到多少数据（每个旋钮被教了多少次）。
- **为什么参数大，token 也要多？** 旋钮多需要更多例子才拧得准；公式上是短板效应。
- **是不是卡多就赢？** 同配方下是，但有递减、数据墙、效率三件事。
- **位置表 1024 行？** 1024 个位置各一行 = 上下文上限；上下文长度只影响这张表，不影响其他参数。
- **LLaMA 不是不用词表吗？** 去掉的是**位置表**，词表还在；输出头另学一个 V×d。
- **q/k 和 Wq/Wk？** Wq 是参数（机器），q 是结果（产品）；参数量数 Wq，RoPE 转 q，KV Cache 存 k、v。
- **temperature 就是不除 √d？** 不是：√d 在每层注意力内部、固定；temperature 在最后挑词时、API 里自己设。
- **GQA 不是只省推理吗，为啥影响质量？** 它是结构改动，训练时 K/V 就少了。

## 易错点
- **算力 ×100 → N、D 各 ×10**（√100），不是 30 倍。（周测第 1 题）
- 参数量题没动手算（周测第 2 题）：按“每层注意力 → 每层 FFN → ×L → 词表 + 输出头”四步走，LLaMA-3 8B = 4194 万 + 1.76 亿，×32，+10.5 亿 ≈ 8.03B。
- RoPE 好处要说全两点：只看相对距离 + 无位置表无长度硬上限；转 q 不转 Wq，因为 Wq 共用、位置是每个 token 自己的。（周测第 3 题）
- 手算注意力：softmax 里 **e⁰ = 1**；权重和为 1，输出是 v 的加权和。（周测第 4 题）
- 显存估算 = 权重（参数 × 2 字节）+ KV Cache（每 token 字节 × 长度 × 并发）+ 计算空间。70B、50×4K → 140 + 62.5 + ~12 ≈ 215 GB → 3 张 80GB。（周测第 5 题）
- **参数量**仍是最大弱项：分清参数（W，和 T 无关）与计算结果（q、k、T×T 分数、KV Cache，和 T 有关）。输出头不共享要加 V·d。
- 加起来等于 1 的是 softmax 后的**权重**，不是 v。
- RoPE 的 base 是转速底数，**不是**上下文长度。
- GQA vs MQA 权衡的是显存 vs **效果**。
- 选型回答还是缺**数字**：每个三段式都要先算出一个数。

## 复习题
1. 算力涨 100 倍，N、D 各涨几倍？为什么开源模型都“过度训练”？
2. 写出 LLaMA-2 7B 的参数量算式，说出和 12Ld²+Vd 的差别在哪。
3. 用两个 2 维向量说明 RoPE 为什么只和相对距离有关。
4. 超过训练长度为什么会崩？长上下文有哪几道坎？
5. 写出 KV Cache 公式，心算 LLaMA-3 8B 32K 上下文要多少显存。
6. 客户要 8B 模型 200 路并发、每路 8K，一张 80GB 卡够吗？还有哪些办法？
