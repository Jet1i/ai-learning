# ai-learning

我学习大模型与 AI Agent 的记录：每周的笔记、提问和代码。

- 学习笔记与总目录：[notes/README.md](notes/README.md)
- 12 周学习计划：https://claude.ai/artifact/GUyzTsikooCCRA4nzdvtma

## 代码（按周分文件夹）
| 文件 | 内容 |
|---|---|
| week1/chat.py | 第一次调用 DeepSeek API |
| week1/chat2.py | 多轮对话（上下文） |
| week1/chat3.py | 多轮对话练习 |
| week1/stress.py | JSON 输出的批量评测（宽松版 vs 严格版） |
| week2/gd1.py | 手算梯度的梯度下降 |
| week2/autograd.py | 迷你自动求导（反向传播） |
| week2/mlp_adam.py | 小神经网络，SGD vs Adam |
| week3/attention1.py | numpy 手写自注意力 |
| week3/block.py | 多头注意力 + Transformer block |
| week3/minigpt.py | torch 迷你 GPT（交叉熵训练） |
| week3/gpt2.py | 加载 GPT-2 生成文本 |
| week4/fc1.py | Function Calling：模型请求、代码执行 |
| week4/react1.py | 从零手写文本版 ReAct 循环 |
| week4/agent.py | 完整 Agent：原生 tools + 循环 + 记忆 |
| week4/mcp_server.py | 最小 MCP Server |
| week4/mcp_agent.py | 通过 MCP 调用工具的 Agent |
| week5/embed1.py | Embedding 与余弦相似度（bge-small-zh） |
| week5/rag1.py | 第一个 RAG：切块、检索、生成 |
| week5/kb_server.py | 知识库检索 MCP Server |
| week5/kb_agent.py | Agentic RAG：通过 MCP 检索的 Agent |
| week5/eval.py | 检索与回答评测（Hit@k、门槛扫描、LLM 裁判） |
| week5/project1/ | 项目 1：学习笔记问答 Agent（混合检索 + MCP），见其 README |
| week6/scaling.py | Scaling Laws：6ND 与 Chinchilla 最优分配 |
| week6/llama_block.py | LLaMA 块：RMSNorm、SwiGLU、参数量复算 |
| week6/rope.py | RoPE 旋转位置编码 |
| week6/gqa.py | KV Cache 与 GQA：显存与并发估算 |
| week7/roofline.py | 屋顶线：batch 实测 + GPU 搬运受限估算 |
| week7/flash.py | FlashAttention：online softmax + 分块注意力（numpy） |
| week7/paged.py | vLLM PagedAttention：预留 vs 分页的并发模拟 |
| week7/quant.py | 量化：int8 / int4、分组量化与误差 |
| week7/*.html | 本周交互 demo（屋顶线、Flash 拆解、vLLM 酒店、连续批处理、量化） |
| week8/moe.py | MoE：路由器（wx）+ top-k 专家 + 负载统计 |
| week8/balance.py | 负载均衡：强者越强 vs DeepSeek-V3 偏置；细粒度专家组合数 |
| week8/mla.py | MLA：只存压缩向量 c、吸收技巧验证、各模型 KV cache 账本 |
| week8/v3.py | DeepSeek-V3：逐矩阵算出 671B 总参数 / 37B 激活 + 部署与训练账 |
| week8/*.html | 本周交互 demo（MoE 路由、MLA 压缩） |
| week9/sft.py | SFT：GPT-2 客服微调（特殊 token、label −100、保存 sft_model） |
| week9/reward.py | 奖励模型：Bradley-Terry loss + 长度偏好被钻空子 |
| week9/rlhf.py | RLHF 玩具：REINFORCE + KL 绳子，β 扫描 |
| week9/ppo.py | PPO：clip 限速 + 价值模型分功劳 |
| week9/ppo_gpt2.py | 手写 PPO 跑真 GPT-2（IMDB 情感奖励） |
| week9/dpo.py | DPO：拒答训练，纯 DPO 失败 → 加 SFT 项 + "正确 ≻ 拒答"对修复 |
| week9/grpo.py | GRPO：组内优势 + 规则奖励（准确率 / 格式） |
| week9/*.html | 本周交互 demo（奖励模型、KL、PPO 逐 token、RLHF 全流程、DPO 修复、R1 流程） |
| week9/qa-log.md | 本周全部提问与解答 |
| week10/lora.py | LoRA 原理：B·A 低秩补丁、B=0 初始化、秩够不够的实验、真实模型参数量 |
| week10/lora_sft.py | 用 peft 给 ShopX 客服挂 LoRA 学新业务：学习率坑、遗忘与 replay、对比全参 |
| week10/merge_qlora.py | 合并 LoRA（W + s·B·A）+ 全参 / LoRA / QLoRA 训练显存估算 |
| week10/qa-log.md | 本周全部提问与解答 |
| week11/tco_api.py | API 账单：缓存命中 / 未命中 / 输出三种价、前缀缓存、多轮对话的真实成本 |
| week11/tco_gpu.py | 自建 GPU 成本：屋顶线算单卡容量、真实流量峰值 + 高可用、和 API 的盈亏平衡表 |
| week11/cloud-map.html | 云上 AI 4 层地图（MaaS / PaaS / IaaS / 私有化）+ 数据位置 5 档 + 场景选型 |
| week11/inject.py | 间接 Prompt 注入：天真 Agent vs 提示词防御 vs 代码护栏（MODEL=fake 演示被骗的模型） |
| week11/compliance.py | 多租户 RAG 隔离、正则脱敏（数字边界）、审计日志 |
| week11/compliance-map.html | 合规 8 站地图 + 法规速查 |
| week11/qa-log.md | 本周全部提问与解答 |

## 运行
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai mcp numpy torch transformers peft   # 第 5 周起用到 torch/transformers,第 10 周用 peft
export DEEPSEEK_API_KEY="sk-..."   # 不要把 Key 写进代码
python week1/chat.py
```
