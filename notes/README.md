# 大模型与 AI Agent 学习笔记

学习计划（12 周）：https://claude.ai/artifact/GUyzTsikooCCRA4nzdvtma

| 周 | 主题 | 笔记 | 小测 | 待回顾 |
|---|---|---|---|---|
| 1 | API 调用、多轮对话、Prompt、JSON、eval | [week01](week01-api-basics.md) | 练习全部完成 | chat2.py 截断作业 |
| 2 | 梯度下降、反向传播、Adam | [week02](week02-backprop-adam.md) | 3.5/5 | 参数 vs 学习率、zero_grad、手推反向传播 |
| 3 | 自注意力、Transformer block、交叉熵、迷你 GPT、GPT-2 | [week03](week03-transformer.md) | 3.2/5 | 参数估算公式、上下文长度 vs 参数 vs 显存、残差的作用 |
| 4 | Function Calling、ReAct 循环、Agent harness、MCP | [week04](week04-agent-react-mcp.md) | 3.3/5 | tool_call_id、Agent 循环两个结束条件、排查看工具日志、softmax e⁰=1 |
| 5 | Embedding、RAG、检索 MCP Server、评测、项目 1 | [week05](week05-rag-mcp-eval.md) · [项目 1](../week5/project1/README.md) | 3.6/5 | RAG 最后一步是生成、门槛只防跑题、选型回答用数据 + 替代方案 |
| 6 | Scaling Laws、LLaMA 架构（RMSNorm/SwiGLU）、RoPE、GQA 与 KV Cache | [week06](week06-modern-arch.md) | 约 1/5 | 参数量四步算法、显存估算（权重 + KV Cache）、算力 ×100 → N、D 各 ×10、e⁰=1 |
| 7 | 推理优化：屋顶线、FlashAttention、vLLM（PagedAttention/连续批处理/chunked prefill）、量化 | [week07](week07-inference-optimization.md) | 约 1.8/5 | softmax 最后除总和、售前题先算数字、连续批处理 vs chunked prefill、量化快是因为搬得少 |
| 8 | MoE（路由器、负载均衡偏置、DeepSeekMoE）、MLA（压缩 KV + 吸收技巧 + 解耦 RoPE）、DeepSeek-V3 参数与部署账 | [week08](week08-moe-mla-deepseek.md) | 约 3.0/5 | 数字算到底（128K=131072、int4=0.5 字节）、MoE 省算力不省显存、权重 vs KV cache、售前回答带数字 |
| 9 | SFT、奖励模型、RLHF/PPO（KL 绳子、GAE、clip）、DPO（失败与修复）、GRPO、DeepSeek-R1 与蒸馏 | [week09](week09-rlhf-dpo-grpo-r1.md) | 约 2.7/5 | 答题写机制不只写结论、训练谁冻结谁、DPO loss 低 ≠ 行为对（看生成指标）、GRPO 组平均是基线不是逐 token 功劳、显存别忘梯度和 Adam |
| 10 | LoRA 原理与实操（学习率、遗忘与 replay）、合并、QLoRA 显存、Prompt / RAG / 微调选型 | [week10](week10-lora-selection.md) | 约 2.8/5 | 冻住 W 也会遗忘（补丁对所有输入生效）、LoRA 快不了几百倍但省显存、量化 ≠ 低秩、判断瓶颈要控制变量、选型写理由和风险 |
| 11 | TCO（API vs 自建）、GPU 选型、云上 4 层、Prompt 注入与代码护栏、租户隔离/脱敏/审计、项目 2 方案书 | [week11](week11-tco-security.md) | 约 1.7/5 | 写依据和数字、prefill 看算力、私有化按峰值买、自建便宜要说跟谁比、三要素漏不可信内容 |
| 12 | 多模态（图片 token、CLIP vs VLM）、多 Agent（拆不拆、交接）、上线监控（P99、分类别、灰度样本、回归）、客户版方案书 + 演示 | [week12](week12-multimodal-multiagent-launch.md) · [客户版方案书](https://claude.ai/artifact/LYNdk3P7pT6V3iHWNTnvrB) | 约 2.5/5 | 每句结论跟依据、合规检查在回答之后、防呆不是格式校验、小样本灰度不可靠、不承诺模型不犯错 |

## 每篇笔记的固定结构
一句话总结 → 核心概念 → 关键代码 → 实验结果 → 我的提问与解答 → 易错点 → 复习题

## 每周的沉淀习惯
1. 学习过程中随时提问，不用自己记。
2. 每周小测讲评后，Claude 自动生成 `weekNN-主题.md` 放进这里，并更新上表。
3. 下一周开课前，先用上一周的“复习题”和“易错点”做 5 分钟回顾。
