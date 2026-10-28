# 大模型与 AI Agent 学习笔记

学习计划（12 周）：https://claude.ai/artifact/GUyzTsikooCCRA4nzdvtma

| 周 | 主题 | 笔记 | 小测 | 待回顾 |
|---|---|---|---|---|
| 1 | API 调用、多轮对话、Prompt、JSON、eval | [week01](week01-api-basics.md) | 练习全部完成 | chat2.py 截断作业 |
| 2 | 梯度下降、反向传播、Adam | [week02](week02-backprop-adam.md) | 3.5/5 | 参数 vs 学习率、zero_grad、手推反向传播 |
| 3 | 自注意力、Transformer block、交叉熵、迷你 GPT、GPT-2 | [week03](week03-transformer.md) | 3.2/5 | 参数估算公式、上下文长度 vs 参数 vs 显存、残差的作用 |
| 4 | Function Calling、ReAct 循环、Agent harness、MCP | [week04](week04-agent-react-mcp.md) | 3.3/5 | tool_call_id、Agent 循环两个结束条件、排查看工具日志、softmax e⁰=1 |

## 每篇笔记的固定结构
一句话总结 → 核心概念 → 关键代码 → 实验结果 → 我的提问与解答 → 易错点 → 复习题

## 每周的沉淀习惯
1. 学习过程中随时提问，不用自己记。
2. 每周小测讲评后，Claude 自动生成 `weekNN-主题.md` 放进这里，并更新上表。
3. 下一周开课前，先用上一周的“复习题”和“易错点”做 5 分钟回顾。
