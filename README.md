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

## 运行
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai
export DEEPSEEK_API_KEY="sk-..."   # 不要把 Key 写进代码
python week1/chat.py
```
