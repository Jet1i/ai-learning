# ai-learning

我学习大模型与 AI Agent 的记录：每周的笔记、提问和代码。

- 学习笔记与总目录：[notes/README.md](notes/README.md)
- 12 周学习计划：https://claude.ai/artifact/GUyzTsikooCCRA4nzdvtma

## 代码
| 文件 | 周 | 内容 |
|---|---|---|
| chat.py | 1 | 第一次调用 DeepSeek API |
| chat2.py | 1 | 多轮对话（上下文） |
| chat3.py | 1 | 多轮对话练习 |
| stress.py | 1 | JSON 输出的批量评测（宽松版 vs 严格版） |
| gd1.py | 2 | 手算梯度的梯度下降 |
| autograd.py | 2 | 迷你自动求导（反向传播） |
| mlp_adam.py | 2 | 小神经网络，SGD vs Adam |

## 运行
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai
export DEEPSEEK_API_KEY="sk-..."   # 不要把 Key 写进代码
python chat.py
```
