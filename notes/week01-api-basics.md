---
week: 1
title: 调用大模型 API、多轮对话、Prompt 与结构化输出
date: 2026-10-27
tags: [api, token, context-window, prompt, json, eval, context-engineering]
code: [chat.py, chat2.py, extract.py, stress.py]
---

# 第 1 周：第一次调用大模型

## 一句话总结
模型本身没有记忆，它只知道你这一次发给它的内容；做产品要用约束和批量测试（eval）把失败率压到接近 0。

## 核心概念
- **OpenAI 兼容接口**：DeepSeek 兼容 OpenAI 格式，用 `openai` SDK，只改 `api_key`、`base_url`、`model` 三处。公司内部网关、Azure、百炼通常也是这样。
- **messages 三种角色**：`system`（设定身份和规则）、`user`（用户说的）、`assistant`（模型说过的）。
- **token 与计费**：token 是模型读写的最小单位，中文约 1 字 ≈ 0.6–1 token。按输入（prompt_tokens）和输出（completion_tokens）分别计费；缓存命中（prompt_cache_hit_tokens）更便宜。
- **多轮对话 = 每次重发全部历史**：所以输入 token 会一轮比一轮大。
- **上下文窗口**：一次最多能读的 token 数（DeepSeek 约 12.8 万）。太长会撞上限、变慢、中间信息容易被忽略。
- **三种应对长对话的办法**：只保留最近 N 轮；把旧对话压缩成摘要；存进数据库按需检索（长期记忆 / RAG，第 5 周）。
- **结构化输出三件套**：角色 + 任务；给出 JSON 格式样例；`response_format={"type":"json_object"}` + `temperature=0`。
- **eval（评测）**：不要相信一次跑通，要看大量调用下的失败率。
- **上下文工程**：往窗口里放什么、放多少、什么时候放，决定 Agent 好不好用。CLAUDE.md / AGENTS.md / Skills / 记忆都属于它。

## 关键代码
```python
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
# 只保留 system + 最近 5 轮（10 条）
messages = [messages[0]] + messages[1:][-10:]
```

## 实验结果
- `stress.py`：temperature=1.5 下各跑 10 次，**宽松版失败 9/10，严格版 0/10**。

## 我的提问与解答
**Q：为什么建议用虚拟环境？**
A：避免不同项目的库版本冲突；不碰 macOS 系统 Python；坏了删掉 `.venv` 重建即可，还能 `pip freeze > requirements.txt` 让别人复现环境。

**Q：在 Mac 上怎么写代码文件？**
A：用 VS Code 打开 `~/ai-learning` 新建 `.py`；或终端 `nano`。不要用“文本编辑”（富文本、会把引号换成弯引号）。

**Q：公司的模型 API 我能用吗？**
A：技术上多半能（改三处）。但先确认公司是否允许个人学习使用；公司 Key 和地址绝不能进 GitHub。作品集项目用自己的 Key。

**Q：报错 `UnicodeEncodeError: 'ascii' codec can't encode character '你'`？**
A：代码里的 `"你的Key"` 没换成真 Key，“你”字进了请求头。读报错先看最后一行。正确做法是把 Key 放环境变量 `DEEPSEEK_API_KEY`。

**Q：ChatGPT / Claude 的“永久记忆”是怎么做的？**
A：存：把值得记的信息写成一句话存进数据库；取：新对话时查出相关记忆塞进 system prompt。模型参数没变，变的只是发给它的内容。真正改模型要靠微调（第 9 周）。

**Q：删掉 `response_format` 和格式样例，为什么 json.loads 也没报错？**
A：prompt 里还有“只输出 JSON”，现代模型经过指令训练（RLHF）很听话；字段名是模型猜的；temperature=0 很稳定。但“通常可以”≠“保证可以”，上线一天上万次调用，1% 的失败就会让程序崩，所以要加约束并做 eval。

**Q：为什么要在不同 thread 里学习？多开对话能省 token 吗？**
A：能，每个对话上下文独立、更快更便宜、好找。代价是新对话不记得背景，这正是项目记忆、CLAUDE.md 要解决的。

**Q：AGENTS.md / CLAUDE.md 只是 system prompt 吗？**
A：技术上就是自动塞进上下文的文本，但它解决了：跨对话持久化、随代码进 Git 团队共享、按目录分层加载。AGENTS.md 是多家通用名，CLAUDE.md 是 Claude Code 专用。Skills 更进一步：平时只放简介，用到才加载全文。

## 易错点
- 删掉 `assistant` 那行后，模型**仍记得你的名字**（你说的话还在），忘掉的是**它自己说过的话**。
- Mac 上没有 `pip`，用 `python3 -m venv .venv` 后在 `(.venv)` 里才有 `pip`；每次新开终端要 `source .venv/bin/activate`。
- DeepSeek 的 JSON 模式要求 prompt 里出现 “json” 这个词。

## 复习题
1. 为什么多轮对话的输入 token 会越来越大？聊到 1000 轮会出现哪三类问题？
2. 只保留最近 5 轮有什么缺点？怎样既省 token 又不忘记名字？
3. 结构化输出的三件套是什么？各自防住了哪种失败？
4. 用一句话向客户解释什么是 eval，为什么上线前必须做。
5. 项目记忆、CLAUDE.md、RAG 三者有什么共同点？哪个会改变模型参数？

## 作业
- [ ] 给 `chat2.py` 加“只保留最近 5 轮”，观察 token 不再无限增长。
