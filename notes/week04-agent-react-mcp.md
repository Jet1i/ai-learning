# 第 4 周：Function Calling、ReAct 循环、MCP

日期：2026-10-28 · 代码：`week4/` · 周测：约 3.3/5

## 一句话总结
模型自己什么都执行不了，它只会输出"我想调用哪个工具、参数是什么"；**Agent = 模型 + 工具 + while 循环 + messages 记忆**，包在模型外面的这层代码叫 harness；MCP 把工具搬进独立的 Server，用统一协议让任何 Agent 都能发现和调用。

## 核心概念

### 1. Function Calling（fc1.py）
- 你给模型一份**工具说明书**（name + description + 参数 JSON Schema），模型看不到函数代码。
- 模型需要工具时，返回的不是答案，而是 `tool_calls`（content 为空），其中 `arguments` 是一段 JSON 字符串，要 `json.loads`。
- **你的代码**执行函数，把结果用 `{"role": "tool", "tool_call_id": <对应调用的 id>, "content": 结果}` 发回去，再请求一次。
- 模型一次可能返回多个 tool_calls（并行调用，比如同时查北京和上海）。
- 调不调工具、调哪个，全由模型根据 description 决定。description 写得好不好，直接决定 Agent 靠不靠谱。
- 没有 system prompt 时，模型的"身份"来自训练语料的统计印象，DeepSeek 会自称 Claude。产品里要用 system prompt 定身份。

### 2. ReAct 循环（react1.py）
- ReAct = Reason + Act（2022 年论文）：`Thought → Action → Observation → … → Final Answer`。
- 文本版：靠 prompt 约定格式，正则解析 `Action: 工具名[参数]`，代码执行后把 `Observation:` 拼回去。
- `stop=["Observation:"]`：服务器在生成时检查，模型一写到这个词就被掐断，相当于 minigpt 生成循环里加了 `if 出现 "Observation:": break`。
- 模型的一次回答是**一整段连续生成的文本**，中途无法暂停等外部结果。工具只在模型停下、代码接手时才真正执行。

### 3. 完整 Agent（agent.py）
```
messages = [system]                      # 放在对话循环外 → 有记忆
for step in range(MAX_STEPS):            # 结束条件 2：步数上限
    msg = llm(messages, tools)
    messages.append(msg)
    if not msg.tool_calls: return msg.content   # 结束条件 1：不再要工具
    for call in msg.tool_calls:
        result = 执行(call)               # 出错也把错误信息发回去
        messages.append(role="tool", tool_call_id=call.id, content=result)
```
- 原生 tools 不需要 stop：模型输出 tool_calls 后服务器就结束这次回答（`finish_reason = "tool_calls"`），等于内置了 stop。
- **harness**：包在模型外的代码（循环、执行工具、管理 messages、步数上限）。产品级 harness（如 Claude Code）还有：危险操作先确认、上下文压缩、日志、重试、流式输出。
- 模型每次请求都是失忆的，"记忆"全靠 harness 把历史 messages 一起发过去。

### 4. MCP（mcp_server.py + mcp_agent.py）
- Model Context Protocol：Agent（Host 里的 Client）和工具（Server）之间的统一协议，消息格式 JSON-RPC，本地走 stdin/stdout，远程走 HTTP。
- 流程：启动 Server 子进程 → `initialize` 握手 → `tools/list` 发现工具 → 翻译成 OpenAI tools 格式 → 模型返回 tool_calls → `tools/call` 发给 Server 执行。
- Server 端不用写 JSON Schema：`city: str` 类型注解 + docstring 由 SDK 自动生成说明书。
- 价值：**工具和 Agent 解耦**。工具方写一次 Server，Claude、Cursor、自己的 Agent 都能接；N 个应用 × M 个工具的胶水代码变成 N + M。
- 注意：mcp SDK 2.x 里 `FastMCP` 改名 `MCPServer`（`from mcp.server import MCPServer`），客户端用 `from mcp import Client`。

## 关键代码
| 文件 | 内容 |
|---|---|
| week4/fc1.py | 一轮 function calling：天气 + 计算器 |
| week4/react1.py | 文本版 ReAct：prompt 格式 + 正则 + stop |
| week4/agent.py | 原生 tools + while 循环 + 多轮记忆，4 个工具（天气、计算、时间、读笔记） |
| week4/mcp_server.py | 最小 MCP Server（3 个工具） |
| week4/mcp_agent.py | 通过 MCP 发现和调用工具的 Agent，和 agent.py 只差两处 |

动画演示：[Function Calling 往返](function-calling-demo.html)（[在线版](https://claude.ai/artifact/4JsFCrzubsd5Jh9E2d7Fqa)）、[MCP 握手与调用](mcp-demo.html)（[在线版](https://claude.ai/artifact/NhoYYtDM36Apq1iykdeyLX)）

## 实验结果
- fc1.py：天气、计算走工具，自我介绍直接回答。问"北京和上海哪个更暖和、温差几度"时最终回答为空：第 1 轮并行调了 2 次天气，第 2 轮模型还想调 calculator，但单轮代码直接打印了空 content → 需要循环。
- react1.py 正常：模型有时不调 calculator，自己心算 16−12（工具是可选的）。
- 只删 stop：模型仍然在 Action 后停下，因为 prompt 里写了"每次只写一个 Action，然后停下"（指令微调的模型很听话）。
- 同时删 stop 和那半句 prompt：模型一轮写了两个 Action 和 Final Answer，上海被编成 18°C，**get_weather 一次都没执行**（代码先检测到 Final Answer 就 return 了）。
- agent.py：温差问题正确调用了天气 + 计算器；能接上追问；把 messages 挪进循环后记不住。Agent 还主动提出"要不要追加到笔记"，但它根本没有写文件的工具。
- mcp_agent.py：给 Server 加 get_time，Agent 代码不改就能用。

## 我的提问与解答
- **删了 stop，模型怎么"知道"要写 Observation 和 Final Answer？** 它不知道，只是在预测下一个 token。system prompt 里有格式，写完 Action 后最可能的下一个词就是 "Observation:"，后面的天气按统计印象编。
- **那 Action（function call）还有什么用？** 模型知识冻结在训练时，拿不到实时数据。Action 是它发给代码的"请求单"，代码真去执行，Observation 才是真的。没 stop 的问题是"不等结果"，不是 Action 没用。
- **为什么去掉 stop 后结果还是对的？** prompt 里"停下等待"那句让模型自己停了。prompt 是软保险（靠自觉），stop 是硬保险（服务器强制），产品里两道都要。
- **一轮就输出了，是 decode 太快吗？** 不是。一次回答是一整段连续文本，生成过程中没法等外部结果；代码只有拿到整段文字后才开始执行。
- **多轮 function call 要写 stop 吗？** 原生 tools 不用，文本版 ReAct 才要。
- **Agent 为什么主动问要不要帮我写笔记？** 训练出来的"贴心助手"习惯。模型说的话 ≠ 它能做的事，能力只由 tools 列表决定。
- **我这里已经有 harness 了吗？** 有，agent.py 就是迷你 harness。
- **Function Calling 和 MCP 最主要的区别？** 它们不是二选一，管的是两段不同的连接：Function Calling 是"模型 ↔ 你的代码"（模型怎么表达要调工具），MCP 是"你的代码 ↔ 工具"（工具住在哪、怎么被发现和调用）。mcp_agent.py 两个都用了。MCP 的价值是解耦、复用（不重复造轮子）、动态发现、跨厂商标准、工具可远程部署并单独管权限。

## 易错点
- 发回结果时要对上的是 **`tool_call_id`**（调用的 id），不是工具名；role 是 `"tool"` 不是 `"tools"`。
- Agent 循环有**两个**结束条件：模型不再返回 tool_calls（完成），和步数上限（防死循环烧钱）。只写 `while` 不写结束条件是周测失分点。
- "模型自己等工具结果"不准确：模型不会等，是服务器在 tool_calls 处结束本次回答，由**你的代码**执行后再发起下一次请求。
- 排查 Agent bug 要看**工具执行日志**，不能只看模型说了什么（"说订了，其实没订"）。
- 复习遗留：softmax 中 e⁰ = 1，不是 0；softmax 永远不会输出正好 0。

## 复习题
1. 模型返回 tool_calls 后，谁执行函数？发回结果的消息长什么样？
2. 写出 Agent 循环伪代码，标出两个结束条件。
3. `stop=["Observation:"]` 的作用是什么？原生 tools 为什么不需要？
4. 删掉 stop 和"停下等待"的提示后，为什么 get_weather 一次都没执行？
5. 把 messages 初始化放进对话循环里会发生什么？为什么？
6. "Agent 说订好了会议室，其实没订"：两个可能原因和对应的预防办法。
7. MCP 解决了什么问题？`tools/list` 和 `tools/call` 分别做什么？
8. 售前一句话：客户有 CRM、工单、知识库，怎么用 MCP 设计 AI 助手？
