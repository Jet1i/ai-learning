# 第 4 周 Q&A 记录

## 复习（2026-10-28）
- 参数估算 GPT-2 small：答 1.24 亿 ✅
- 上下文 1024→4096 参数变不变：答"不变，计算资源变大" ✅（补充：GPT-2 可学习位置表在 block 外 +236 万；KV Cache 和 n² 计算变大）
- softmax([2,1,0])：答 0.74 0.26 0 ❌ → e⁰=1，正确 0.67 0.24 0.09；softmax 不会输出正好 0

## L1 Function Calling（fc1.py）
- 三个问题：天气、计算走工具，自我介绍直接答 ✅
- 练习 2：答"模型先判断有没有 function 可以 call" → 补充：判断依据只有 name + description + 参数 schema 文字，看不到代码
- 练习 3："北京和上海哪个暖和"最终回答为空 → 第 1 轮并行返回 2 个 tool_calls；第 2 轮模型还想调 calculator，返回 tool_calls、content 为空；fc1.py 只处理一轮 → 需要 while 循环（= Agent 本质）
- 问：模型自称 Claude？→ 没有 system prompt 时身份来自训练语料统计印象，会认错；产品里用 system prompt 定身份

## L2 ReAct（react1.py）
- 问：为什么没调计算器就知道温差？→ 工具是可选的，模型自己决定；简单算术会心算，大数常算错（按 token 猜）→ prompt/description 里写"必须用 calculator"
- 删掉 stop：答"调用了 action"（不完整）→ 讲解：模型会自己续写 Observation 甚至 Final Answer，编造工具结果；stop 逼它停下等真数据
- max_steps=2：超过最大步数 ✅ → 上限防死循环、控成本（步数 + 超时 + token 预算）
- 追问：删了 stop，模型怎么"知道"要写 Observation 和 Final Answer？→ 它不知道，只是在预测下一个 token：system prompt 里有格式，写完 Action 后最可能的下一个词就是 "Observation:"，后面的天气按统计印象编；stop = 生成循环里的 `if 出现 "Observation:": break`（服务器端检查）。原生 tools 更稳：模型被训练成输出 tool_calls 就停
- 追问：那 Action（function call）有什么用？→ 模型知识冻结、拿不到真数据；Action 是模型发给代码的"请求单"，代码真去执行，Observation 才是真的。没 stop 的问题是"不等结果"，不是 Action 没用。Action = 模型伸向真实世界的手
- 追问：去掉 stop，Observation 还是对的，还会 function call？→ 纠正之前说法："可能"编，不是"一定"。system prompt 里"每次只写一个 Action，然后停下"让指令微调过的模型自己停了，Observation 是代码打印的真结果。prompt = 软保险（靠自觉），stop = 硬保险（服务器强制），产品里两道都要。实验：同时删 stop 和那半句 prompt
- 实验（删 stop + 删"停下"提示）：一轮里写了 2 个 Action + Final Answer，上海编成 18°C。Jet 问"是不是 decode 太快"？→ 不是：一次回答是一整段连续文本，生成中无法暂停等结果；代码先检测到 Final Answer 就 return，get_weather 一次都没执行。"调用"只在模型停下、代码接手时发生。排查 Agent bug 要看工具执行日志，不能只看模型说法
- 问：多轮 function call 要记得写 stop 吗？→ 原生 tools 不用：服务器看到 tool_calls 就结束（finish_reason="tool_calls"），等于内置 stop；只需 while 循环 + 最大步数。文本版 ReAct 才必须写 stop

## L3 完整 Agent（agent.py）
- 练习 1：温差问题工具都用了，回答正确 ✅
- 练习 2：能记住，因为 messages append 了 ✅
- 练习 3：messages 挪进循环 → 记不住 ✅
- 问：Agent 为什么主动问"要不要追加到 week03 笔记"？→ 训练出来的助手习惯，但它没有写文件工具；模型说的话 ≠ 能做的事，能力只由 tools 决定（会假装"已追加"）
- 问：我这里已经有 harness 了吗？→ 有，agent.py 就是迷你 harness（循环 + 执行工具 + 管理 messages + 步数上限）；产品级 harness 还有权限确认、上下文压缩、日志、重试、流式

## L4 MCP（mcp_server.py + mcp_agent.py）
- 练习 1：工具列表来自 MCP Server 子进程 ✅
- 练习 2：Server 加 get_time，Agent 不改也会调用 ✅
- 练习 3：agent.py 要自己写函数；MCP 是统一协议，写好的工具拿来就能调 ✅（补充：工具与 Agent 解耦、动态发现）

## 周测（09:40 发出）
1. tool_calls 之后谁做什么、role/字段
2. Agent 循环伪代码 + 两种结束条件
3. stop 的作用、原生 tools 为何不需要
4. "说订了会议室其实没订"的两个原因 + 排查
5. 售前：CRM/工单/知识库 + MCP 方案一两句话
- 作答（09:42）：1 代码执行、role tools、"tool 名字要对上"（应为 tool_call_id，role="tool"）½；2 while(reason observation action)，缺两个结束条件 ½；3 防止直接生成结果 ✅（"模型自己等待"不准确：服务器在 tool_calls 处结束，代码执行后再请求）；4 幻觉、MCP 返回失败，缺排查/预防 ½；5 解耦，加功能、换 LLM 方便 ✅（补：每个系统一个 Server，各自管权限）。约 3.3/5
