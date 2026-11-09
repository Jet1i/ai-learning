"""
项目 1 · 评测：三种检索方式（vector / bm25 / hybrid）在同一批题上比一比
  python week5/project1/eval.py
测试题在 TESTS 里：expect 写"出处里必须包含的关键字"，比如 "week4" 表示第 4 周的笔记或问答记录都算对，None 表示笔记里没有。
👉 你的任务：再加 5 道你自己会问的题（见 README）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re

import kb

TESTS = [
    {"q": "为什么多轮对话的 token 会越来越多", "expect": "week1"},
    {"q": "zero_grad 为什么要放在每一步训练里", "expect": "week2"},
    {"q": "学习率和模型参数有什么区别", "expect": "week2"},
    {"q": "注意力分数为什么要除以根号 d", "expect": "week3"},
    {"q": "残差连接有什么用", "expect": "week3"},
    {"q": "GPT-2 的参数量怎么估算", "expect": "week3"},
    {"q": "tool 消息要用哪个字段和调用对上", "expect": "week4"},
    {"q": "去掉 stop 以后模型会怎样", "expect": "week4"},
    {"q": "agent 循环有哪两个停止条件", "expect": "week4"},
    {"q": "向量数据库是不是把数据变成结构化了", "expect": "week5"},
    {"q": "CLS token 是什么", "expect": "week5"},
    {"q": "stdio 和 HTTP 两种 MCP 传输有什么区别", "expect": "week5"},
    {"q": "怎么用 Docker 部署 vLLM", "expect": None},
    {"q": "LoRA 微调的原理是什么", "expect": None},
]


def main(k=4):
    idx = kb.get_index()
    print(f"知识库 {len(idx.chunks)} 块，测试题 {len(TESTS)} 道，门槛 {kb.THRESHOLD}\n")
    ans = [t for t in TESTS if t["expect"]]
    print(f"{'问题':<30}{'最高相似度':>8}   vector  bm25  hybrid（正确出处的名次）")
    total = {"vector": [], "bm25": [], "hybrid": []}
    for t in TESTS:
        best = idx.best_cos(t["q"])
        cells = []
        for mode in total:
            # 出处统一写法：notes/week04-… 和 week4/qa-log 都算"week4"
            srcs = [re.sub(r"week0(\d)", r"week\1", c["source"]) for c, _ in idx.search(t["q"], k=k, mode=mode)]
            rank = next((i + 1 for i, s in enumerate(srcs) if t["expect"] and t["expect"] in s), None)
            total[mode].append(rank)
            cells.append("-" if not t["expect"] else (str(rank) if rank else "✗"))
        flag = "" if t["expect"] else ("  ✅挡住" if best < kb.THRESHOLD else "  ⚠️没挡住")
        print(f"{t['q']:<30}{best:>8.3f}   {cells[0]:^6}  {cells[1]:^4}  {cells[2]:^6}{flag}")

    print()
    for mode, ranks in total.items():
        r = [x for x, t in zip(ranks, TESTS) if t["expect"]]
        print(f"{mode:<7} Hit@1 {sum(x == 1 for x in r)}/{len(ans)}   Hit@{k} {sum(x is not None for x in r)}/{len(ans)}")
    print(f"能答的题被门槛误挡：{sum(idx.best_cos(t['q']) < kb.THRESHOLD for t in ans)}/{len(ans)}")


if __name__ == "__main__":
    main()
