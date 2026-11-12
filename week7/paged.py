"""第 7 周 L3：vLLM 的 PagedAttention —— 用 Python 模拟"KV Cache 怎么分显存"

运行：python week7/paged.py

场景（接复习题 3）：13B 模型 fp16，一张 80GB 卡
  - 权重 26 GB，计算空间 4 GB → 留给 KV Cache 约 50 GB
  - 每个 token 的 KV = 800 KB → 显存里一共能放 64000 个 token 的 KV
  - 500 个请求排队，每个提示词 100~1000 token，回答 20~1500 token（生成完才知道多长）
  - 最大长度 4096

两种分法：
  朴素：每个请求一进来就按最大长度 4096 预留（因为不知道它会说多长）
  分页：每页 16 个 token，用满一页再给下一页（vLLM 默认 block_size = 16）
每一步 = 所有在跑的请求各生成 1 个 token（一个 batch）。看谁先把 500 个请求做完。
"""
import random

random.seed(0)
KV_TOKENS = 64_000        # 显存里能放多少个 token 的 KV
MAX_LEN = 4096
PAGE = 16
N_REQ = 500

reqs = [dict(prompt=random.randint(100, 1000), out=random.randint(20, 1500)) for _ in range(N_REQ)]


MAX_SEQS = 256            # vLLM 默认最多同时跑 256 个请求


def simulate(mode):
    waiting = list(range(N_REQ))
    length = {i: reqs[i]["prompt"] for i in range(N_REQ)}   # 每个请求当前有多少 token
    running = []              # 在跑的请求（按进来的先后）
    reserved = 0              # 被占用（预留或分配）的格子数
    step, preempt = 0, 0
    batch_sizes, util = [], []

    def need(n):              # 有 n 个 token 时要占多少格子
        if mode == "朴素":
            return MAX_LEN
        return -(-n // PAGE) * PAGE          # 向上取整到整页

    while waiting or running:
        # 1) 有空位就放人进来（连续批处理：不用等整批结束）
        while waiting and len(running) < MAX_SEQS:
            i = waiting[0]
            if reserved + need(length[i]) > KV_TOKENS:
                break
            waiting.pop(0)
            running.append(i)
            reserved += need(length[i])
        # 2) 每个在跑的请求生成 1 个 token
        step += 1
        batch_sizes.append(len(running))
        util.append(sum(length[i] for i in running) / KV_TOKENS)
        for i in list(running):
            if i not in running:
                continue
            extra = need(length[i] + 1) - need(length[i])  # 分页时可能要新的一页
            while reserved + extra > KV_TOKENS and len(running) > 1:
                victim = running.pop()                     # 没页了：把最后进来的请求请出去
                reserved -= need(length[victim])           # （vLLM 的"抢占"，它之后重新排队）
                waiting.insert(0, victim)                  # 简化：回来时保留已生成的 token，不算重算的时间
                preempt += 1
                if victim == i:
                    break
            if i not in running:
                continue
            reserved += extra
            length[i] += 1
            if length[i] >= reqs[i]["prompt"] + reqs[i]["out"]:
                reserved -= need(length[i])                # 做完了，格子还回去
                running.remove(i)
    return step, sum(batch_sizes) / len(batch_sizes), max(batch_sizes), sum(util) / len(util), preempt


print(f"显存可放 {KV_TOKENS:,} 个 token 的 KV；{N_REQ} 个请求；最大长度 {MAX_LEN}\n")
print(f"{'分法':<6}{'总步数':>8}{'平均并发':>10}{'最大并发':>10}{'显存真用率':>12}{'抢占次数':>10}")
res = {}
for mode in ["朴素", "分页"]:
    steps, avg_b, max_b, u, pre = simulate(mode)
    res[mode] = steps
    print(f"{mode:<6}{steps:>8,}{avg_b:>10.1f}{max_b:>10}{u:>11.0%}{pre:>10}")
print(f"\n分页比朴素快 {res['朴素'] / res['分页']:.1f} 倍（同一张卡、同样的请求）")
print("""
看什么：
  - 朴素：每人占 4096 格，64000 ÷ 4096 = 最多 15 人同时跑，平均真用的格子很少
  - 分页：按实际长度占格子，同时能跑几十人 → batch 大 → 每一步虽然时间差不多（L1：搬运受限），
    但一步出的 token 多得多
""")
