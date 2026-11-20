"""
第 11 周 L2:自建 GPU 账本 —— 什么时候自建比调 API 便宜?
运行:python week11/tco_gpu.py

所有价格、效率都是"示例假设",写在最上面,改数字就能重算。真做方案时用客户拿到的报价。
"""
import math

# ---------- 场景:和 L1 一样的客服 ----------
IN_TOK, OUT_TOK = 3600, 300          # 每次请求:输入 3600、输出 300
API_PER_REQ = (IN_TOK * 2.0 + OUT_TOK * 3.0) / 1e6    # DeepSeek API 不算缓存:约 0.0081 元/次
FLAG_PER_REQ = (IN_TOK * 22 + OUT_TOK * 108) / 1e6    # 旗舰闭源 API(示例,约 $3/$15 一档):约 0.11 元/次

# ---------- 两种自建方案 ----------
# 吞吐用第 6~8 周的 roofline 粗估:
#   decode 一步 = 把权重 + 这批请求的 KV cache 从显存读一遍(显存带宽瓶颈),一步出 batch 个 token
#   prefill = 算力瓶颈,每个 token 约 2 × 激活参数 FLOPs
#   EFF = 实际能达到理论值的比例(调度、通信、长短不齐……),先给 0.4
DEPLOYS = {
    "8B 模型 @ 1×A100-80G": dict(
        gpus=1, price_per_gpu_hour=25,      # 元/卡时(按需租用,示例)
        weight_GB=16, active_params=8e9,    # fp16 权重 16 GB
        kv_KB_per_token=128,                # LLaMA-3 8B GQA(第 6 周)
        bw_TBs=2.0, tflops=312, batch=64),
    "32B 模型 int8 @ 1×H200": dict(
        gpus=1, price_per_gpu_hour=50,
        weight_GB=32, active_params=32e9,   # int8 权重 32 GB
        kv_KB_per_token=256,                # 64 层 × 8 个 KV 头 × 128 维 × 2(K、V)× 2 字节
        bw_TBs=4.8, tflops=989, batch=64),
    "DeepSeek-V3 671B @ 8×H200": dict(
        gpus=8, price_per_gpu_hour=50,
        weight_GB=671, active_params=37e9,  # FP8 权重 671 GB,每个 token 激活 37B(第 8 周)
        kv_KB_per_token=69,                 # MLA(第 8 周)
        bw_TBs=4.8 * 8, tflops=1979 * 8, batch=256),
}
EFF = 0.4
OPS_PER_MONTH = 30_000               # 运维:半个到一个工程师(元/月,示例)

def capacity(d):
    """一套机器每秒能服务多少个请求(理论 × EFF)。"""
    ctx = IN_TOK + OUT_TOK / 2                                     # 解码时平均上下文长度
    bytes_step = d["weight_GB"] * 1e9 + d["batch"] * ctx * d["kv_KB_per_token"] * 1e3
    step_s = bytes_step / (d["bw_TBs"] * 1e12)
    decode_tps = d["batch"] / step_s                               # 每秒输出多少 token
    prefill_tps = d["tflops"] * 1e12 / (2 * d["active_params"])    # 每秒能读入多少 token
    gpu_s_per_req = IN_TOK / prefill_tps + OUT_TOK / decode_tps     # 一次请求占用整套机器多少秒
    return EFF / gpu_s_per_req, decode_tps * EFF, prefill_tps * EFF

def node_month(d):
    return d["gpus"] * d["price_per_gpu_hour"] * 24 * 30

# ======================================================================
# 第 1 部分:一套机器的能力与"每次请求成本"——天真算法(故意有错)
# ======================================================================
def part1():
    print("[第 1 部分] 一套机器能扛多少、每次请求多少钱(天真:假设 24 小时满载)")
    for name, d in DEPLOYS.items():
        rps, dec, pre = capacity(d)
        naive_per_req = node_month(d) / (rps * 86400 * 30)
        print(f"  {name}")
        print(f"    解码 {dec:,.0f} tok/s,预填充 {pre:,.0f} tok/s → 约 {rps:.1f} 次请求/秒"
              f"(每天最多 {rps*86400/1e4:.0f} 万次)")
        print(f"    租金 {node_month(d)/1e4:.1f} 万元/月 → 满载时每次 {naive_per_req*100:.3f} 分钱"
              f"(API {API_PER_REQ*100:.3f} 分钱)")
    print()

# ======================================================================
# 第 2 部分:真实流量 —— 白天高峰、夜里空闲,机器要按高峰买
# ======================================================================
HOURLY = [0.2,0.1,0.1,0.1,0.1,0.2,0.4,0.8,1.3,1.7,1.9,1.8,   # 0~11 点
          1.5,1.6,1.8,1.9,1.7,1.4,1.2,1.3,1.4,1.2,0.8,0.5]   # 12~23 点;平均 = 1
HA_MIN = 2   # 高可用:至少 2 套,坏一套还能服务

def selfhost_month(d, req_per_day):
    rps, _, _ = capacity(d)
    avg_rps = req_per_day / 86400
    peak_rps = avg_rps * max(HOURLY) / (sum(HOURLY) / 24)
    nodes = max(HA_MIN, math.ceil(peak_rps / rps))
    util = avg_rps / (nodes * rps)
    return nodes * node_month(d) + OPS_PER_MONTH, nodes, util

def part2(req_per_day=1_000_000):
    print(f"[第 2 部分] 真实流量:每天 {req_per_day/1e4:.0f} 万次,高峰是平均的 {max(HOURLY):.1f} 倍,至少 {HA_MIN} 套")
    api = API_PER_REQ * req_per_day * 30
    print(f"  API(DeepSeek,不算缓存)       {api/1e4:7.1f} 万元/月")
    for name, d in DEPLOYS.items():
        cost, nodes, util = selfhost_month(d, req_per_day)
        print(f"  {name:<26} {cost/1e4:7.1f} 万元/月  ({nodes} 套,平均利用率 {util:.0%},"
              f"每次 {cost/(req_per_day*30)*100:.3f} 分钱)")
    print()

# ======================================================================
# 第 3 部分:盈亏平衡 —— 量多大时自建更便宜?
# ======================================================================
def part3():
    print("[第 3 部分] 不同日请求量下的月成本(万元)")
    names = list(DEPLOYS)
    print(f"  {'每天请求':>10} {'DS-API':>8} {'旗舰API':>8} " + " ".join(f"{n[:12]:>14}" for n in names))
    for q in [10_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000]:
        api = API_PER_REQ * q * 30
        row = [selfhost_month(DEPLOYS[n], q)[0] for n in names]
        flag = FLAG_PER_REQ * q * 30
        print(f"  {q:>10,} {api/1e4:8.1f} {flag/1e4:8.1f} " + " ".join(f"{c/1e4:14.1f}" for c in row))
    print("  注意:8B 和 671B 不是同一个水平的模型,两列不能直接比谁便宜。\n")

if __name__ == "__main__":
    part1(); part2(); part3()
