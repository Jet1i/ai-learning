"""
第 5 周 L4：给 RAG 打分 —— 用一批测试题，把"感觉还行"变成数字
运行:
  python week5/eval.py            只评检索（不花钱，几秒钟）
  python week5/eval.py --answer   再评回答（每题调用 DeepSeek 2 次：1 次回答 + 1 次当裁判）

评测分两层：
  1. 检索评测：该找的那一块，有没有排进前 k？不该答的问题，有没有被门槛挡住？
  2. 回答评测：① 关键词检查（便宜、死板） ② LLM 当裁判（灵活、要花钱，自己也可能判错）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import rag1

# ---------- 测试集：问题 + 应该命中的小节 + 回答里必须出现的关键信息 ----------
# expect=None 表示知识库里没有答案，正确行为是拒答
# facts 里每一项是一组"任选其一"的写法
TESTS = [
    {"q": "银行卡退款多久能到账", "expect": "计费与退款.md > 退款到账时间", "facts": [("5",), ("7",)]},
    {"q": "买了服务器后悔了还能退吗", "expect": "计费与退款.md > 退款规则", "facts": [("5 天", "5天", "五天")]},
    {"q": "欠费了资源会被删掉吗", "expect": "计费与退款.md > 欠费处理", "facts": [("7 天", "7天")]},
    {"q": "开发票最少要多少钱", "expect": "计费与退款.md > 发票", "facts": [("100",)]},
    {"q": "密码输错几次会被锁", "expect": "账号与安全.md > 登录方式", "facts": [("5 次", "5次"), ("30",)]},
    {"q": "手机号和邮箱都不能用了怎么找回账号", "expect": "账号与安全.md > 找回密码", "facts": [("申诉", "身份证明")]},
    {"q": "一个主账号能建多少个子账号", "expect": "账号与安全.md > 子账号与权限", "facts": [("200",)]},
    {"q": "AccessKey 不小心传到 GitHub 上了怎么办", "expect": "账号与安全.md > 访问密钥", "facts": [("禁用",)]},
    {"q": "跑 MySQL 数据库选什么规格", "expect": "云服务器使用.md > 实例规格", "facts": [("内存型",)]},
    {"q": "ssh 登不上服务器", "expect": "云服务器使用.md > 无法远程连接", "facts": [("22",)]},
    {"q": "不同地域的服务器内网能互通吗", "expect": "云服务器使用.md > 地域与可用区", "facts": [("不互通", "不能互通", "隔离", "不通")]},
    {"q": "服务器配置能降低吗", "expect": "云服务器使用.md > 升级配置", "facts": [("续费",)]},
    {"q": "你们支持 GPU 实例吗", "expect": None},
    {"q": "公司年假有几天", "expect": None},
    {"q": "对象存储怎么收费", "expect": None},
    {"q": "怎么配置 CDN 加速", "expect": None},
]


# ---------- 1. 检索评测 ----------
def eval_retrieval(k=3):
    rows = []
    for t in TESTS:
        hits = rag1.search(t["q"], k=k)
        srcs = [c["source"] for _, c in hits]
        rows.append({"t": t, "top1": hits[0][0], "rank": srcs.index(t["expect"]) + 1 if t["expect"] in srcs else None,
                     "got": srcs[0]})

    print(f"{'问题':<24}{'最高分':>6}  {'正确块排名':<8}第 1 名")
    for r in rows:
        if r["t"]["expect"]:
            rank = f"第 {r['rank']} 名" if r["rank"] else "❌ 没进前 %d" % k
        else:
            rank = "（不该答）"
        print(f"{r['t']['q']:<24}{r['top1']:>6.3f}  {rank:<8}{r['got']}")

    ans = [r for r in rows if r["t"]["expect"]]
    print(f"\n命中率 Hit@1 = {sum(r['rank'] == 1 for r in ans)}/{len(ans)}，"
          f"Hit@{k} = {sum(r['rank'] is not None for r in ans)}/{len(ans)}")

    # 扫一遍门槛：门槛高 → 拒答更准，但会误挡能答的题；门槛低 → 反过来
    no = [r for r in rows if not r["t"]["expect"]]
    print("\n门槛    能答的题被放行   不该答的题被挡住")
    for th in np.arange(0.40, 0.86, 0.05):
        ok = sum(r["rank"] is not None and r["top1"] >= th for r in ans)
        rej = sum(r["top1"] < th for r in no)
        mark = "  ← 当前" if abs(th - rag1.THRESHOLD) < 1e-6 else ""
        print(f"{th:.2f}       {ok:>2}/{len(ans)}            {rej}/{len(no)}{mark}")


# ---------- 2. 回答评测 ----------
JUDGE = """你是严格的评测员。根据【参考资料】判断【回答】是否正确回答了【问题】。
规则：回答里的事实必须和参考资料一致；参考资料里没有的信息算编造。
如果参考资料是"（无，知识库里没有答案）"，那么回答明确表示不知道/没有相关信息才算正确。
只输出 JSON：{"correct": true 或 false, "reason": "一句话理由"}"""


def keyword_ok(t, ans):
    if t["expect"] is None:
        return "没有" in ans or "不知道" in ans or "未找到" in ans
    return all(any(alt in ans for alt in group) for group in t["facts"])


def eval_answers():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    chunks, _ = rag1.get_index()
    kw_pass = judge_pass = 0
    print(f"\n{'问题':<24}关键词  裁判  理由")
    for t in TESTS:
        ans = rag1.answer(t["q"], verbose=False)
        ref = next((c["text"] for c in chunks if c["source"] == t["expect"]), "（无，知识库里没有答案）")
        resp = client.chat.completions.create(
            model="deepseek-chat", temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE},
                      {"role": "user", "content": f"【问题】{t['q']}\n【参考资料】{ref}\n【回答】{ans}"}])
        verdict = json.loads(resp.choices[0].message.content)
        kw, jd = keyword_ok(t, ans), bool(verdict.get("correct"))
        kw_pass += kw
        judge_pass += jd
        print(f"{t['q']:<24}{'✅' if kw else '❌'}    {'✅' if jd else '❌'}   {verdict.get('reason', '')}")
        if kw != jd:
            print(f"{'':<24}↑ 两种打分不一致，看看回答：{ans[:80]}")
    print(f"\n关键词检查 {kw_pass}/{len(TESTS)}，LLM 裁判 {judge_pass}/{len(TESTS)}")


if __name__ == "__main__":
    print(f"当前 THRESHOLD = {rag1.THRESHOLD}，测试题 {len(TESTS)} 道\n")
    eval_retrieval()
    if "--answer" in sys.argv:
        eval_answers()
