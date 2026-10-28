# 第 4 周 L4（上）：一个最小的 MCP Server —— 把工具"装进一个独立程序"
# 不用手动运行它：mcp_agent.py 会自动把它当子进程启动，通过 stdin/stdout 和它对话。
# 想单独看看它：mcp dev week4/mcp_server.py（会打开 MCP Inspector 网页，可以手动点工具）
#
# 对比 agent.py：那里的工具是写死在 Agent 代码里的 Python 函数 + 手写 JSON Schema。
# 这里工具住在另一个程序里，Agent 启动时问它"你有哪些工具？"，再按标准协议调用。
# 注意：这里没有一行 JSON Schema，类型注解 city: str 和 docstring 就是说明书，SDK 自动生成。

from mcp.server import MCPServer

mcp = MCPServer("jet-tools")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询某个城市今天的天气"""
    fake = {"北京": "晴，12°C", "上海": "小雨，16°C", "深圳": "多云，24°C"}
    return fake.get(city, "未知城市")


@mcp.tool()
def calculator(expression: str) -> str:
    """计算四则运算表达式，如 (16-12)*2。涉及数字计算时必须用它"""
    if not set(expression) <= set("0123456789+-*/(). "):
        return "错误：只支持四则运算"
    return str(eval(expression))


@mcp.tool()
def word_count(text: str) -> int:
    """统计一段文字的字符数（不含空格）"""
    return len(text.replace(" ", ""))


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输：从 stdin 读请求，往 stdout 写回复
