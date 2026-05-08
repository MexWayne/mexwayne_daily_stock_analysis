import os
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("未读取到 OPENAI_API_KEY")
        return

    client = OpenAI(api_key=api_key)

    prompt = """
请搜索并整理 A 股股票 罗博特科（300757） 最近 3 天的重要事件新闻。
重点关注：
1. 重大合同
2. 中标 / 订单 / 签约
3. 子公司重大进展
4. 停牌 / 控制权变更
5. 问询函 / 处罚 / 诉讼
6. 业绩预告 / 财报重大变化

要求：
- 优先返回最近 3 天的信息
- 如果找到，请给出：日期、标题、来源、简要摘要
- 如果没找到，明确写“未找到”
"""

    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search"}],
            input=prompt,
        )

        print("=== 原始响应文本 ===")
        print(response.output_text)

    except Exception as e:
        print("调用失败:", repr(e))

if __name__ == "__main__":
    main()
