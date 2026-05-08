import os
import json
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
你是一名A股事件新闻检索助手。

请搜索并整理 A 股股票 罗博特科（300757）最近 3 天的重要事件新闻。
重点关注：
1. 重大合同
2. 中标 / 订单 / 签约
3. 子公司重大进展
4. 停牌 / 控制权变更
5. 问询函 / 处罚 / 诉讼
6. 业绩预告 / 财报重大变化

严格要求：
- 只保留最近 3 天内的信息
- 如果某一类未找到，不要编造
- 不要输出解释文字
- 必须严格输出 JSON

输出格式：
{
  "items": [
    {
      "date": "YYYY-MM-DD",
      "title": "新闻标题",
      "source": "来源",
      "summary": "一句话摘要",
      "category": "latest_news"
    }
  ]
}

category 只能取：
- latest_news
- event_catalyst
- risk_check
"""

    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search"}],
            input=prompt,
        )

        text = response.output_text
        print("=== 原始输出 ===")
        print(text)

        print("\n=== 尝试解析 JSON ===")
        data = json.loads(text)
        print(data)

    except Exception as e:
        print("调用失败:", repr(e))

if __name__ == "__main__":
    main()
