# -*- coding: utf-8 -*-
"""
测试 OpenAI Web Search 是否已经接入 src/search_service.py

测试内容：
1. 直接测试 OpenAIWebSearchProvider
2. 测试 SearchService provider 列表里是否包含 OpenAI-WebSearch
3. 测试 search_comprehensive_intel() 是否能拿到 event_catalyst 事件

运行：
python test_openai_web_search_integration.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv


# 确保项目根目录加入 Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def print_response(title, response):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print("query:", response.query)
    print("provider:", response.provider)
    print("success:", response.success)
    print("error_message:", response.error_message)
    print("results_count:", len(response.results))
    print("search_time:", response.search_time)

    for i, item in enumerate(response.results, 1):
        print("-" * 80)
        print(f"[{i}] date:", item.published_date)
        print(f"[{i}] title:", item.title)
        print(f"[{i}] source:", item.source)
        print(f"[{i}] url:", item.url)
        print(f"[{i}] snippet:", item.snippet)


def test_01_direct_provider():
    """
    直接测试 OpenAIWebSearchProvider。

    如果这里成功，说明：
    - OPENAI_API_KEY 正确
    - openai SDK 正确
    - 你新增的 OpenAIWebSearchProvider 类可用
    """
    from src.search_service import OpenAIWebSearchProvider

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未读取到 OPENAI_API_KEY，请检查 .env")

    provider = OpenAIWebSearchProvider([api_key])

    query = "罗博特科 300757 最近3天 重大合同 订单 中标 子公司 业绩 风险"
    response = provider.search(
        query=query,
        max_results=5,
        days=3,
    )

    print_response("TEST 01: 直接测试 OpenAIWebSearchProvider", response)

    assert response.provider == "OpenAI-WebSearch"
    assert response.success is True, response.error_message

    return response


def test_02_search_service_provider_list():
    """
    测试 SearchService 初始化后是否包含 OpenAI-WebSearch。

    如果这里没有 OpenAI-WebSearch，说明你还没有把 provider 插入到
    SearchService.__init__() 的 self._providers 里。
    """
    from src.search_service import get_search_service

    service = get_search_service()

    provider_names = [p.name for p in getattr(service, "_providers", [])]

    print("\n" + "=" * 100)
    print("TEST 02: SearchService provider 列表")
    print("=" * 100)
    print(json.dumps(provider_names, ensure_ascii=False, indent=2))

    assert "OpenAI-WebSearch" in provider_names, (
        "SearchService 里没有 OpenAI-WebSearch。"
        "请确认你已经在 SearchService.__init__() 里 append 了 OpenAIWebSearchProvider。"
    )

    return service


def test_03_search_service_stock_news():
    """
    测试 SearchService.search_stock_news() 是否优先走 OpenAI-WebSearch。

    你的 SearchService 没有 service.search() 方法；
    主流程里搜索股票新闻应使用 search_stock_news() 或 search_comprehensive_intel()。
    """
    from src.search_service import get_search_service

    service = get_search_service()

    response = service.search_stock_news(
        stock_code="300757",
        stock_name="罗博特科",
        max_results=5,
    )

    print_response("TEST 03: SearchService.search_stock_news()", response)

    assert response.success is True, response.error_message

    return response


def test_04_comprehensive_intel():
    """
    测试完整消息面聚合 search_comprehensive_intel()。

    你真正关心的是这里：
    - latest_news
    - event_catalyst
    - risk_check
    - earnings

    如果 event_catalyst 里能看到：
    - 4.03亿元重大合同
    - ficonTEC OCS 整线设备订单
    就说明主流程消息面搜索已经生效。
    """
    from src.search_service import get_search_service

    service = get_search_service()

    stock_code = "300757"
    stock_name = "罗博特科"

    intel = service.search_comprehensive_intel(
        stock_code=stock_code,
        stock_name=stock_name,
        max_searches=3,
    )

    print("\n" + "=" * 100)
    print("TEST 04: search_comprehensive_intel()")
    print("=" * 100)

    for dimension, response in intel.items():
        print_response(f"维度: {dimension}", response)

    # 不强制要求每个维度都有结果，但至少应该有一个维度成功
    success_count = sum(1 for r in intel.values() if r.success)
    result_count = sum(len(r.results) for r in intel.values())

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print("success_count:", success_count)
    print("total_result_count:", result_count)

    assert success_count > 0, "所有消息面维度都失败了"
    assert result_count > 0, "所有消息面维度都没有结果"

    return intel


def main():
    setup_logging()
    load_dotenv()

    print("=" * 100)
    print("OpenAI Web Search 集成测试")
    print("=" * 100)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    print("OPENAI_API_KEY:", "已读取" if api_key else "未读取")

    if not api_key:
        raise RuntimeError("未读取到 OPENAI_API_KEY，请检查 .env")

    test_01_direct_provider()
    test_02_search_service_provider_list()
    test_03_search_service_stock_news()
    test_04_comprehensive_intel()

    print("\n" + "=" * 100)
    print("全部测试完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
