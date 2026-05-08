from data_provider.tushare_fetcher import TushareFetcher

def main():
    fetcher = TushareFetcher()

    df = fetcher._call_api_with_rate_limit(
        "news",
        start_date="2026-05-05 00:00:00",
        end_date="2026-05-07 23:59:59",
        src="cls",
    )

    if df is None or df.empty:
        print("news 原始结果为空")
        return

    print("columns:", df.columns.tolist())
    print("rows:", len(df))
    print(df.head(50).to_string())

    # 宽松匹配
    keywords = ["罗博特科", "300757", "罗博", "重大合同", "签订", "子公司"]
    for kw in keywords:
        mask = (
            df["title"].astype(str).str.contains(kw, na=False) |
            df["content"].astype(str).str.contains(kw, na=False)
        )
        sub = df[mask]
        print(f"\n===== keyword={kw} 命中 {len(sub)} 条 =====")
        if not sub.empty:
            print(sub[["datetime", "title", "content"]].head(20).to_string())

if __name__ == "__main__":
    main()
