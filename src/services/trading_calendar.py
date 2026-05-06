# src/services/trading_calendar.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class MarketStatus:
    run_datetime: str
    calendar_date: str
    is_trade_day: bool
    is_market_hours: bool
    analysis_mode: str
    last_trade_day: Optional[str]
    next_trade_day: Optional[str]
    mode_note: str


class TradingCalendarService:
    """
    A股交易日/交易时段判断。

    优先级：
    1. 本地缓存 data/trade_calendar_a_stock.json
    2. akshare.tool_trade_date_hist_sina()
    3. fallback：周一到周五粗略判断

    注意：
    fallback 不识别节假日，只用于兜底。
    """

    def __init__(self, cache_path: str = "data/trade_calendar_a_stock.json"):
        self.cache_path = Path(cache_path)
        self._trade_dates: Optional[Set[date]] = None

    def get_market_status(self, now: Optional[datetime] = None) -> Dict:
        now = now or datetime.now()
        today = now.date()

        trade_dates = self._load_trade_dates()
        is_trade_day = self._is_trade_day(today, trade_dates)

        last_trade_day = self._get_last_trade_day(today, trade_dates)
        next_trade_day = self._get_next_trade_day(today, trade_dates)

        morning_open = time(9, 30)
        morning_close = time(11, 30)
        afternoon_open = time(13, 0)
        afternoon_close = time(15, 0)

        now_time = now.time()

        is_market_hours = (
            is_trade_day
            and (
                morning_open <= now_time <= morning_close
                or afternoon_open <= now_time <= afternoon_close
            )
        )

        if not is_trade_day:
            analysis_mode = "non_trading_day"
            mode_note = "今日A股休市，行情和K线应按最近可用交易日理解，不得写成今日盘中行情。"
        elif now_time < morning_open:
            analysis_mode = "pre_market"
            mode_note = "今日尚未开盘，行情和K线应按上一交易日或最近可用交易日理解，操作需等待开盘确认。"
        elif morning_close < now_time < afternoon_open:
            analysis_mode = "lunch_break"
            mode_note = "当前处于午间休市，实时行情为上午收盘附近状态，下午需继续确认。"
        elif is_market_hours:
            analysis_mode = "trading_intraday"
            mode_note = "当前处于A股交易时段，实时行情可作为盘中参考。"
        else:
            analysis_mode = "after_market"
            mode_note = "今日已收盘，行情可按当日收盘或最新可用行情理解。"

        return {
            "run_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "calendar_date": today.strftime("%Y-%m-%d"),
            "is_trade_day": is_trade_day,
            "is_market_hours": is_market_hours,
            "analysis_mode": analysis_mode,
            "last_trade_day": last_trade_day.strftime("%Y-%m-%d") if last_trade_day else None,
            "next_trade_day": next_trade_day.strftime("%Y-%m-%d") if next_trade_day else None,
            "mode_note": mode_note,
        }

    def _load_trade_dates(self) -> Optional[Set[date]]:
        if self._trade_dates is not None:
            return self._trade_dates

        # 1. 先读本地缓存
        cached = self._load_from_cache()
        if cached:
            self._trade_dates = cached
            return self._trade_dates

        # 2. 再尝试 akshare
        fetched = self._fetch_from_akshare()
        if fetched:
            self._trade_dates = fetched
            self._save_to_cache(fetched)
            return self._trade_dates

        # 3. 兜底：None 表示只按 weekday 判断
        logger.warning("[交易日历] 无法加载正式交易日历，回退到工作日粗略判断")
        self._trade_dates = None
        return None

    def _load_from_cache(self) -> Optional[Set[date]]:
        try:
            if not self.cache_path.exists():
                return None

            items = json.loads(self.cache_path.read_text(encoding="utf-8"))
            dates = {datetime.strptime(x, "%Y-%m-%d").date() for x in items}
            if dates:
                logger.info("[交易日历] 已从缓存加载 %d 个交易日", len(dates))
                return dates
        except Exception as exc:
            logger.warning("[交易日历] 读取缓存失败: %s", exc, exc_info=True)

        return None

    def _save_to_cache(self, dates: Set[date]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            items = sorted(d.strftime("%Y-%m-%d") for d in dates)
            self.cache_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("[交易日历] 已写入缓存: %s", self.cache_path)
        except Exception as exc:
            logger.warning("[交易日历] 写入缓存失败: %s", exc, exc_info=True)

    def _fetch_from_akshare(self) -> Optional[Set[date]]:
        try:
            import akshare as ak

            df = ak.tool_trade_date_hist_sina()
            if df is None or df.empty:
                return None

            # 常见列名：trade_date
            date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]

            dates = set()
            for value in df[date_col].tolist():
                parsed = self._parse_date_value(value)
                if parsed:
                    dates.add(parsed)

            if dates:
                logger.info("[交易日历] 已从 AkShare 获取 %d 个交易日", len(dates))
                return dates

        except Exception as exc:
            logger.warning("[交易日历] AkShare 获取失败: %s", exc, exc_info=True)

        return None

    def _parse_date_value(self, value) -> Optional[date]:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        s = str(value).strip()
        if not s:
            return None

        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass

        return None

    def _is_trade_day(self, d: date, trade_dates: Optional[Set[date]]) -> bool:
        if trade_dates:
            return d in trade_dates

        # fallback：只按周一到周五判断，不识别节假日
        return d.weekday() < 5

    def _get_last_trade_day(self, d: date, trade_dates: Optional[Set[date]]) -> Optional[date]:
        cur = d - timedelta(days=1)
        for _ in range(30):
            if self._is_trade_day(cur, trade_dates):
                return cur
            cur -= timedelta(days=1)
        return None

    def _get_next_trade_day(self, d: date, trade_dates: Optional[Set[date]]) -> Optional[date]:
        cur = d + timedelta(days=1)
        for _ in range(30):
            if self._is_trade_day(cur, trade_dates):
                return cur
            cur += timedelta(days=1)
        return None


_default_service: Optional[TradingCalendarService] = None


def get_market_status(now: Optional[datetime] = None) -> Dict:
    global _default_service
    if _default_service is None:
        _default_service = TradingCalendarService()
    return _default_service.get_market_status(now)