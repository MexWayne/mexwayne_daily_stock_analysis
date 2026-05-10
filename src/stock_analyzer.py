# -*- coding: utf-8 -*-
"""
===================================
趋势交易分析器 - 基于用户交易理念
===================================

交易理念核心原则：
1. 严进策略 - 不追高，追求每笔交易成功率
2. 趋势交易 - MA5>MA10>MA20 多头排列，顺势而为
3. 效率优先 - 关注筹码结构好的股票
4. 买点偏好 - 在 MA5/MA10 附近回踩买入

技术标准：
- 多头排列：MA5 > MA10 > MA20
- 乖离率：(Close - MA5) / MA5 < 5%（不追高）
- 量能形态：缩量回调优先
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

import pandas as pd
import numpy as np

from src.config import get_config

logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    """趋势状态枚举"""
    STRONG_BULL = "强势多头"      # MA5 > MA10 > MA20，且间距扩大
    BULL = "多头排列"             # MA5 > MA10 > MA20
    WEAK_BULL = "弱势多头"        # MA5 > MA10，但 MA10 < MA20
    CONSOLIDATION = "盘整"        # 均线缠绕
    WEAK_BEAR = "弱势空头"        # MA5 < MA10，但 MA10 > MA20
    BEAR = "空头排列"             # MA5 < MA10 < MA20
    STRONG_BEAR = "强势空头"      # MA5 < MA10 < MA20，且间距扩大


class VolumeStatus(Enum):
    """量能状态枚举"""
    HEAVY_VOLUME_UP = "放量上涨"       # 量价齐升
    HEAVY_VOLUME_DOWN = "放量下跌"     # 放量杀跌
    SHRINK_VOLUME_UP = "缩量上涨"      # 无量上涨
    SHRINK_VOLUME_DOWN = "缩量回调"    # 缩量回调（好）
    NORMAL = "量能正常"


class BuySignal(Enum):
    """买入信号枚举"""
    STRONG_BUY = "强烈买入"       # 多条件满足
    BUY = "买入"                  # 基本条件满足
    HOLD = "持有"                 # 已持有可继续
    WAIT = "观望"                 # 等待更好时机
    SELL = "卖出"                 # 趋势转弱
    STRONG_SELL = "强烈卖出"      # 趋势破坏


class MACDStatus(Enum):
    """MACD状态枚举"""
    GOLDEN_CROSS_ZERO = "零轴上金叉"      # DIF上穿DEA，且在零轴上方
    GOLDEN_CROSS = "金叉"                # DIF上穿DEA
    BULLISH = "多头"                    # DIF>DEA>0
    CROSSING_UP = "上穿零轴"             # DIF上穿零轴
    CROSSING_DOWN = "下穿零轴"           # DIF下穿零轴
    BEARISH = "空头"                    # DIF<DEA<0
    DEATH_CROSS = "死叉"                # DIF下穿DEA


class RSIStatus(Enum):
    """RSI状态枚举"""
    OVERBOUGHT = "超买"        # RSI > 70
    STRONG_BUY = "强势买入"    # 50 < RSI < 70
    NEUTRAL = "中性"          # 40 <= RSI <= 60
    WEAK = "弱势"             # 30 < RSI < 40
    OVERSOLD = "超卖"         # RSI < 30


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    code: str
    
    # 趋势判断
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    ma_alignment: str = ""           # 均线排列描述
    trend_strength: float = 0.0      # 趋势强度 0-100
    
    # 均线数据
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    current_price: float = 0.0
    
    # 乖离率（与 MA5 的偏离度）
    bias_ma5: float = 0.0            # (Close - MA5) / MA5 * 100
    bias_ma10: float = 0.0
    bias_ma20: float = 0.0
    
    # 量能分析
    volume_status: VolumeStatus = VolumeStatus.NORMAL
    volume_ratio_5d: float = 0.0     # 当日成交量/5日均量
    volume_trend: str = ""           # 量能趋势描述
    
    # 支撑压力
    support_ma5: bool = False        # MA5 是否构成支撑
    support_ma10: bool = False       # MA10 是否构成支撑
    resistance_levels: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)

    # K线结构分析
    latest_candle_type: str = ""          # 最新K线类型：强实体阳线/长上影/长下影等
    candle_body_pct: float = 0.0          # 实体占全天振幅比例
    upper_shadow_pct: float = 0.0         # 上影线占全天振幅比例
    lower_shadow_pct: float = 0.0         # 下影线占全天振幅比例
    close_position_pct: float = 0.0       # 收盘价在日内区间的位置，越高越强
    price_change_pct: float = 0.0         # 当日涨跌幅

    break_20d_high: bool = False          # 是否突破近20日高点
    break_20d_low: bool = False           # 是否跌破近20日低点
    near_ma5: bool = False                # 是否贴近MA5
    near_ma10: bool = False               # 是否贴近MA10
    near_ma20: bool = False               # 是否贴近MA20

    kline_structure_score: int = 50       # K线结构分，0-100
    kline_structure_label: str = "中性结构"
    kline_signals: List[str] = field(default_factory=list)
    kline_risks: List[str] = field(default_factory=list)
    kline_summary: str = ""

    # MACD 指标
    macd_dif: float = 0.0          # DIF 快线
    macd_dea: float = 0.0          # DEA 慢线
    macd_bar: float = 0.0           # MACD 柱状图
    macd_status: MACDStatus = MACDStatus.BULLISH
    macd_signal: str = ""            # MACD 信号描述

    # RSI 指标
    rsi_6: float = 0.0              # RSI(6) 短期
    rsi_12: float = 0.0             # RSI(12) 中期
    rsi_24: float = 0.0             # RSI(24) 长期
    rsi_status: RSIStatus = RSIStatus.NEUTRAL
    rsi_signal: str = ""              # RSI 信号描述

    # 买入信号
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0            # 综合评分 0-100
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'ma_alignment': self.ma_alignment,
            'trend_strength': self.trend_strength,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'current_price': self.current_price,
            'bias_ma5': self.bias_ma5,
            'bias_ma10': self.bias_ma10,
            'bias_ma20': self.bias_ma20,
            'volume_status': self.volume_status.value,
            'volume_ratio_5d': self.volume_ratio_5d,
            'volume_trend': self.volume_trend,
            'support_ma5': self.support_ma5,
            'support_ma10': self.support_ma10,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,

            # K线结构
            'latest_candle_type': self.latest_candle_type,
            'candle_body_pct': self.candle_body_pct,
            'upper_shadow_pct': self.upper_shadow_pct,
            'lower_shadow_pct': self.lower_shadow_pct,
            'close_position_pct': self.close_position_pct,
            'price_change_pct': self.price_change_pct,
            'break_20d_high': self.break_20d_high,
            'break_20d_low': self.break_20d_low,
            'near_ma5': self.near_ma5,
            'near_ma10': self.near_ma10,
            'near_ma20': self.near_ma20,
            'kline_structure_score': self.kline_structure_score,
            'kline_structure_label': self.kline_structure_label,
            'kline_signals': self.kline_signals,
            'kline_risks': self.kline_risks,
            'kline_summary': self.kline_summary,

            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
            'macd_dif': self.macd_dif,
            'macd_dea': self.macd_dea,
            'macd_bar': self.macd_bar,
            'macd_status': self.macd_status.value,
            'macd_signal': self.macd_signal,
            'rsi_6': self.rsi_6,
            'rsi_12': self.rsi_12,
            'rsi_24': self.rsi_24,
            'rsi_status': self.rsi_status.value,
            'rsi_signal': self.rsi_signal,
        }


class StockTrendAnalyzer:
    """
    股票趋势分析器

    基于用户交易理念实现：
    1. 趋势判断 - MA5>MA10>MA20 多头排列
    2. 乖离率检测 - 不追高，偏离 MA5 超过 5% 不买
    3. 量能分析 - 偏好缩量回调
    4. 买点识别 - 回踩 MA5/MA10 支撑
    5. MACD 指标 - 趋势确认和金叉死叉信号
    6. RSI 指标 - 超买超卖判断
    """
    
    # 交易参数配置（BIAS_THRESHOLD 从 Config 读取，见 _generate_signal）
    VOLUME_SHRINK_RATIO = 0.7   # 缩量判断阈值（当日量/5日均量）
    VOLUME_HEAVY_RATIO = 1.5    # 放量判断阈值
    MA_SUPPORT_TOLERANCE = 0.02  # MA 支撑判断容忍度（2%）

    # MACD 参数（标准12/26/9）
    MACD_FAST = 12              # 快线周期
    MACD_SLOW = 26             # 慢线周期
    MACD_SIGNAL = 9             # 信号线周期

    # RSI 参数
    RSI_SHORT = 6               # 短期RSI周期
    RSI_MID = 12               # 中期RSI周期
    RSI_LONG = 24              # 长期RSI周期
    RSI_OVERBOUGHT = 70        # 超买阈值
    RSI_OVERSOLD = 30          # 超卖阈值
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    # fix bug in log: volume=363742 but latest_volume=36374206.0
    def _normalize_volume_for_ratio(self, latest_volume: float, avg_volume: float) -> float:
        """
        修正最新成交量与历史成交量单位不一致的问题。

        常见情况：
        - 实时行情 volume 是“手”
        - 历史日线 volume 是“股”
        二者可能相差约 100 倍。

        例：
        latest_volume = 363742
        avg_volume = 32869273
        直接相除 = 0.01，明显错误
        修正后 latest_volume * 100 = 36374200
        量比约 = 1.11
        """
        try:
            latest_volume = float(latest_volume)
            avg_volume = float(avg_volume)
        except (TypeError, ValueError):
            return latest_volume

        if latest_volume <= 0 or avg_volume <= 0:
            return latest_volume

        raw_ratio = latest_volume / avg_volume

        # 最新量明显比历史均量小两个数量级，大概率最新量单位是“手”，历史量单位是“股”
        if 0 < raw_ratio < 0.05:
            fixed_volume = latest_volume * 100
            fixed_ratio = fixed_volume / avg_volume

            # 修正后回到合理区间，才采用修正值
            if 0.05 <= fixed_ratio <= 20:
                logger.info(
                    "成交量单位自动修正: latest_volume=%s -> %s, avg_volume=%s, ratio %.4f -> %.4f",
                    latest_volume,
                    fixed_volume,
                    avg_volume,
                    raw_ratio,
                    fixed_ratio,
                )
                return fixed_volume

        # 反向兜底：最新量明显比历史均量大两个数量级
        if raw_ratio > 50:
            fixed_volume = latest_volume / 100
            fixed_ratio = fixed_volume / avg_volume

            if 0.05 <= fixed_ratio <= 20:
                logger.info(
                    "成交量单位自动修正: latest_volume=%s -> %s, avg_volume=%s, ratio %.4f -> %.4f",
                    latest_volume,
                    fixed_volume,
                    avg_volume,
                    raw_ratio,
                    fixed_ratio,
                )
                return fixed_volume

        return latest_volume
    

    def _resolve_price_change_pct(
        self,
        latest,
        prev,
        realtime_change_pct: Optional[float] = None,
    ) -> float:
        """
        统一解析涨跌幅。

        优先级：
        1. 实时行情 change_pct
        2. DataFrame 内已有 change_pct / pct_chg 字段
        3. 用 latest.close 和 prev.close 兜底计算

        注意：
        - 实时行情在开盘前/盘中通常比 df 中的伪最新行更可靠
        - 当前修复目标：避免 K线摘要里出现“涨跌幅=0.00%”
        """
        # 1. 优先使用实时行情 change_pct
        try:
            if realtime_change_pct is not None:
                v = float(realtime_change_pct)
                if not np.isnan(v) and not np.isinf(v):
                    return v
        except (TypeError, ValueError):
            pass

        # 2. 其次使用 df 中已有涨跌幅字段
        for col in ["change_pct", "pct_chg", "pct_change", "涨跌幅"]:
            try:
                if hasattr(latest, "index") and col in latest.index:
                    v = float(latest[col])
                    if not np.isnan(v) and not np.isinf(v):
                        return v
            except (TypeError, ValueError):
                continue

        # 3. 最后兜底用 close 计算
        try:
            prev_close = float(prev["close"])
            latest_close = float(latest["close"])
            if prev_close > 0:
                return (latest_close - prev_close) / prev_close * 100
        except (TypeError, ValueError, KeyError):
            pass

        return 0.0



    
    def analyze(
        self,
        df: pd.DataFrame,
        code: str,
        realtime_change_pct: Optional[float] = None,
    ) -> TrendAnalysisResult:
        """
        分析股票趋势
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            code: 股票代码
            
        Returns:
            TrendAnalysisResult 分析结果
        """
        result = TrendAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code} 数据不足，无法进行趋势分析")
            result.risk_factors.append("数据不足，无法完成分析")
            return result
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算均线
        df = self._calculate_mas(df)

        # 计算 MACD 和 RSI
        df = self._calculate_macd(df)
        df = self._calculate_rsi(df)

        # 获取最新数据
        latest = df.iloc[-1]
        result.current_price = float(latest['close'])
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        result.ma60 = float(latest.get('MA60', 0))

        # 1. 趋势判断
        self._analyze_trend(df, result)

        # 2. 乖离率计算
        self._calculate_bias(result)

        # 3. 量能分析
        self._analyze_volume(df, result, realtime_change_pct=realtime_change_pct)

        # 4. 支撑压力分析
        self._analyze_support_resistance(df, result)

        # 4.5 K线结构分析
        self._analyze_kline_structure(df, result, realtime_change_pct=realtime_change_pct)

        # 5. MACD 分析
        self._analyze_macd(df, result)

        # 6. RSI 分析
        self._analyze_rsi(df, result)

        # 7. 生成买入信号
        self._generate_signal(result)

        return result
    
    def _calculate_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 60:
            df['MA60'] = df['close'].rolling(window=60).mean()
        else:
            df['MA60'] = df['MA20']  # 数据不足时使用 MA20 替代
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 MACD 指标

        公式：
        - EMA(12)：12日指数移动平均
        - EMA(26)：26日指数移动平均
        - DIF = EMA(12) - EMA(26)
        - DEA = EMA(DIF, 9)
        - MACD = (DIF - DEA) * 2
        """
        df = df.copy()

        # 计算快慢线 EMA
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False).mean()

        # 计算快线 DIF
        df['MACD_DIF'] = ema_fast - ema_slow

        # 计算信号线 DEA
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=self.MACD_SIGNAL, adjust=False).mean()

        # 计算柱状图
        df['MACD_BAR'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 RSI 指标

        公式：
        - RS = 平均上涨幅度 / 平均下跌幅度
        - RSI = 100 - (100 / (1 + RS))
        """
        df = df.copy()

        for period in [self.RSI_SHORT, self.RSI_MID, self.RSI_LONG]:
            # 计算价格变化
            delta = df['close'].diff()

            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 计算平均涨跌幅
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # 计算 RS 和 RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # 填充 NaN 值
            rsi = rsi.fillna(50)  # 默认中性值

            # 添加到 DataFrame
            col_name = f'RSI_{period}'
            df[col_name] = rsi

        return df
    
    def _analyze_trend(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析趋势状态
        
        核心逻辑：判断均线排列和趋势强度
        """
        ma5, ma10, ma20 = result.ma5, result.ma10, result.ma20
        
        # 判断均线排列
        if ma5 > ma10 > ma20:
            # 检查间距是否在扩大（强势）
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20'] * 100 if prev['MA20'] > 0 else 0
            curr_spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "强势多头排列，均线发散上行"
                result.trend_strength = 90
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "多头排列 MA5>MA10>MA20"
                result.trend_strength = 75
                
        elif ma5 > ma10 and ma10 <= ma20:
            result.trend_status = TrendStatus.WEAK_BULL
            result.ma_alignment = "弱势多头，MA5>MA10 但 MA10≤MA20"
            result.trend_strength = 55
            
        elif ma5 < ma10 < ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA20'] - prev['MA5']) / prev['MA5'] * 100 if prev['MA5'] > 0 else 0
            curr_spread = (ma20 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BEAR
                result.ma_alignment = "强势空头排列，均线发散下行"
                result.trend_strength = 10
            else:
                result.trend_status = TrendStatus.BEAR
                result.ma_alignment = "空头排列 MA5<MA10<MA20"
                result.trend_strength = 25
                
        elif ma5 < ma10 and ma10 >= ma20:
            result.trend_status = TrendStatus.WEAK_BEAR
            result.ma_alignment = "弱势空头，MA5<MA10 但 MA10≥MA20"
            result.trend_strength = 40
            
        else:
            result.trend_status = TrendStatus.CONSOLIDATION
            result.ma_alignment = "均线缠绕，趋势不明"
            result.trend_strength = 50
    
    def _calculate_bias(self, result: TrendAnalysisResult) -> None:
        """
        计算乖离率
        
        乖离率 = (现价 - 均线) / 均线 * 100%
        
        严进策略：乖离率超过 5% 不追高
        """
        price = result.current_price
        
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
        if result.ma10 > 0:
            result.bias_ma10 = (price - result.ma10) / result.ma10 * 100
        if result.ma20 > 0:
            result.bias_ma20 = (price - result.ma20) / result.ma20 * 100
    
    def _analyze_volume(
        self,
        df: pd.DataFrame,
        result: TrendAnalysisResult,
        realtime_change_pct: Optional[float] = None,
    ) -> None:
        """
        分析量能

        偏好：缩量回调 > 放量上涨 > 缩量上涨 > 放量下跌

        注意：
        - 部分实时行情源的 volume 可能是“手”
        - 历史日线 volume 可能是“股”
        - 因此计算量比前必须统一成交量单位
        """
        if df is None or df.empty or len(df) < 6:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_ratio_5d = 0.0
            result.volume_trend = "量能数据不足"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        vol_5d_avg = df["volume"].iloc[-6:-1].mean()

        try:
            latest_volume_raw = float(latest["volume"])
            vol_5d_avg = float(vol_5d_avg)
        except (TypeError, ValueError):
            result.volume_status = VolumeStatus.NORMAL
            result.volume_ratio_5d = 0.0
            result.volume_trend = "量能数据异常"
            return

        if vol_5d_avg <= 0:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_ratio_5d = 0.0
            result.volume_trend = "5日均量无效"
            return

        # 关键修复：统一最新成交量与历史均量单位
        latest_volume = self._normalize_volume_for_ratio(
            latest_volume=latest_volume_raw,
            avg_volume=vol_5d_avg,
        )

        result.volume_ratio_5d = latest_volume / vol_5d_avg

        # 判断价格变化
        try:
            prev_close = float(prev["close"])
            latest_close = float(latest["close"])
            # 判断价格变化：优先使用实时行情 change_pct
            price_change = self._resolve_price_change_pct(
                latest=latest,
                prev=prev,
                realtime_change_pct=realtime_change_pct,
            )
        except (TypeError, ValueError):
            price_change = 0.0

        # 量能状态判断
        if result.volume_ratio_5d >= self.VOLUME_HEAVY_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_UP
                result.volume_trend = "放量上涨，多头力量强劲"
            else:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_DOWN
                result.volume_trend = "放量下跌，注意风险"

        elif result.volume_ratio_5d <= self.VOLUME_SHRINK_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_UP
                result.volume_trend = "缩量上涨，上攻动能不足"
            else:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_DOWN
                result.volume_trend = "缩量回调，洗盘特征明显（好）"

        else:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_trend = "量能正常"

        logger.info(
            "%s 量能分析: latest_volume_raw=%s, latest_volume_used=%.2f, "
            "vol_5d_avg=%.2f, volume_ratio_5d=%.2f, price_change=%.2f%%, status=%s",
            result.code,
            latest_volume_raw,
            latest_volume,
            vol_5d_avg,
            result.volume_ratio_5d,
            price_change,
            result.volume_status.value,
        )
    
    def _analyze_support_resistance(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析支撑压力位
        
        买点偏好：回踩 MA5/MA10 获得支撑
        """
        price = result.current_price
        
        # 检查是否在 MA5 附近获得支撑
        if result.ma5 > 0:
            ma5_distance = abs(price - result.ma5) / result.ma5
            if ma5_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma5:
                result.support_ma5 = True
                result.support_levels.append(result.ma5)
        
        # 检查是否在 MA10 附近获得支撑
        if result.ma10 > 0:
            ma10_distance = abs(price - result.ma10) / result.ma10
            if ma10_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma10:
                result.support_ma10 = True
                if result.ma10 not in result.support_levels:
                    result.support_levels.append(result.ma10)
        
        # MA20 作为重要支撑
        if result.ma20 > 0 and price >= result.ma20:
            result.support_levels.append(result.ma20)
        
        # 近期高点作为压力
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            if recent_high > price:
                result.resistance_levels.append(recent_high)

    def _analyze_kline_structure(
        self,
        df: pd.DataFrame,
        result: TrendAnalysisResult,
        realtime_change_pct: Optional[float] = None,
    ) -> None:
        """
        分析最新K线结构。

        重点补足原趋势分析没有覆盖的部分：
        1. 实体大小
        2. 上下影线
        3. 收盘位置
        4. 突破/跌破近20日高低点
        5. 是否贴近 MA5/MA10/MA20
        6. 量价配合是否健康
        """
        if df is None or df.empty or len(df) < 20:
            result.kline_summary = "K线数据不足，无法完成结构分析"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        open_p = float(latest["open"])
        high_p = float(latest["high"])
        low_p = float(latest["low"])
        close_p = float(latest["close"])
        volume = float(latest["volume"])

        prev_close = float(prev["close"])
        day_range = max(high_p - low_p, 1e-9)

        body = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p

        body_pct = body / day_range * 100
        upper_pct = upper_shadow / day_range * 100
        lower_pct = lower_shadow / day_range * 100
        close_pos_pct = (close_p - low_p) / day_range * 100

        price_change_pct = self._resolve_price_change_pct(
            latest=latest,
            prev=prev,
            realtime_change_pct=realtime_change_pct,
        )

        result.candle_body_pct = round(body_pct, 2)
        result.upper_shadow_pct = round(upper_pct, 2)
        result.lower_shadow_pct = round(lower_pct, 2)
        result.close_position_pct = round(close_pos_pct, 2)
        result.price_change_pct = round(price_change_pct, 2)

        is_red = close_p >= open_p
        is_green = close_p < open_p

        # === 1. 单根K线类型 ===
        if body_pct <= 10:
            candle_type = "十字星/小实体"
        elif upper_pct >= 45:
            candle_type = "长上影线"
        elif lower_pct >= 45:
            candle_type = "长下影线"
        elif is_red and body_pct >= 55 and close_pos_pct >= 70:
            candle_type = "强实体阳线"
        elif is_green and body_pct >= 55 and close_pos_pct <= 35:
            candle_type = "强实体阴线"
        elif is_red:
            candle_type = "普通阳线"
        else:
            candle_type = "普通阴线"

        result.latest_candle_type = candle_type

        # === 2. 量能基准 ===
        vol_5d_avg = df["volume"].iloc[-6:-1].mean()
        if vol_5d_avg is not None and vol_5d_avg > 0:
            volume = self._normalize_volume_for_ratio(volume, vol_5d_avg)
            volume_ratio_5d = volume / vol_5d_avg
        else:
            volume_ratio_5d = 1.0

        # === 3. 近20日突破/跌破 ===
        prev_20_high = float(df["high"].iloc[-21:-1].max())
        prev_20_low = float(df["low"].iloc[-21:-1].min())

        result.break_20d_high = close_p > prev_20_high
        result.break_20d_low = close_p < prev_20_low

        # === 4. 是否贴近均线 ===
        def _near(price: float, ma: float, tolerance: float = 0.02) -> bool:
            return ma > 0 and abs(price - ma) / ma <= tolerance

        result.near_ma5 = _near(close_p, result.ma5)
        result.near_ma10 = _near(close_p, result.ma10)
        result.near_ma20 = _near(close_p, result.ma20, tolerance=0.025)

        # === 5. 结构信号评分 ===
        score = 50
        signals: List[str] = []
        risks: List[str] = []

        # 正向信号
        if result.break_20d_high and is_red and close_pos_pct >= 70 and volume_ratio_5d >= 1.2:
            score += 22
            signals.append("放量突破近20日高点，且收盘位于日内高位，突破结构偏强")

        if is_red and body_pct >= 55 and close_pos_pct >= 75 and volume_ratio_5d >= 1.1:
            score += 16
            signals.append("放量实体阳线，主动买盘较强")

        if lower_pct >= 45 and close_pos_pct >= 60:
            score += 12
            signals.append("长下影线收回，盘中下探后有承接")

        if result.near_ma5 and close_p >= result.ma5 and volume_ratio_5d <= 1.1:
            score += 10
            signals.append("贴近MA5且未有效跌破，短线支撑仍在")

        if result.near_ma10 and close_p >= result.ma10 and volume_ratio_5d <= 1.0:
            score += 14
            signals.append("缩量回踩MA10，符合低吸观察区")

        if result.near_ma20 and close_p >= result.ma20 and volume_ratio_5d <= 1.0:
            score += 10
            signals.append("靠近MA20但未跌破，中期支撑暂未破坏")

        # 风险信号
        if upper_pct >= 45 and volume_ratio_5d >= 1.3:
            score -= 22
            risks.append("放量长上影，冲高回落明显，上方抛压较重")

        if is_green and body_pct >= 50 and close_pos_pct <= 35 and volume_ratio_5d >= 1.2:
            score -= 24
            risks.append("放量实体阴线，主动卖压较强")

        if close_p < result.ma20 and volume_ratio_5d >= 1.2:
            score -= 25
            risks.append("放量跌破MA20，中期趋势支撑被破坏")

        if result.break_20d_low:
            score -= 20
            risks.append("跌破近20日低点，短线结构明显转弱")

        if price_change_pct >= 7 and upper_pct >= 35:
            score -= 12
            risks.append("大涨后出现明显上影线，短线追高性价比下降")

        if price_change_pct <= -5 and close_pos_pct <= 35:
            score -= 12
            risks.append("大跌且收盘靠近低位，弱势未修复")

        # 分数归一化
        score = max(0, min(100, score))
        result.kline_structure_score = score

        if score >= 80:
            label = "强势突破结构"
        elif score >= 65:
            label = "偏强结构"
        elif score >= 50:
            label = "中性偏强结构"
        elif score >= 35:
            label = "中性偏弱结构"
        else:
            label = "风险结构"

        result.kline_structure_label = label
        result.kline_signals = signals
        result.kline_risks = risks

        # === 6. 生成一句话摘要，方便后续 prompt 直接引用 ===
        parts = [
            f"最新K线为{result.latest_candle_type}",
            f"K线结构={label}",
            f"结构分={score}/100",
            f"涨跌幅={result.price_change_pct:.2f}%",
            f"收盘位置={result.close_position_pct:.1f}%",
            f"5日量比={volume_ratio_5d:.2f}",
        ]

        if signals:
            parts.append("正向信号：" + "；".join(signals[:2]))
        if risks:
            parts.append("风险信号：" + "；".join(risks[:2]))

        result.kline_summary = "；".join(parts)

        # for debug
        logger.info(
            "%s K线涨跌幅解析: realtime_change_pct=%s, final_price_change_pct=%.2f%%",
            result.code,
            realtime_change_pct,
            result.price_change_pct,
        )


    def _analyze_macd(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析 MACD 指标

        核心信号：
        - 零轴上金叉：最强买入信号
        - 金叉：DIF 上穿 DEA
        - 死叉：DIF 下穿 DEA
        """
        if len(df) < self.MACD_SLOW:
            result.macd_signal = "数据不足"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 获取 MACD 数据
        result.macd_dif = float(latest['MACD_DIF'])
        result.macd_dea = float(latest['MACD_DEA'])
        result.macd_bar = float(latest['MACD_BAR'])

        # 判断金叉死叉
        prev_dif_dea = prev['MACD_DIF'] - prev['MACD_DEA']
        curr_dif_dea = result.macd_dif - result.macd_dea

        # 金叉：DIF 上穿 DEA
        is_golden_cross = prev_dif_dea <= 0 and curr_dif_dea > 0

        # 死叉：DIF 下穿 DEA
        is_death_cross = prev_dif_dea >= 0 and curr_dif_dea < 0

        # 零轴穿越
        prev_zero = prev['MACD_DIF']
        curr_zero = result.macd_dif
        is_crossing_up = prev_zero <= 0 and curr_zero > 0
        is_crossing_down = prev_zero >= 0 and curr_zero < 0

        # 判断 MACD 状态
        if is_golden_cross and curr_zero > 0:
            result.macd_status = MACDStatus.GOLDEN_CROSS_ZERO
            result.macd_signal = "⭐ 零轴上金叉，强烈买入信号！"
        elif is_crossing_up:
            result.macd_status = MACDStatus.CROSSING_UP
            result.macd_signal = "⚡ DIF上穿零轴，趋势转强"
        elif is_golden_cross:
            result.macd_status = MACDStatus.GOLDEN_CROSS
            result.macd_signal = "✅ 金叉，趋势向上"
        elif is_death_cross:
            result.macd_status = MACDStatus.DEATH_CROSS
            result.macd_signal = "❌ 死叉，趋势向下"
        elif is_crossing_down:
            result.macd_status = MACDStatus.CROSSING_DOWN
            result.macd_signal = "⚠️ DIF下穿零轴，趋势转弱"
        elif result.macd_dif > 0 and result.macd_dea > 0:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = "✓ 多头排列，持续上涨"
        elif result.macd_dif < 0 and result.macd_dea < 0:
            result.macd_status = MACDStatus.BEARISH
            result.macd_signal = "⚠ 空头排列，持续下跌"
        else:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = " MACD 中性区域"

    def _analyze_rsi(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析 RSI 指标

        核心判断：
        - RSI > 70：超买，谨慎追高
        - RSI < 30：超卖，关注反弹
        - 40-60：中性区域
        """
        if len(df) < self.RSI_LONG:
            result.rsi_signal = "数据不足"
            return

        latest = df.iloc[-1]

        # 获取 RSI 数据
        result.rsi_6 = float(latest[f'RSI_{self.RSI_SHORT}'])
        result.rsi_12 = float(latest[f'RSI_{self.RSI_MID}'])
        result.rsi_24 = float(latest[f'RSI_{self.RSI_LONG}'])

        # 以中期 RSI(12) 为主进行判断
        rsi_mid = result.rsi_12

        # 判断 RSI 状态
        if rsi_mid > self.RSI_OVERBOUGHT:
            result.rsi_status = RSIStatus.OVERBOUGHT
            result.rsi_signal = f"⚠️ RSI超买({rsi_mid:.1f}>70)，短期回调风险高"
        elif rsi_mid > 60:
            result.rsi_status = RSIStatus.STRONG_BUY
            result.rsi_signal = f"✅ RSI强势({rsi_mid:.1f})，多头力量充足"
        elif rsi_mid >= 40:
            result.rsi_status = RSIStatus.NEUTRAL
            result.rsi_signal = f" RSI中性({rsi_mid:.1f})，震荡整理中"
        elif rsi_mid >= self.RSI_OVERSOLD:
            result.rsi_status = RSIStatus.WEAK
            result.rsi_signal = f"⚡ RSI弱势({rsi_mid:.1f})，关注反弹"
        else:
            result.rsi_status = RSIStatus.OVERSOLD
            result.rsi_signal = f"⭐ RSI超卖({rsi_mid:.1f}<30)，反弹机会大"

    def _generate_signal(self, result: TrendAnalysisResult) -> None:
        """
        生成买入信号

        综合评分系统：
        - 趋势（30分）：多头排列得分高
        - 乖离率（20分）：接近 MA5 得分高
        - 量能（15分）：缩量回调得分高
        - 支撑（10分）：获得均线支撑得分高
        - MACD（15分）：金叉和多头得分高
        - RSI（10分）：超卖和强势得分高
        """
        score = 0
        reasons = []
        risks = []

        # === 趋势评分（30分）===
        trend_scores = {
            TrendStatus.STRONG_BULL: 30,
            TrendStatus.BULL: 26,
            TrendStatus.WEAK_BULL: 18,
            TrendStatus.CONSOLIDATION: 12,
            TrendStatus.WEAK_BEAR: 8,
            TrendStatus.BEAR: 4,
            TrendStatus.STRONG_BEAR: 0,
        }
        trend_score = trend_scores.get(result.trend_status, 12)
        score += trend_score

        if result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            reasons.append(f"✅ {result.trend_status.value}，顺势做多")
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            risks.append(f"⚠️ {result.trend_status.value}，不宜做多")

        # === 乖离率评分（20分，强势趋势补偿）===
        bias = result.bias_ma5
        if bias != bias or bias is None:  # NaN or None defense
            bias = 0.0
        base_threshold = get_config().bias_threshold

        # Strong trend compensation: relax threshold for STRONG_BULL with high strength
        trend_strength = result.trend_strength if result.trend_strength == result.trend_strength else 0.0
        if result.trend_status == TrendStatus.STRONG_BULL and (trend_strength or 0) >= 70:
            effective_threshold = base_threshold * 1.5
            is_strong_trend = True
        else:
            effective_threshold = base_threshold
            is_strong_trend = False

        if bias < 0:
            # Price below MA5 (pullback)
            if bias > -3:
                score += 20
                reasons.append(f"✅ 价格略低于MA5({bias:.1f}%)，回踩买点")
            elif bias > -5:
                score += 16
                reasons.append(f"✅ 价格回踩MA5({bias:.1f}%)，观察支撑")
            else:
                score += 8
                risks.append(f"⚠️ 乖离率过大({bias:.1f}%)，可能破位")
        elif bias < 2:
            score += 18
            reasons.append(f"✅ 价格贴近MA5({bias:.1f}%)，介入好时机")
        elif bias < base_threshold:
            score += 14
            reasons.append(f"⚡ 价格略高于MA5({bias:.1f}%)，可小仓介入")
        elif bias > effective_threshold:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{effective_threshold:.1f}%)，严禁追高！"
            )
        elif bias > base_threshold and is_strong_trend:
            score += 10
            reasons.append(
                f"⚡ 强势趋势中乖离率偏高({bias:.1f}%)，可轻仓追踪"
            )
        else:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{base_threshold:.1f}%)，严禁追高！"
            )

        # === 量能评分（15分）===
        volume_scores = {
            VolumeStatus.SHRINK_VOLUME_DOWN: 15,  # 缩量回调最佳
            VolumeStatus.HEAVY_VOLUME_UP: 12,     # 放量上涨次之
            VolumeStatus.NORMAL: 10,
            VolumeStatus.SHRINK_VOLUME_UP: 6,     # 无量上涨较差
            VolumeStatus.HEAVY_VOLUME_DOWN: 0,    # 放量下跌最差
        }
        vol_score = volume_scores.get(result.volume_status, 8)
        score += vol_score

        if result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN:
            reasons.append("✅ 缩量回调，主力洗盘")
        elif result.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
            risks.append("⚠️ 放量下跌，注意风险")

        # === 支撑评分（10分）===
        if result.support_ma5:
            score += 5
            reasons.append("✅ MA5支撑有效")
        if result.support_ma10:
            score += 5
            reasons.append("✅ MA10支撑有效")

        # === MACD 评分（15分）===
        macd_scores = {
            MACDStatus.GOLDEN_CROSS_ZERO: 15,  # 零轴上金叉最强
            MACDStatus.GOLDEN_CROSS: 12,      # 金叉
            MACDStatus.CROSSING_UP: 10,       # 上穿零轴
            MACDStatus.BULLISH: 8,            # 多头
            MACDStatus.BEARISH: 2,            # 空头
            MACDStatus.CROSSING_DOWN: 0,       # 下穿零轴
            MACDStatus.DEATH_CROSS: 0,        # 死叉
        }
        macd_score = macd_scores.get(result.macd_status, 5)
        score += macd_score

        if result.macd_status in [MACDStatus.GOLDEN_CROSS_ZERO, MACDStatus.GOLDEN_CROSS]:
            reasons.append(f"✅ {result.macd_signal}")
        elif result.macd_status in [MACDStatus.DEATH_CROSS, MACDStatus.CROSSING_DOWN]:
            risks.append(f"⚠️ {result.macd_signal}")
        else:
            reasons.append(result.macd_signal)

        # === RSI 评分（10分）===
        rsi_scores = {
            RSIStatus.OVERSOLD: 10,       # 超卖最佳
            RSIStatus.STRONG_BUY: 8,     # 强势
            RSIStatus.NEUTRAL: 5,        # 中性
            RSIStatus.WEAK: 3,            # 弱势
            RSIStatus.OVERBOUGHT: 0,       # 超买最差
        }
        rsi_score = rsi_scores.get(result.rsi_status, 5)
        score += rsi_score

        if result.rsi_status in [RSIStatus.OVERSOLD, RSIStatus.STRONG_BUY]:
            reasons.append(f"✅ {result.rsi_signal}")
        elif result.rsi_status == RSIStatus.OVERBOUGHT:
            risks.append(f"⚠️ {result.rsi_signal}")
        else:
            reasons.append(result.rsi_signal)

        # === K线结构修正（不改变原100分体系，只做小幅加减分）===
        if result.kline_structure_score >= 75:
            score += 5
            reasons.append(f"✅ K线结构偏强：{result.kline_summary}")
        elif result.kline_structure_score <= 35:
            score -= 8
            risks.append(f"⚠️ K线结构偏弱：{result.kline_summary}")
        elif result.kline_summary:
            reasons.append(f"ℹ️ K线结构参考：{result.kline_summary}")

        # 强风险信号直接压制进攻性
        serious_kline_risks = [
            r for r in result.kline_risks
            if ("放量跌破MA20" in r) or ("跌破近20日低点" in r) or ("放量实体阴线" in r)
        ]
        if serious_kline_risks:
            score -= 10
            risks.extend([f"⚠️ {r}" for r in serious_kline_risks[:2]])

        # 分数边界保护
        score = max(0, min(100, score))

        # === 综合判断 ===
        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks

        # 生成买入信号（调整阈值以适应新的100分制）
        if score >= 75 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 60 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL, TrendStatus.WEAK_BULL]:
            result.buy_signal = BuySignal.BUY
        elif score >= 45:
            result.buy_signal = BuySignal.HOLD
        elif score >= 30:
            result.buy_signal = BuySignal.WAIT
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            result.buy_signal = BuySignal.STRONG_SELL
        else:
            result.buy_signal = BuySignal.SELL
    
    def format_analysis(self, result: TrendAnalysisResult) -> str:
        """
        格式化分析结果为文本

        Args:
            result: 分析结果

        Returns:
            格式化的分析文本
        """
        lines = [
            f"=== {result.code} 趋势分析 ===",
            f"",
            f"📊 趋势判断: {result.trend_status.value}",
            f"   均线排列: {result.ma_alignment}",
            f"   趋势强度: {result.trend_strength}/100",
            f"",
            f"📈 均线数据:",
            f"   现价: {result.current_price:.2f}",
            f"   MA5:  {result.ma5:.2f} (乖离 {result.bias_ma5:+.2f}%)",
            f"   MA10: {result.ma10:.2f} (乖离 {result.bias_ma10:+.2f}%)",
            f"   MA20: {result.ma20:.2f} (乖离 {result.bias_ma20:+.2f}%)",
            f"",
            f"📊 量能分析: {result.volume_status.value}",
            f"   量比(vs5日): {result.volume_ratio_5d:.2f}",
            f"   量能趋势: {result.volume_trend}",
            f"",
            f"📈 MACD指标: {result.macd_status.value}",
            f"   DIF: {result.macd_dif:.4f}",
            f"   DEA: {result.macd_dea:.4f}",
            f"   MACD: {result.macd_bar:.4f}",
            f"   信号: {result.macd_signal}",
            f"",
            f"📊 RSI指标: {result.rsi_status.value}",
            f"   RSI(6): {result.rsi_6:.1f}",
            f"   RSI(12): {result.rsi_12:.1f}",
            f"   RSI(24): {result.rsi_24:.1f}",
            f"   信号: {result.rsi_signal}",
            f"",
            f"🎯 操作建议: {result.buy_signal.value}",
            f"   综合评分: {result.signal_score}/100",
        ]

        if result.signal_reasons:
            lines.append(f"")
            lines.append(f"✅ 买入理由:")
            for reason in result.signal_reasons:
                lines.append(f"   {reason}")

        if result.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ 风险因素:")
            for risk in result.risk_factors:
                lines.append(f"   {risk}")

        return "\n".join(lines)


def analyze_stock(
    df: pd.DataFrame,
    code: str,
    realtime_change_pct: Optional[float] = None,
) -> TrendAnalysisResult:
    """
    便捷函数：分析单只股票
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        code: 股票代码
        
    Returns:
        TrendAnalysisResult 分析结果
    """
    analyzer = StockTrendAnalyzer()
    return analyzer.analyze(df, code)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据测试
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    # 模拟多头排列的数据
    base_price = 10.0
    prices = [base_price]
    for i in range(59):
        change = np.random.randn() * 0.02 + 0.003  # 轻微上涨趋势
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices],
    })
    
    analyzer = StockTrendAnalyzer()
    result = analyzer.analyze(df, '000001')
    print(analyzer.format_analysis(result))
