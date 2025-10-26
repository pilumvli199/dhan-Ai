"""
🤖 ADVANCED NIFTY 50 STOCKS TRADING BOT v8.0 - AGGRESSIVE HIGH PROBABILITY
Version: 8.0 - AGGRESSIVE MODE (Fewer but Stronger Signals)
Expected: 2-4 premium signals/day | Win Rate Target: 75-85%
"""

import asyncio
import os
import json
import csv
import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import traceback
import pytz
import redis
from dataclasses import dataclass
import html
import re

from telegram import Bot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Bot Configuration - AGGRESSIVE HIGH PROBABILITY MODE"""
    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    DHAN_API_BASE = "https://api.dhan.co"
    DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
    DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
    DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
    DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    SCAN_INTERVAL = 900
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 86400
    
    # AGGRESSIVE HIGH PROBABILITY FILTERS
    CONFIDENCE_THRESHOLD = 80
    MIN_CHART_SCORE = 22
    MIN_OPTION_SCORE = 25
    MIN_OI_DIVERGENCE_PCT = 5.0
    MIN_VOLUME_INCREASE_PCT = 50.0
    MIN_TOTAL_OI = 50000
    PCR_BULLISH_MIN = 1.2
    PCR_BEARISH_MAX = 0.8
    MIN_ATR_PCT = 0.5
    MIN_STRONG_PATTERNS = 2
    REQUIRE_MTF_ALIGNMENT = True
    SKIP_OPENING_MINUTES = 15
    SKIP_CLOSING_MINUTES = 30
    
    LOOKBACK_DAYS = 10
    ATM_STRIKE_RANGE = 11
    MIN_CANDLES_REQUIRED = 50
    
    NIFTY_50_STOCKS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
        "ASIANPAINT", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "DMART", "WIPRO", "BAJAJFINSV", "ADANIENT", "ONGC",
        "NTPC", "TECHM", "POWERGRID", "M&M", "TATASTEEL",
        "INDUSINDBK", "COALINDIA", "JSWSTEEL", "GRASIM", "BRITANNIA",
        "TATACONSUM", "HINDALCO", "EICHERMOT", "ADANIPORTS", "APOLLOHOSP",
        "SBILIFE", "BAJAJ-AUTO", "CIPLA", "DIVISLAB", "HDFCLIFE",
        "BPCL", "HEROMOTOCO", "TATAMOTORS", "UPL", "DRREDDY"
    ]


@dataclass
class OIData:
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_oi_change: int = 0
    pe_oi_change: int = 0
    ce_iv: float = 0.0
    pe_iv: float = 0.0
    pcr_at_strike: float = 0.0
    oi_flow_type: str = "UNKNOWN"


@dataclass
class CandlePattern:
    timestamp: str
    pattern_name: str
    candle_type: str
    body_size: float
    upper_wick: float
    lower_wick: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    significance: str
    volume_confirmed: bool = False


@dataclass
class AggregateOIAnalysis:
    total_ce_oi: int
    total_pe_oi: int
    total_ce_volume: int
    total_pe_volume: int
    total_ce_oi_change: int
    total_pe_oi_change: int
    total_ce_volume_change: int
    total_pe_volume_change: int
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    ce_volume_change_pct: float
    pe_volume_change_pct: float
    pcr: float
    overall_sentiment: str


@dataclass
class HighProbabilityCheck:
    passed: bool
    confidence_check: bool
    oi_divergence_check: bool
    volume_check: bool
    pcr_check: bool
    mtf_alignment_check: bool
    pattern_strength_check: bool
    liquidity_check: bool
    volatility_check: bool
    time_check: bool
    score_check: bool
    rejection_reason: str = ""


class RedisCache:
    def __init__(self):
        try:
            logger.info("Connecting to Redis...")
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis connected!")
        except Exception as e:
            logger.error(f"Redis failed: {e}")
            self.redis_client = None
    
    def store_option_chain(self, symbol: str, oi_data: List[OIData], spot_price: float):
        try:
            if not self.redis_client:
                return False
            
            key = f"oi_data:{symbol}"
            value = json.dumps({
                'spot_price': spot_price,
                'strikes': [
                    {
                        'strike': oi.strike,
                        'ce_oi': oi.ce_oi,
                        'pe_oi': oi.pe_oi,
                        'ce_volume': oi.ce_volume,
                        'pe_volume': oi.pe_volume,
                        'ce_iv': oi.ce_iv,
                        'pe_iv': oi.pe_iv,
                        'oi_flow_type': oi.oi_flow_type
                    }
                    for oi in oi_data
                ],
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            return True
        except Exception as e:
            logger.error(f"Redis store error: {e}")
            return False
    
    def get_oi_comparison(self, symbol: str, current_oi: List[OIData], current_price: float) -> Dict:
        try:
            if not self.redis_client:
                return {'change': 'NO_CACHE', 'flow_summary': {}, 'aggregate_analysis': None}
            
            key = f"oi_data:{symbol}"
            cached = self.redis_client.get(key)
            
            if not cached:
                return {'change': 'FIRST_SCAN', 'flow_summary': {}, 'aggregate_analysis': None}
            
            old_data = json.loads(cached)
            old_strikes = {s['strike']: s for s in old_data['strikes']}
            previous_price = old_data.get('spot_price', current_price)
            
            price_change = current_price - previous_price
            price_direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
            
            total_ce_oi_old = sum(s['ce_oi'] for s in old_data['strikes'])
            total_pe_oi_old = sum(s['pe_oi'] for s in old_data['strikes'])
            total_ce_volume_old = sum(s['ce_volume'] for s in old_data['strikes'])
            total_pe_volume_old = sum(s['pe_volume'] for s in old_data['strikes'])
            
            total_ce_oi_new = sum(oi.ce_oi for oi in current_oi)
            total_pe_oi_new = sum(oi.pe_oi for oi in current_oi)
            total_ce_volume_new = sum(oi.ce_volume for oi in current_oi)
            total_pe_volume_new = sum(oi.pe_volume for oi in current_oi)
            
            ce_oi_change = total_ce_oi_new - total_ce_oi_old
            pe_oi_change = total_pe_oi_new - total_pe_oi_old
            ce_volume_change = total_ce_volume_new - total_ce_volume_old
            pe_volume_change = total_pe_volume_new - total_pe_volume_old
            
            ce_oi_change_pct = (ce_oi_change / total_ce_oi_old * 100) if total_ce_oi_old > 0 else 0
            pe_oi_change_pct = (pe_oi_change / total_pe_oi_old * 100) if total_pe_oi_old > 0 else 0
            ce_volume_change_pct = (ce_volume_change / total_ce_volume_old * 100) if total_ce_volume_old > 0 else 0
            pe_volume_change_pct = (pe_volume_change / total_pe_volume_old * 100) if total_pe_volume_old > 0 else 0
            
            pcr = total_pe_oi_new / total_ce_oi_new if total_ce_oi_new > 0 else 0
            
            sentiment = "NEUTRAL"
            if pe_oi_change_pct > 5 and pe_oi_change_pct > ce_oi_change_pct:
                sentiment = "BULLISH"
            elif ce_oi_change_pct > 5 and ce_oi_change_pct > pe_oi_change_pct:
                sentiment = "BEARISH"
            elif pcr > 1.3:
                sentiment = "BULLISH"
            elif pcr < 0.7:
                sentiment = "BEARISH"
            
            aggregate_analysis = AggregateOIAnalysis(
                total_ce_oi=total_ce_oi_new,
                total_pe_oi=total_pe_oi_new,
                total_ce_volume=total_ce_volume_new,
                total_pe_volume=total_pe_volume_new,
                total_ce_oi_change=ce_oi_change,
                total_pe_oi_change=pe_oi_change,
                total_ce_volume_change=ce_volume_change,
                total_pe_volume_change=pe_volume_change,
                ce_oi_change_pct=ce_oi_change_pct,
                pe_oi_change_pct=pe_oi_change_pct,
                ce_volume_change_pct=ce_volume_change_pct,
                pe_volume_change_pct=pe_volume_change_pct,
                pcr=pcr,
                overall_sentiment=sentiment
            )
            
            flow_summary = {
                'LONG_BUILDUP': [],
                'SHORT_BUILDUP': [],
                'LONG_UNWINDING': [],
                'SHORT_COVERING': []
            }
            
            for curr_oi in current_oi:
                old = old_strikes.get(curr_oi.strike, {})
                ce_oi_change_val = curr_oi.ce_oi - old.get('ce_oi', 0)
                pe_oi_change_val = curr_oi.pe_oi - old.get('pe_oi', 0)
                total_oi_change = ce_oi_change_val + pe_oi_change_val
                
                if abs(total_oi_change) > 100:
                    if price_direction == "UP" and total_oi_change > 0 and pe_oi_change_val > ce_oi_change_val:
                        curr_oi.oi_flow_type = "LONG_BUILDUP"
                        flow_summary['LONG_BUILDUP'].append({
                            'strike': curr_oi.strike,
                            'ce_oi_change': ce_oi_change_val,
                            'pe_oi_change': pe_oi_change_val,
                            'total_change': total_oi_change
                        })
                    elif price_direction == "DOWN" and total_oi_change > 0 and ce_oi_change_val > pe_oi_change_val:
                        curr_oi.oi_flow_type = "SHORT_BUILDUP"
                        flow_summary['SHORT_BUILDUP'].append({
                            'strike': curr_oi.strike,
                            'ce_oi_change': ce_oi_change_val,
                            'pe_oi_change': pe_oi_change_val,
                            'total_change': total_oi_change
                        })
                    elif price_direction == "DOWN" and total_oi_change < 0 and pe_oi_change_val < ce_oi_change_val:
                        curr_oi.oi_flow_type = "LONG_UNWINDING"
                        flow_summary['LONG_UNWINDING'].append({
                            'strike': curr_oi.strike,
                            'ce_oi_change': ce_oi_change_val,
                            'pe_oi_change': pe_oi_change_val,
                            'total_change': total_oi_change
                        })
                    elif price_direction == "UP" and total_oi_change < 0 and ce_oi_change_val < pe_oi_change_val:
                        curr_oi.oi_flow_type = "SHORT_COVERING"
                        flow_summary['SHORT_COVERING'].append({
                            'strike': curr_oi.strike,
                            'ce_oi_change': ce_oi_change_val,
                            'pe_oi_change': pe_oi_change_val,
                            'total_change': total_oi_change
                        })
            
            for flow_type in flow_summary:
                flow_summary[flow_type].sort(key=lambda x: abs(x['total_change']), reverse=True)
            
            time_diff = (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                        datetime.fromisoformat(old_data['timestamp'])).seconds / 60
            
            return {
                'change': 'UPDATED',
                'price_movement': price_direction,
                'price_change': price_change,
                'flow_summary': flow_summary,
                'aggregate_analysis': aggregate_analysis,
                'time_diff': time_diff,
                'old_spot': previous_price
            }
            
        except Exception as e:
            logger.error(f"Redis comparison error: {e}")
            return {'change': 'ERROR', 'flow_summary': {}, 'aggregate_analysis': None}


class AdvancedPatternDetector:
    @staticmethod
    def detect_patterns(df: pd.DataFrame, lookback: int = 50) -> List[CandlePattern]:
        patterns = []
        actual_lookback = min(lookback, len(df))
        recent_df = df.tail(actual_lookback)
        avg_volume = recent_df['volume'].mean()
        
        for i in range(len(recent_df)):
            row = recent_df.iloc[i]
            idx = recent_df.index[i]
            
            body = abs(row['close'] - row['open'])
            candle_range = row['high'] - row['low']
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            is_bullish = row['close'] > row['open']
            candle_type = "BULLISH" if is_bullish else "BEARISH"
            volume_confirmed = row['volume'] > avg_volume * 1.2
            
            pattern_name = "NORMAL"
            significance = "WEAK"
            
            if candle_range > 0:
                body_ratio = body / candle_range
                
                if body_ratio < 0.1:
                    pattern_name = "DOJI"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                elif lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.3:
                    pattern_name = "HAMMER"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                elif upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.3:
                    pattern_name = "INVERTED_HAMMER"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                elif upper_wick > body * 2 and lower_wick < body * 0.5 and not is_bullish:
                    pattern_name = "SHOOTING_STAR"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                elif upper_wick < body * 0.1 and lower_wick < body * 0.1 and body_ratio > 0.8:
                    pattern_name = "MARUBOZU_BULLISH" if is_bullish else "MARUBOZU_BEARISH"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                elif body_ratio < 0.3 and upper_wick > body and lower_wick > body:
                    pattern_name = "SPINNING_TOP"
                    significance = "MODERATE"
                elif i > 0:
                    prev_row = recent_df.iloc[i-1]
                    prev_body = abs(prev_row['close'] - prev_row['open'])
                    prev_is_bullish = prev_row['close'] > prev_row['open']
                    
                    if (is_bullish and not prev_is_bullish and row['open'] < prev_row['close'] and 
                        row['close'] > prev_row['open'] and body > prev_body * 0.7):
                        pattern_name = "BULLISH_ENGULFING"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    elif (not is_bullish and prev_is_bullish and row['open'] > prev_row['close'] and 
                          row['close'] < prev_row['open'] and body > prev_body * 0.7):
                        pattern_name = "BEARISH_ENGULFING"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    elif (is_bullish and not prev_is_bullish and row['open'] < prev_row['low'] and
                          row['close'] > (prev_row['open'] + prev_row['close']) / 2):
                        pattern_name = "PIERCING_LINE"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    elif (not is_bullish and prev_is_bullish and row['open'] > prev_row['high'] and
                          row['close'] < (prev_row['open'] + prev_row['close']) / 2):
                        pattern_name = "DARK_CLOUD_COVER"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    elif (body < prev_body * 0.5 and row['high'] < prev_row['high'] and row['low'] > prev_row['low']):
                        pattern_name = "HARAMI_BULLISH" if is_bullish else "HARAMI_BEARISH"
                        significance = "MODERATE"
                
                if i > 1:
                    prev1 = recent_df.iloc[i-1]
                    prev2 = recent_df.iloc[i-2]
                    
                    if (is_bullish and prev1['close'] > prev1['open'] and prev2['close'] > prev2['open'] and
                        row['close'] > prev1['close'] > prev2['close']):
                        pattern_name = "THREE_WHITE_SOLDIERS"
                        significance = "STRONG"
                    elif (not is_bullish and prev1['close'] < prev1['open'] and prev2['close'] < prev2['open'] and
                          row['close'] < prev1['close'] < prev2['close']):
                        pattern_name = "THREE_BLACK_CROWS"
                        significance = "STRONG"
                
                if pattern_name == "NORMAL":
                    if body_ratio > 0.7:
                        significance = "STRONG"
                    elif body_ratio > 0.4:
                        significance = "MODERATE"
            
            patterns.append(CandlePattern(
                timestamp=idx.strftime('%Y-%m-%d %H:%M'),
                pattern_name=pattern_name,
                candle_type=candle_type,
                body_size=body,
                upper_wick=upper_wick,
                lower_wick=lower_wick,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                significance=significance,
                volume_confirmed=volume_confirmed
            ))
        
        return patterns


class ChartAnalyzer:
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        if len(df) < 20:
            return "INSUFFICIENT_DATA"
        
        sma_len = min(50, len(df))
        sma_20_len = min(20, len(df))
        recent = df.tail(sma_len)
        sma_20 = recent['close'].tail(sma_20_len).mean()
        sma_50 = recent['close'].mean()
        current_price = recent['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "UPTREND"
        elif current_price < sma_20 < sma_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            current = df['close'].iloc[-1]
            return {
                'nearest_support': current * 0.98,
                'nearest_resistance': current * 1.02,
                'swing_high': df['high'].max(),
                'swing_low': df['low'].min()
            }
        
        lookback = min(100, len(df))
        recent = df.tail(lookback)
        current = recent['close'].iloc[-1]
        pivot_window = min(50, len(recent))
        highs = recent['high'].tail(pivot_window)
        lows = recent['low'].tail(pivot_window)
        
        resistance_levels = []
        support_levels = []
        window_size = min(5, len(highs) // 3)
        
        if window_size >= 2:
            for i in range(window_size, len(highs) - window_size):
                if all(highs.iloc[i] >= highs.iloc[i-j] for j in range(1, window_size+1)) and \
                   all(highs.iloc[i] >= highs.iloc[i+j] for j in range(1, window_size+1)):
                    resistance_levels.append(highs.iloc[i])
            
            for i in range(window_size, len(lows) - window_size):
                if all(lows.iloc[i] <= lows.iloc[i-j] for j in range(1, window_size+1)) and \
                   all(lows.iloc[i] <= lows.iloc[i+j] for j in range(1, window_size+1)):
                    support_levels.append(lows.iloc[i])
        
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            clustered.append(np.mean(current_cluster))
            return clustered
        
        resistance = cluster(resistance_levels)
        support = cluster(support_levels)
        resistance = [r for r in resistance if 0.01 <= (r - current)/current <= 0.08]
        support = [s for s in support if 0.01 <= (current - s)/current <= 0.08]
        
        return {
            'nearest_support': min(support) if support else current * 0.98,
            'nearest_resistance': min(resistance) if resistance else current * 1.02,
            'swing_high': highs.max(),
            'swing_low': lows.min()
        }
    
    @staticmethod
    def calculate_atr_percentage(df: pd.DataFrame) -> float:
        if len(df) < 14:
            return 0.0
        
        recent = df.tail(14)
        tr_values = []
        
        for i in range(1, len(recent)):
            high = recent['high'].iloc[i]
            low = recent['low'].iloc[i]
            prev_close = recent['close'].iloc[i-1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        
        atr = np.mean(tr_values)
        current_price = recent['close'].iloc[-1]
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        return atr_pct


class HighProbabilityFilter:
    @staticmethod
    def check_all_filters(symbol: str, analysis: Dict, oi_comparison: Dict,
                         patterns_dict: Dict, mtf_data: Dict, oi_data: List[OIData]) -> HighProbabilityCheck:
        
        checks = {
            'confidence_check': False,
            'oi_divergence_check': False,
            'volume_check': False,
            'pcr_check': False,
            'mtf_alignment_check': False,
            'pattern_strength_check': False,
            'liquidity_check': False,
            'volatility_check': False,
            'time_check': False,
            'score_check': False
        }
        
        rejection_reason = ""
        
        confidence = analysis.get('confidence', 0)
        if confidence >= Config.CONFIDENCE_THRESHOLD:
            checks['confidence_check'] = True
        else:
            rejection_reason = f"Confidence {confidence}% < {Config.CONFIDENCE_THRESHOLD}%"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        aggregate = oi_comparison.get('aggregate_analysis')
        if aggregate:
            oi_divergence = abs(aggregate.pe_oi_change_pct - aggregate.ce_oi_change_pct)
            if oi_divergence >= Config.MIN_OI_DIVERGENCE_PCT:
                checks['oi_divergence_check'] = True
            else:
                rejection_reason = f"OI Divergence {oi_divergence:.1f}% < {Config.MIN_OI_DIVERGENCE_PCT}%"
                logger.info(f"{symbol}: {rejection_reason}")
                return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        else:
            rejection_reason = "First scan - No OI comparison"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        opportunity = analysis.get('opportunity', 'WAIT')
        if opportunity == "PE_BUY":
            volume_increase = aggregate.pe_volume_change_pct
        elif opportunity == "CE_BUY":
            volume_increase = aggregate.ce_volume_change_pct
        else:
            rejection_reason = "Opportunity is WAIT"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        if volume_increase >= Config.MIN_VOLUME_INCREASE_PCT:
            checks['volume_check'] = True
        else:
            rejection_reason = f"Volume increase {volume_increase:.1f}% < {Config.MIN_VOLUME_INCREASE_PCT}%"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        pcr = aggregate.pcr
        if opportunity == "PE_BUY":
            if pcr >= Config.PCR_BULLISH_MIN:
                checks['pcr_check'] = True
            else:
                rejection_reason = f"PCR {pcr:.2f} < {Config.PCR_BULLISH_MIN} (not extreme bullish)"
                logger.info(f"{symbol}: {rejection_reason}")
                return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        elif opportunity == "CE_BUY":
            if pcr <= Config.PCR_BEARISH_MAX:
                checks['pcr_check'] = True
            else:
                rejection_reason = f"PCR {pcr:.2f} > {Config.PCR_BEARISH_MAX} (not extreme bearish)"
                logger.info(f"{symbol}: {rejection_reason}")
                return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        base_tf = '5m' if '5m' in mtf_data else '15m'
        trend_1h = ChartAnalyzer.identify_trend(mtf_data.get('1h', mtf_data[base_tf]))
        trend_entry = ChartAnalyzer.identify_trend(mtf_data[base_tf])
        
        if Config.REQUIRE_MTF_ALIGNMENT:
            if opportunity == "PE_BUY":
                if trend_1h == "DOWNTREND" and trend_entry == "DOWNTREND":
                    checks['mtf_alignment_check'] = True
                else:
                    rejection_reason = f"MTF not aligned: 1H={trend_1h}, {base_tf}={trend_entry}"
                    logger.info(f"{symbol}: {rejection_reason}")
                    return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
            elif opportunity == "CE_BUY":
                if trend_1h == "UPTREND" and trend_entry == "UPTREND":
                    checks['mtf_alignment_check'] = True
                else:
                    rejection_reason = f"MTF not aligned: 1H={trend_1h}, {base_tf}={trend_entry}"
                    logger.info(f"{symbol}: {rejection_reason}")
                    return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        else:
            checks['mtf_alignment_check'] = True
        
        entry_tf_patterns = patterns_dict.get(base_tf, [])
        recent_patterns = entry_tf_patterns[-10:] if len(entry_tf_patterns) >= 10 else entry_tf_patterns
        strong_patterns = [p for p in recent_patterns if p.significance == "STRONG"]
        
        if len(strong_patterns) >= Config.MIN_STRONG_PATTERNS:
            checks['pattern_strength_check'] = True
        else:
            rejection_reason = f"Strong patterns {len(strong_patterns)} < {Config.MIN_STRONG_PATTERNS}"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        total_oi = aggregate.total_ce_oi + aggregate.total_pe_oi
        if total_oi >= Config.MIN_TOTAL_OI:
            checks['liquidity_check'] = True
        else:
            rejection_reason = f"Total OI {total_oi:,} < {Config.MIN_TOTAL_OI:,}"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        atr_pct = ChartAnalyzer.calculate_atr_percentage(mtf_data[base_tf])
        if atr_pct >= Config.MIN_ATR_PCT:
            checks['volatility_check'] = True
        else:
            rejection_reason = f"ATR {atr_pct:.2f}% < {Config.MIN_ATR_PCT}%"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        hour = now_ist.hour
        minute = now_ist.minute
        
        if hour == 9 and minute < 15 + Config.SKIP_OPENING_MINUTES:
            rejection_reason = f"Market opening period"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        if hour == 15 or (hour == 14 and minute >= (60 - Config.SKIP_CLOSING_MINUTES)):
            rejection_reason = f"Market closing period"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        checks['time_check'] = True
        
        scoring = analysis.get('scoring_breakup', {})
        chart_score = scoring.get('chart_setup', 0)
        option_score = scoring.get('option_flow', 0)
        
        if chart_score >= Config.MIN_CHART_SCORE and option_score >= Config.MIN_OPTION_SCORE:
            checks['score_check'] = True
        else:
            rejection_reason = f"Scores: Chart {chart_score}/{Config.MIN_CHART_SCORE}, Options {option_score}/{Config.MIN_OPTION_SCORE}"
            logger.info(f"{symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        
        logger.info(f"{symbol}: ALL 10 AGGRESSIVE FILTERS PASSED!")
        return HighProbabilityCheck(passed=True, rejection_reason="", **checks)


class DhanAPI:
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("DhanAPI initialized")
    
    async def load_security_ids(self):
        try:
            logger.info("Loading NIFTY 50 stock security IDs...")
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                return False
            
            csv_reader = csv.DictReader(io.StringIO(response.text))
            all_rows = list(csv_reader)
            
            for stock_symbol in Config.NIFTY_50_STOCKS:
                for row in all_rows:
                    try:
                        trading_symbol = row.get('SEM_TRADING_SYMBOL', '').strip()
                        segment = row.get('SEM_SEGMENT', '').strip()
                        exch_segment = row.get('SEM_EXM_EXCH_ID', '').strip()
                        
                        if (segment == 'E' and exch_segment == 'NSE' and trading_symbol == stock_symbol):
                            sec_id = row.get('SEM_SMST_SECURITY_ID', '').strip()
                            if sec_id:
                                instrument_type = "INDEX" if stock_symbol in ["NIFTY", "BANKNIFTY"] else "STOCK"
                                
                                self.security_id_map[stock_symbol] = {
                                    'security_id': int(sec_id),
                                    'segment': 'NSE_EQ',
                                    'trading_symbol': trading_symbol,
                                    'instrument': 'EQUITY',
                                    'instrument_type': instrument_type
                                }
                                logger.info(f"{stock_symbol}: ID={sec_id}")
                                break
                    except Exception:
                        continue
            
            logger.info(f"Loaded {len(self.security_id_map)}/50 stocks")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        try:
            payload = {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": "NSE_EQ"
            }
            
            response = requests.post(
                Config.DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    return data['data'][0]
            return None
            
        except Exception as e:
            logger.error(f"Expiry error: {e}")
            return None
    
    def get_multi_timeframe_data(self, security_id: int, segment: str, 
                                 symbol: str, instrument_type: str) -> Optional[Dict[str, pd.DataFrame]]:
        try:
            logger.info(f"Fetching MTF data for {symbol}")
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=Config.LOOKBACK_DAYS)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data['open']) == 0:
                return None
            
            df_base = pd.DataFrame({
                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            
            df_base = df_base.dropna()
            df_base.set_index('timestamp', inplace=True)
            
            logger.info(f"Received {len(df_base)} base candles")
            
            if len(df_base) < Config.MIN_CANDLES_REQUIRED:
                logger.warning(f"Only {len(df_base)} candles")
            
            result = {}
            
            if instrument_type == "INDEX":
                result['5m'] = df_base.copy()
                
                result['15m'] = df_base.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                result['1h'] = df_base.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
            else:
                if len(df_base) > 1:
                    time_diff = (df_base.index[1] - df_base.index[0]).seconds / 60
                else:
                    time_diff = 15
                
                if time_diff <= 5:
                    df_15m = df_base.resample('15min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                else:
                    df_15m = df_base.copy()
                
                result['15m'] = df_15m
                
                result['1h'] = df_15m.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            logger.info(f"{symbol}: MTF data ready")
            for tf, df in result.items():
                logger.info(f"   {tf}: {len(df)} candles")
            
            return result
            
        except Exception as e:
            logger.error(f"MTF data error: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, 
                        symbol: str, spot_price: float) -> Optional[List[OIData]]:
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            response = requests.post(
                Config.DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('data'):
                return None
            
            oc_data = data['data'].get('oc', {})
            
            strikes = [float(s) for s in oc_data.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            logger.info(f"{symbol} ATM: {atm_strike} (Spot: {spot_price:.2f})")
            
            strike_range = Config.ATM_STRIKE_RANGE
            oi_list = []
            
            for strike_str, strike_data in oc_data.items():
                try:
                    strike = float(strike_str)
                    
                    strikes_sorted = sorted(strikes)
                    atm_index = strikes_sorted.index(atm_strike)
                    start_idx = max(0, atm_index - strike_range)
                    end_idx = min(len(strikes_sorted), atm_index + strike_range + 1)
                    valid_strikes = strikes_sorted[start_idx:end_idx]
                    
                    if strike not in valid_strikes:
                        continue
                    
                    ce_data = strike_data.get('ce', {})
                    pe_data = strike_data.get('pe', {})
                    
                    ce_oi = ce_data.get('oi', 0)
                    pe_oi = pe_data.get('oi', 0)
                    
                    pcr = pe_oi / ce_oi if ce_oi > 0 else 0
                    
                    oi_list.append(OIData(
                        strike=strike,
                        ce_oi=ce_oi,
                        pe_oi=pe_oi,
                        ce_volume=ce_data.get('volume', 0),
                        pe_volume=pe_data.get('volume', 0),
                        ce_oi_change=0,
                        pe_oi_change=0,
                        ce_iv=ce_data.get('iv', 0.0),
                        pe_iv=pe_data.get('iv', 0.0),
                        pcr_at_strike=pcr
                    ))
                except Exception:
                    continue
            
            logger.info(f"{symbol}: {len(oi_list)} strikes fetched")
            return oi_list
            
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None


class DeepSeekAnalyzer:
    @staticmethod
    def extract_json_from_response(content: str) -> Optional[Dict]:
        try:
            try:
                return json.loads(content)
            except:
                pass
            
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{]*?"opportunity"[^}]*\})',
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        continue
            
            brace_count = 0
            start_idx = content.find('{')
            if start_idx != -1:
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                return json.loads(content[start_idx:i+1])
                            except:
                                break
            
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None
    
    @staticmethod
    def create_analysis(symbol: str, spot_price: float, mtf_data: Dict,
                       patterns_dict: Dict, oi_data: List[OIData], 
                       oi_comparison: Dict, levels_dict: Dict) -> Optional[Dict]:
        try:
            logger.info(f"DeepSeek: Analyzing {symbol}...")
            
            base_tf = '5m' if '5m' in mtf_data else '15m'
            entry_tf_patterns = patterns_dict.get(base_tf, [])
            
            pattern_summary = []
            for i, p in enumerate(entry_tf_patterns[-10:], 1):
                vol_flag = "V" if p.volume_confirmed else ""
                pattern_summary.append(f"{i}. {p.timestamp} | {p.pattern_name} ({p.significance}) {vol_flag}")
            
            patterns_text = "\n".join(pattern_summary) if pattern_summary else "No significant patterns"
            
            strong_patterns = [p for p in entry_tf_patterns[-20:] if p.significance == "STRONG"]
            pattern_types = {}
            for p in strong_patterns:
                pattern_types[p.pattern_name] = pattern_types.get(p.pattern_name, 0) + 1
            
            aggregate = oi_comparison.get('aggregate_analysis')
            
            if aggregate:
                agg_text = f"""AGGREGATE OI (All {len(oi_data)} Strikes):
CE OI: {aggregate.total_ce_oi:,} ({aggregate.ce_oi_change_pct:+.2f}%)
PE OI: {aggregate.total_pe_oi:,} ({aggregate.pe_oi_change_pct:+.2f}%)
CE Vol: {aggregate.total_ce_volume:,} ({aggregate.ce_volume_change_pct:+.2f}%)
PE Vol: {aggregate.total_pe_volume:,} ({aggregate.pe_volume_change_pct:+.2f}%)
PCR: {aggregate.pcr:.2f} | Sentiment: {aggregate.overall_sentiment}"""
            else:
                agg_text = "First scan"
            
            oi_data_sorted = sorted(oi_data, key=lambda x: x.strike)
            atm_strike = min(oi_data, key=lambda x: abs(x.strike - spot_price)).strike
            
            oi_table = []
            for oi in oi_data_sorted[:8]:
                marker = " ATM" if oi.strike == atm_strike else ""
                oi_table.append(f"Strike {oi.strike}{marker} | CE:{oi.ce_oi:,} PE:{oi.pe_oi:,} | PCR:{oi.pcr_at_strike:.2f}")
            
            oi_text = "\n".join(oi_table)
            
            flow_summary = oi_comparison.get('flow_summary', {})
            flow_parts = []
            for flow_type in ['LONG_BUILDUP', 'SHORT_BUILDUP', 'LONG_UNWINDING', 'SHORT_COVERING']:
                items = flow_summary.get(flow_type, [])
                if items:
                    flow_parts.append(f"{flow_type}: {len(items)}")
            
            flow_text = ", ".join(flow_parts) if flow_parts else "First scan"
            
            levels_1h = levels_dict.get('1h', {})
            levels_entry = levels_dict.get(base_tf, {})
            
            trend_1h = ChartAnalyzer.identify_trend(mtf_data.get('1h', mtf_data[base_tf]))
            trend_entry = ChartAnalyzer.identify_trend(mtf_data[base_tf])
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Analyze {symbol} for AGGRESSIVE HIGH PROBABILITY options trading.

DATA:
Spot: Rs {spot_price:.2f} | ATM: {atm_strike}
1H: {trend_1h} | {base_tf}: {trend_entry}

{agg_text}

PATTERNS (Last 10):
{patterns_text}

Strong: {', '.join([f"{k}({v})" for k, v in pattern_types.items()]) if pattern_types else "None"}

OPTION CHAIN:
{oi_text}

FLOW: {flow_text}

LEVELS:
1H: Supp={levels_1h.get('nearest_support', 'N/A')} Res={levels_1h.get('nearest_resistance', 'N/A')}
{base_tf}: Supp={levels_entry.get('nearest_support', 'N/A')} Res={levels_entry.get('nearest_resistance', 'N/A')}

Reply JSON only:
{{
  "opportunity": "PE_BUY or CE_BUY or WAIT",
  "confidence": 85,
  "scoring_breakup": {{"chart_setup": 25, "option_flow": 28, "risk_management": 18, "probability": 16}},
  "recommended_strike": {int(atm_strike)},
  "entry_price": {spot_price:.2f},
  "target": {spot_price * 1.02:.2f},
  "stop_loss": {spot_price * 0.98:.2f},
  "risk_reward": "1:2",
  "timeframe_confluence": "alignment",
  "pattern_signal": "pattern",
  "oi_flow_signal": "flow",
  "key_levels": "levels",
  "reasoning": "why",
  "probability": "80%",
  "risk_factors": ["risk1", "risk2"]
}}"""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Expert trader. Reply JSON only. 80%+ confidence only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code != 200:
                logger.error(f"DeepSeek error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            analysis = DeepSeekAnalyzer.extract_json_from_response(content)
            
            if not analysis:
                logger.warning(f"Parse failed")
                return None
            
            required = ['opportunity', 'confidence', 'entry_price', 'target', 'stop_loss']
            if all(f in analysis for f in required):
                logger.info(f"DeepSeek: {analysis['opportunity']} | Conf: {analysis['confidence']}%")
                return analysis
            
            return None
            
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            return None


class ChartGenerator:
    @staticmethod
    def create_mtf_chart(mtf_data: Dict, symbol: str, entry: float, 
                        target: float, stop_loss: float, opportunity: str) -> BytesIO:
        try:
            logger.info(f"Generating chart for {symbol}")
            
            base_tf = '5m' if '5m' in mtf_data else '15m'
            chart_df = mtf_data[base_tf].tail(100).copy()
            
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray',
                facecolor='white',
                figcolor='white',
                y_on_right=False
            )
            
            hlines = dict(
                hlines=[entry, target, stop_loss],
                colors=['blue', 'green', 'red'],
                linestyle='--',
                linewidths=2
            )
            
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=s,
                title=f"{symbol} - {opportunity} ({base_tf}) - AGGRESSIVE",
                ylabel='Price',
                volume=True,
                hlines=hlines,
                returnfig=True,
                figsize=(16, 9),
                tight_layout=True
            )
            
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            
            ax.text(len(chart_df), entry, f' Entry: {entry:.2f}', 
                   color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), target, f' Target: {target:.2f}', 
                   color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), stop_loss, f' SL: {stop_loss:.2f}', 
                   color='red', fontweight='bold', va='center', fontsize=10)
            ax.axhline(y=current_price, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.text(len(chart_df), current_price, f' Current: {current_price:.2f}', 
                   color='orange', fontweight='bold', va='center', fontsize=10)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            logger.info(f"Chart generated")
            return buf
            
        except Exception as e:
            logger.error(f"Chart error: {e}")
            return None


class AdvancedFOBot:
    def __init__(self):
        logger.info("Initializing Bot v8.0 - AGGRESSIVE MODE...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.pattern_detector = AdvancedPatternDetector()
        self.chart_analyzer = ChartAnalyzer()
        self.chart_generator = ChartGenerator()
        self.filter = HighProbabilityFilter()
        self.running = True
        
        self.total_scans = 0
        self.filter_rejections = 0
        self.alerts_sent = 0
        
        logger.info("Bot v8.0 initialized")
    
    def is_market_open(self) -> bool:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        return html.escape(str(text))
    
    async def scan_symbol(self, symbol: str, info: Dict):
        try:
            self.total_scans += 1
            
            security_id = info['security_id']
            segment = info['segment']
            instrument_type = info['instrument_type']
            
            logger.info(f"SCANNING: {symbol}")
            
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"{symbol}: No F&O - SKIP")
                self.filter_rejections += 1
                return
            
            mtf_data = self.dhan.get_multi_timeframe_data(security_id, segment, symbol, instrument_type)
            if not mtf_data:
                logger.warning(f"{symbol}: No MTF data - SKIP")
                self.filter_rejections += 1
                return
            
            base_tf = '5m' if '5m' in mtf_data else '15m'
            spot_price = mtf_data[base_tf]['close'].iloc[-1]
            logger.info(f"Spot: {spot_price:.2f}")
            
            if len(mtf_data[base_tf]) < 30:
                logger.warning(f"{symbol}: Insufficient data - SKIP")
                self.filter_rejections += 1
                return
            
            patterns_dict = {}
            levels_dict = {}
            
            for tf, df in mtf_data.items():
                patterns = self.pattern_detector.detect_patterns(df)
                levels = self.chart_analyzer.calculate_support_resistance(df)
                patterns_dict[tf] = patterns
                levels_dict[tf] = levels
            
            oi_data = self.dhan.get_option_chain(security_id, segment, expiry, symbol, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning(f"{symbol}: No OI data - SKIP")
                self.filter_rejections += 1
                return
            
            oi_comparison = self.redis.get_oi_comparison(symbol, oi_data, spot_price)
            self.redis.store_option_chain(symbol, oi_data, spot_price)
            
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                logger.info(f"Aggregate: CE {aggregate.ce_oi_change_pct:+.2f}%, PE {aggregate.pe_oi_change_pct:+.2f}%")
            else:
                logger.info(f"Aggregate: First scan")
            
            analysis = DeepSeekAnalyzer.create_analysis(
                symbol, spot_price, mtf_data, patterns_dict, 
                oi_data, oi_comparison, levels_dict
            )
            
            if not analysis:
                logger.warning(f"{symbol}: No analysis - SKIP")
                self.filter_rejections += 1
                return
            
            logger.info(f"Running AGGRESSIVE FILTER...")
            hp_check = self.filter.check_all_filters(
                symbol, analysis, oi_comparison, patterns_dict, mtf_data, oi_data
            )
            
            if not hp_check.passed:
                logger.info(f"{symbol}: REJECTED - {hp_check.rejection_reason}")
                self.filter_rejections += 1
                return
            
            logger.info(f"{symbol}: AGGRESSIVE HIGH PROBABILITY TRADE!")
            
            chart_image = self.chart_generator.create_mtf_chart(
                mtf_data, symbol,
                analysis.get('entry_price', spot_price),
                analysis.get('target', spot_price * 1.03),
                analysis.get('stop_loss', spot_price * 0.97),
                analysis['opportunity']
            )
            
            await self.send_alert(symbol, spot_price, analysis, mtf_data, 
                                 oi_data, oi_comparison, expiry, chart_image, hp_check, patterns_dict)
            
            self.alerts_sent += 1
            logger.info(f"{symbol}: ALERT SENT!")
            logger.info(f"Stats: Scans={self.total_scans}, Rejected={self.filter_rejections}, Alerts={self.alerts_sent}")
            
        except Exception as e:
            logger.error(f"Scan error {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_alert(self, symbol: str, spot_price: float, analysis: Dict,
                        mtf_data: Dict, oi_data: List[OIData], 
                        oi_comparison: Dict, expiry: str, chart_image: BytesIO,
                        hp_check: HighProbabilityCheck, patterns_dict: Dict):
        try:
            signal_map = {
                "PE_BUY": ("🔴", "PE BUY"),
                "CE_BUY": ("🟢", "CE BUY"),
                "WAIT": ("⚪", "WAIT")
            }
            
            signal_emoji, signal_text = signal_map.get(analysis['opportunity'], ("⚪", "WAIT"))
            
            def safe(val):
                return self.escape_html(val)
            
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
            
            entry = analysis.get('entry_price', spot_price)
            target = analysis.get('target', spot_price * 1.03)
            sl = analysis.get('stop_loss', spot_price * 0.97)
            
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                agg_summary = f"CE {aggregate.ce_oi_change_pct:+.1f}% PE {aggregate.pe_oi_change_pct:+.1f}%"
                sentiment = aggregate.overall_sentiment
                pcr = aggregate.pcr
                volume_change = aggregate.pe_volume_change_pct if analysis['opportunity'] == "PE_BUY" else aggregate.ce_volume_change_pct
                oi_divergence = abs(aggregate.pe_oi_change_pct - aggregate.ce_oi_change_pct)
            else:
                agg_summary = "First scan"
                sentiment = "N/A"
                pcr = 0
                volume_change = 0
                oi_divergence = 0
            
            base_tf = '5m' if '5m' in mtf_data else '15m'
            trend_1h = ChartAnalyzer.identify_trend(mtf_data.get('1h', mtf_data[base_tf]))
            trend_entry = ChartAnalyzer.identify_trend(mtf_data[base_tf])
            
            caption = f"🔥 AGGRESSIVE HIGH PROBABILITY 🎯\n\n📊 {safe(symbol)} {signal_emoji} {signal_text}\n\n✅ Confidence: {safe(analysis['confidence'])}% (≥80%)\n📈 PCR: {pcr:.2f} | Sentiment: {sentiment}\n\n💰 Entry: ₹{safe(f'{entry:.2f}')} → Target: ₹{safe(f'{target:.2f}')}\n🛑 Stop Loss: ₹{safe(f'{sl:.2f}')}\n🎯 Strike: {safe(analysis.get('recommended_strike', 'N/A'))} | Expiry: {expiry}\n\n📊 OI Change: {agg_summary}\n📈 Volume Surge: {volume_change:+.1f}%\n⚡ OI Divergence: {oi_divergence:.1f}%\n📉 MTF: 1H={trend_1h}, {base_tf}={trend_entry}\n\n⏰ {ist_time} IST | v8.0 AGGRESSIVE 🔥"
            
            if chart_image:
                try:
                    await self.bot.send_photo(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        photo=chart_image,
                        caption=caption,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Chart failed: {e}")
                    await self.bot.send_message(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        text=caption,
                        parse_mode='HTML'
                    )
            
            entry_tf_patterns = patterns_dict.get(base_tf, [])
            recent_patterns = entry_tf_patterns[-10:] if len(entry_tf_patterns) >= 10 else entry_tf_patterns
            strong_patterns = [p for p in recent_patterns if p.significance == "STRONG"]
            
            detailed = f"🔥 AGGRESSIVE HIGH PROBABILITY ANALYSIS\n\n✅ ALL 10 AGGRESSIVE FILTERS PASSED:\n\n1️⃣ Confidence: {analysis['confidence']}% (≥80%) ✅\n2️⃣ OI Divergence: {oi_divergence:.1f}% (≥5%) ✅\n3️⃣ Volume Surge: {volume_change:.1f}% (≥50%) ✅\n4️⃣ PCR Extreme: {pcr:.2f} ({'✅ >1.2' if analysis['opportunity']=='PE_BUY' and pcr>=1.2 else '✅ <0.8' if analysis['opportunity']=='CE_BUY' and pcr<=0.8 else '✅'}) ✅\n5️⃣ MTF Aligned: 1H={trend_1h}, {base_tf}={trend_entry} ✅\n6️⃣ Strong Patterns: {len(strong_patterns)} found (≥2) ✅\n7️⃣ Liquidity: {aggregate.total_ce_oi + aggregate.total_pe_oi:,} OI (≥50k) ✅\n8️⃣ Volatility: ATR Passed ✅\n9️⃣ Time Filter: OK ✅\n🔟 Score: Chart {analysis.get('scoring_breakup', {}).get('chart_setup', 0)}/30 + Options {analysis.get('scoring_breakup', {}).get('option_flow', 0)}/30 ✅\n\n━━━━━━━━━━━━━━━━━━━━━━\n\n🕯️ Pattern: {safe(analysis.get('pattern_signal', 'N/A')[:120])}\n\n⛓️ OI Flow: {safe(analysis.get('oi_flow_signal', 'N/A')[:150])}\n\n🎯 MTF: {safe(analysis.get('timeframe_confluence', 'N/A')[:100])}\n\n💡 Reasoning: {safe(analysis.get('reasoning', 'N/A')[:200])}\n\n⚠️ Risk Factors:\n"
            
            risk_factors = analysis.get('risk_factors', ['See analysis'])[:3]
            for r in risk_factors:
                detailed += f"• {safe(r)}\n"
            
            detailed += f"\n━━━━━━━━━━━━━━━━━━━━━━\n\n📊 Total Score: {sum(analysis.get('scoring_breakup', {}).values())}/60\n🎲 Win Probability: {safe(analysis.get('probability', '75-85%'))}\n📈 Risk-Reward: {safe(analysis.get('risk_reward', '1:2'))}\n\n🤖 DeepSeek V3 | Aggressive Mode\n⚡ Only 2-4 Premium Signals/Day\n🎯 Target Win Rate: 75-85%"
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed,
                parse_mode='HTML'
            )
            
            logger.info("Alert sent!")
            return True
            
        except Exception as e:
            logger.error(f"Alert error: {e}")
            return False
    
    async def send_startup_message(self):
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            
            msg = f"🔥 NIFTY 50 BOT v8.0 - ACTIVE 🔥\n⚡ AGGRESSIVE HIGH PROBABILITY MODE\n\n━━━━━━━━━━━━━━━━━━━━━━\n\n🎯 10 AGGRESSIVE FILTERS ENABLED:\n\n1️⃣ Confidence: ≥80% (was 70%)\n2️⃣ OI Divergence: ≥5% (clear winner)\n3️⃣ Volume Surge: ≥50% (strong)\n4️⃣ PCR Extremes: &gt;1.2 or &lt;0.8 ONLY\n5️⃣ MTF Alignment: MANDATORY\n6️⃣ Strong Patterns: ≥2 required\n7️⃣ Liquidity: ≥50k OI\n8️⃣ Volatility: ≥0.5% ATR\n9️⃣ Time Filter: Skip opening/closing\n🔟 Score: Chart 22/30, Options 25/30\n\n━━━━━━━━━━━━━━━━━━━━━━\n\n📊 Stocks: {len(self.dhan.security_id_map)}/50 NIFTY 50\n⏰ Scan: 15 minutes\n🔴 Redis: {redis_status}\n🤖 AI: DeepSeek V3\n\n━━━━━━━━━━━━━━━━━━━━━━\n\n🎯 Expected: 2-4 Premium Signals/Day\n📈 Target Win Rate: 75-85%\n⚡ Fewer Signals, Higher Quality\n\nStatus: 🟢 RUNNING (AGGRESSIVE MODE)"
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("Startup message sent!")
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def run(self):
        logger.info("="*70)
        logger.info("NIFTY 50 BOT v8.0 - AGGRESSIVE HIGH PROBABILITY MODE")
        logger.info("="*70)
        
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"Missing: {', '.join(missing)}")
            return
        
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("Failed to load securities")
            return
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("Bot RUNNING with AGGRESSIVE FILTERS!")
        logger.info("Only 80%+ confidence trades")
        logger.info("Expected: 2-4 premium signals/day")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"AGGRESSIVE SCAN CYCLE - {datetime.now(ist).strftime('%H:%M:%S')}")
                
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"[{idx}/{len(self.dhan.security_id_map)}] {symbol}")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)
                
                rejection_rate = (self.filter_rejections/self.total_scans*100) if self.total_scans > 0 else 0
                
                logger.info(f"CYCLE COMPLETE!")
                logger.info(f"Stats: Scans={self.total_scans}, Rejected={self.filter_rejections}, Alerts={self.alerts_sent}")
                logger.info(f"Filter Rate: {rejection_rate:.1f}% rejected")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


async def main():
    try:
        bot = AdvancedFOBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("NIFTY 50 BOT v8.0 - AGGRESSIVE MODE STARTING...")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"Critical: {e}")
        logger.error(traceback.format_exc())
