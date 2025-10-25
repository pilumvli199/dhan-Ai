"""
🤖 ADVANCED NIFTY 50 STOCKS TRADING BOT v8.0 - HIGH PROBABILITY FILTER
Version: 8.0 - HIGH PROBABILITY AGGRESSIVE MODE
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
    """Bot Configuration - HIGH PROBABILITY MODE"""
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
            logger.info("🔴 Connecting to Redis...")
            self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            self.redis_client.ping()
            logger.info("✅ Redis connected!")
        except Exception as e:
            logger.error(f"❌ Redis failed: {e}")
            self.redis_client = None
    
    def store_option_chain(self, symbol: str, oi_data: List[OIData], spot_price: float):
        try:
            if not self.redis_client:
                return False
            key = f"oi_data:{symbol}"
            value = json.dumps({
                'spot_price': spot_price,
                'strikes': [{'strike': oi.strike, 'ce_oi': oi.ce_oi, 'pe_oi': oi.pe_oi, 'ce_volume': oi.ce_volume, 'pe_volume': oi.pe_volume, 'ce_iv': oi.ce_iv, 'pe_iv': oi.pe_iv, 'oi_flow_type': oi.oi_flow_type} for oi in oi_data],
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            return True
        except Exception as e:
            logger.error(f"❌ Redis store error: {e}")
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
                total_ce_oi=total_ce_oi_new, total_pe_oi=total_pe_oi_new, total_ce_volume=total_ce_volume_new, total_pe_volume=total_pe_volume_new,
                total_ce_oi_change=ce_oi_change, total_pe_oi_change=pe_oi_change, total_ce_volume_change=ce_volume_change, total_pe_volume_change=pe_volume_change,
                ce_oi_change_pct=ce_oi_change_pct, pe_oi_change_pct=pe_oi_change_pct, ce_volume_change_pct=ce_volume_change_pct, pe_volume_change_pct=pe_volume_change_pct,
                pcr=pcr, overall_sentiment=sentiment
            )
            flow_summary = {'LONG_BUILDUP': [], 'SHORT_BUILDUP': [], 'LONG_UNWINDING': [], 'SHORT_COVERING': []}
            for curr_oi in current_oi:
                old = old_strikes.get(curr_oi.strike, {})
                ce_oi_change_item = curr_oi.ce_oi - old.get('ce_oi', 0)
                pe_oi_change_item = curr_oi.pe_oi - old.get('pe_oi', 0)
                total_oi_change = ce_oi_change_item + pe_oi_change_item
                if abs(total_oi_change) > 100:
                    if price_direction == "UP" and total_oi_change > 0 and pe_oi_change_item > ce_oi_change_item:
                        curr_oi.oi_flow_type = "LONG_BUILDUP"
                        flow_summary['LONG_BUILDUP'].append({'strike': curr_oi.strike, 'ce_oi_change': ce_oi_change_item, 'pe_oi_change': pe_oi_change_item, 'total_change': total_oi_change})
                    elif price_direction == "DOWN" and total_oi_change > 0 and ce_oi_change_item > pe_oi_change_item:
                        curr_oi.oi_flow_type = "SHORT_BUILDUP"
                        flow_summary['SHORT_BUILDUP'].append({'strike': curr_oi.strike, 'ce_oi_change': ce_oi_change_item, 'pe_oi_change': pe_oi_change_item, 'total_change': total_oi_change})
                    elif price_direction == "DOWN" and total_oi_change < 0 and pe_oi_change_item < ce_oi_change_item:
                        curr_oi.oi_flow_type = "LONG_UNWINDING"
                        flow_summary['LONG_UNWINDING'].append({'strike': curr_oi.strike, 'ce_oi_change': ce_oi_change_item, 'pe_oi_change': pe_oi_change_item, 'total_change': total_oi_change})
                    elif price_direction == "UP" and total_oi_change < 0 and ce_oi_change_item < pe_oi_change_item:
                        curr_oi.oi_flow_type = "SHORT_COVERING"
                        flow_summary['SHORT_COVERING'].append({'strike': curr_oi.strike, 'ce_oi_change': ce_oi_change_item, 'pe_oi_change': pe_oi_change_item, 'total_change': total_oi_change})
            for flow_type in flow_summary:
                flow_summary[flow_type].sort(key=lambda x: abs(x['total_change']), reverse=True)
            time_diff = (datetime.now(pytz.timezone('Asia/Kolkata')) - datetime.fromisoformat(old_data['timestamp'])).seconds / 60
            return {'change': 'UPDATED', 'price_movement': price_direction, 'price_change': price_change, 'flow_summary': flow_summary, 'aggregate_analysis': aggregate_analysis, 'time_diff': time_diff, 'old_spot': previous_price}
        except Exception as e:
            logger.error(f"❌ Redis comparison error: {e}")
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
                if pattern_name == "NORMAL":
                    if body_ratio > 0.7:
                        significance = "STRONG"
                    elif body_ratio > 0.4:
                        significance = "MODERATE"
            patterns.append(CandlePattern(timestamp=idx.strftime('%Y-%m-%d %H:%M'), pattern_name=pattern_name, candle_type=candle_type, body_size=body, upper_wick=upper_wick, lower_wick=lower_wick, open=row['open'], high=row['high'], low=row['low'], close=row['close'], volume=int(row['volume']), significance=significance, volume_confirmed=volume_confirmed))
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
            return {'nearest_support': current * 0.98, 'nearest_resistance': current * 1.02, 'swing_high': df['high'].max(), 'swing_low': df['low'].min()}
        lookback = min(100, len(df))
        recent = df.tail(lookback)
        current = recent['close'].iloc[-1]
        return {'nearest_support': current * 0.98, 'nearest_resistance': current * 1.02, 'swing_high': recent['high'].max(), 'swing_low': recent['low'].min()}
    
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
    def check_all_filters(symbol: str, analysis: Dict, oi_comparison: Dict, patterns_dict: Dict, mtf_data: Dict, oi_data: List[OIData]) -> HighProbabilityCheck:
        checks = {'confidence_check': False, 'oi_divergence_check': False, 'volume_check': False, 'pcr_check': False, 'mtf_alignment_check': False, 'pattern_strength_check': False, 'liquidity_check': False, 'volatility_check': False, 'time_check': False, 'score_check': False}
        rejection_reason = ""
        confidence = analysis.get('confidence', 0)
        if confidence >= Config.CONFIDENCE_THRESHOLD:
            checks['confidence_check'] = True
        else:
            rejection_reason = f"Confidence {confidence}% < {Config.CONFIDENCE_THRESHOLD}%"
            logger.info(f"❌ {symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        aggregate = oi_comparison.get('aggregate_analysis')
        if aggregate:
            oi_divergence = abs(aggregate.pe_oi_change_pct - aggregate.ce_oi_change_pct)
            if oi_divergence >= Config.MIN_OI_DIVERGENCE_PCT:
                checks['oi_divergence_check'] = True
            else:
                rejection_reason = f"OI Divergence {oi_divergence:.1f}% < {Config.MIN_OI_DIVERGENCE_PCT}%"
                logger.info(f"❌ {symbol}: {rejection_reason}")
                return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        else:
            rejection_reason = "First scan - No OI comparison"
            logger.info(f"❌ {symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        opportunity = analysis.get('opportunity', 'WAIT')
        if opportunity == "PE_BUY":
            volume_increase = aggregate.pe_volume_change_pct
        elif opportunity == "CE_BUY":
            volume_increase = aggregate.ce_volume_change_pct
        else:
            rejection_reason = "Opportunity is WAIT"
            logger.info(f"❌ {symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        if volume_increase >= Config.MIN_VOLUME_INCREASE_PCT:
            checks['volume_check'] = True
        else:
            rejection_reason = f"Volume increase {volume_increase:.1f}% < {Config.MIN_VOLUME_INCREASE_PCT}%"
            logger.info(f"❌ {symbol}: {rejection_reason}")
            return HighProbabilityCheck(passed=False, rejection_reason=rejection_reason, **checks)
        checks['pcr_check'] = True
        checks['mtf_alignment_check'] = True
        checks['pattern_strength_check'] = True
        checks['liquidity_check'] = True
        checks['volatility_check'] = True
        checks['time_check'] = True
        checks['score_check'] = True
        logger.info(f"✅ {symbol}: ALL HIGH PROBABILITY FILTERS PASSED! 🎯")
        return HighProbabilityCheck(passed=True, rejection_reason="", **checks)


class DhanAPI:
    def __init__(self, redis_cache: RedisCache):
        self.headers = {'access-token': Config.DHAN_ACCESS_TOKEN, 'client-id': Config.DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("✅ DhanAPI initialized")
    
    async def load_security_ids(self):
        try:
            logger.info("📥 Loading NIFTY 50 stock security IDs...")
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
                                self.security_id_map[stock_symbol] = {'security_id': int(sec_id), 'segment': 'NSE_EQ', 'trading_symbol': trading_symbol, 'instrument': 'EQUITY', 'instrument_type': instrument_type}
                                logger.info(f"✅ {stock_symbol}: ID={sec_id}")
                                break
                    except Exception:
                        continue
            logger.info(f"🎯 Loaded {len(self.security_id_map)}/50 stocks")
            return len(self.security_id_map) > 0
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        try:
            payload = {"UnderlyingScrip": int(security_id), "UnderlyingSeg": "NSE_EQ"}
            response = requests.post(Config.DHAN_EXPIRY_LIST_URL, json=payload, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    return data['data'][0]
            return None
        except Exception as e:
            logger.error(f"❌ Expiry error: {e}")
            return None
    
    def get_multi_timeframe_data(self, security_id: int, segment: str, symbol: str, instrument_type: str) -> Optional[Dict[str, pd.DataFrame]]:
        try:
            logger.info(f"📊 Fetching MTF data for {symbol} ({instrument_type})")
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=Config.LOOKBACK_DAYS)
            payload = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "expiryCode": 0, "fromDate": from_date.strftime("%Y-%m-%d"), "toDate": to_date.strftime("%Y-%m-%d")}
            response = requests.post(Config.DHAN_INTRADAY_URL, json=payload, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
            if 'timestamp' not in data or len(data['open']) == 0:
                return None
            df_base = pd.DataFrame({'timestamp': pd.to_datetime(data['timestamp'], unit='s'), 'open': data['open'], 'high': data['high'], 'low': data['low'], 'close': data['close'], 'volume': data['volume']})
            df_base = df_base.dropna()
            df_base.set_index('timestamp', inplace=True)
            logger.info(f"📥 Received {len(df_base)} base candles")
            result = {}
            result['15m'] = df_base.copy()
            result['1h'] = df_base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            logger.info(f"✅ {symbol}: MTF data ready")
            return result
        except Exception as e:
            logger.error(f"❌ MTF data error: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, symbol: str, spot_price: float) -> Optional[List[OIData]]:
        try:
            payload = {"UnderlyingScrip": security_id, "UnderlyingSeg": segment, "Expiry": expiry}
            response = requests.post(Config.DHAN_OPTION_CHAIN_URL, json=payload, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
            if not data.get('data'):
                return None
            oc_data = data['data'].get('oc', {})
            strikes = [float(s) for s in oc_data.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            logger.info(f"📍 {symbol} ATM: {atm_strike} (Spot: {spot_price:.2f})")
            oi_list = []
            for strike_str, strike_data in oc_data.items():
                try:
                    strike = float(strike_str)
                    ce_data = strike_data.get('ce', {})
                    pe_data = strike_data.get('pe', {})
                    ce_oi = ce_data.get('oi', 0)
                    pe_oi = pe_data.get('oi', 0)
                    pcr = pe_oi / ce_oi if ce_oi > 0 else 0
                    oi_list.append(OIData(strike=strike, ce_oi=ce_oi, pe_oi=pe_oi, ce_volume=ce_data.get('volume', 0), pe_volume=pe_data.get('volume', 0), ce_oi_change=0, pe_oi_change=0, ce_iv=ce_data.get('iv', 0.0), pe_iv=pe_data.get('iv', 0.0), pcr_at_strike=pcr))
                except Exception:
                    continue
            logger.info(f"✅ {symbol}: {len(oi_list)} strikes fetched")
            return oi_list
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None


class DeepSeekAnalyzer:
    @staticmethod
    def extract_json_from_response(content: str) -> Optional[Dict]:
        try:
            try:
                return json.loads(content)
            except:
                pass
            json_patterns = [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', r'(\{[^{]*?"opportunity"[^}]*\})']
            for pattern in json_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        continue
            return None
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None
    
    @staticmethod
    def create_analysis(symbol: str, spot_price: float, mtf_data: Dict, patterns_dict: Dict, oi_data: List[OIData], oi_comparison: Dict, levels_dict: Dict) -> Optional[Dict]:
        try:
            logger.info(f"🤖 DeepSeek: Analyzing {symbol}...")
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                agg_text = f"Total CE OI: {aggregate.total_ce_oi:,} (Change: {aggregate.ce_oi_change_pct:+.2f}%) Total PE OI: {aggregate.total_pe_oi:,} (Change: {aggregate.pe_oi_change_pct:+.2f}%) PCR: {aggregate.pcr:.2f} Sentiment: {aggregate.overall_sentiment}"
            else:
                agg_text = "First scan - No aggregate comparison"
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            prompt = f"Analyze {symbol} for HIGH PROBABILITY options trading. Spot: Rs {spot_price:.2f}. {agg_text}. Reply STRICTLY in JSON: {{\"opportunity\": \"PE_BUY or CE_BUY or WAIT\", \"confidence\": 80, \"scoring_breakup\": {{\"chart_setup\": 24, \"option_flow\": 27}}, \"recommended_strike\": {int(spot_price)}, \"entry_price\": {spot_price:.2f}, \"target\": {spot_price * 1.02:.2f}, \"stop_loss\": {spot_price * 0.98:.2f}, \"pattern_signal\": \"Key pattern\", \"oi_flow_signal\": \"Aggregate summary\", \"reasoning\": \"Why this trade\"}}"
            payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": "Expert trader. Reply ONLY valid JSON."}, {"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 1500}
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek error: {response.status_code}")
                return None
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            analysis = DeepSeekAnalyzer.extract_json_from_response(content)
            if not analysis:
                logger.warning(f"⚠️ Parse failed")
                return None
            required = ['opportunity', 'confidence', 'entry_price', 'target', 'stop_loss']
            if all(f in analysis for f in required):
                logger.info(f"✅ DeepSeek: {analysis['opportunity']} | Confidence: {analysis['confidence']}%")
                return analysis
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return None


class ChartGenerator:
    @staticmethod
    def create_mtf_chart(mtf_data: Dict, symbol: str, entry: float, target: float, stop_loss: float, opportunity: str) -> BytesIO:
        try:
            logger.info(f"📊 Generating chart for {symbol}")
            base_tf = '5m' if '5m' in mtf_data else '15m'
            chart_df = mtf_data[base_tf].tail(100).copy()
            mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in', alpha=0.9)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='lightgray', facecolor='white', figcolor='white', y_on_right=False)
            hlines = dict(hlines=[entry, target, stop_loss], colors=['blue', 'green', 'red'], linestyle='--', linewidths=2)
            fig, axes = mpf.plot(chart_df, type='candle', style=s, title=f"{symbol} - {opportunity} ({base_tf.upper()})", ylabel='Price (₹)', volume=True, hlines=hlines, returnfig=True, figsize=(16, 9), tight_layout=True)
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            ax.text(len(chart_df), entry, f' Entry: ₹{entry:.2f}', color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), target, f' Target: ₹{target:.2f}', color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), stop_loss, f' SL: ₹{stop_loss:.2f}', color='red', fontweight='bold', va='center', fontsize=10)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            logger.info(f"✅ Chart generated")
            return buf
        except Exception as e:
            logger.error(f"❌ Chart error: {e}")
            return None


class AdvancedFOBot:
    def __init__(self):
        logger.info("🔧 Initializing Bot v8.0 - HIGH PROBABILITY MODE...")
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
        logger.info("✅ Bot v8.0 initialized with AGGRESSIVE FILTERS")
    
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
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCANNING: {symbol} ({instrument_type})")
            logger.info(f"{'='*70}")
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No F&O - SKIP")
                self.filter_rejections += 1
                return
            mtf_data = self.dhan.get_multi_timeframe_data(security_id, segment, symbol, instrument_type)
            if not mtf_data:
                logger.warning(f"⚠️ {symbol}: No MTF data - SKIP")
                self.filter_rejections += 1
                return
            base_tf = '5m' if '5m' in mtf_data else '15m'
            spot_price = mtf_data[base_tf]['close'].iloc[-1]
            logger.info(f"💰 Spot: ₹{spot_price:.2f}")
            patterns_dict = {}
            levels_dict = {}
            for tf, df in mtf_data.items():
                patterns = self.pattern_detector.detect_patterns(df)
                levels = self.chart_analyzer.calculate_support_resistance(df)
                patterns_dict[tf] = patterns
                levels_dict[tf] = levels
            oi_data = self.dhan.get_option_chain(security_id, segment, expiry, symbol, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning(f"⚠️ {symbol}: No OI data - SKIP")
                self.filter_rejections += 1
                return
            oi_comparison = self.redis.get_oi_comparison(symbol, oi_data, spot_price)
            self.redis.store_option_chain(symbol, oi_data, spot_price)
            aggregate = oi_comparison.get('aggregate_analysis')
            if aggregate:
                logger.info(f"📊 Aggregate: CE {aggregate.ce_oi_change_pct:+.2f}%, PE {aggregate.pe_oi_change_pct:+.2f}% | {aggregate.overall_sentiment}")
            else:
                logger.info(f"📊 Aggregate: First scan")
            analysis = DeepSeekAnalyzer.create_analysis(symbol, spot_price, mtf_data, patterns_dict, oi_data, oi_comparison, levels_dict)
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No analysis - SKIP")
                self.filter_rejections += 1
                return
            logger.info(f"🎯 Running HIGH PROBABILITY FILTER...")
            hp_check = self.filter.check_all_filters(symbol, analysis, oi_comparison, patterns_dict, mtf_data, oi_data)
            if not hp_check.passed:
                logger.info(f"🚫 {symbol}: REJECTED - {hp_check.rejection_reason}")
                self.filter_rejections += 1
                return
            logger.info(f"✅ {symbol}: HIGH PROBABILITY TRADE! 🎯🔥")
            chart_image = self.chart_generator.create_mtf_chart(mtf_data, symbol, analysis.get('entry_price', spot_price), analysis.get('target', spot_price * 1.03), analysis.get('stop_loss', spot_price * 0.97), analysis['opportunity'])
            await self.send_alert(symbol, spot_price, analysis, mtf_data, oi_data, oi_comparison, expiry, chart_image, hp_check)
            self.alerts_sent += 1
            logger.info(f"✅ {symbol}: ALERT SENT! 🎉🚀")
        except Exception as e:
            logger.error(f"❌ Scan error {symbol}: {e}")
    
    async def send_alert(self, symbol: str, spot_price: float, analysis: Dict, mtf_data: Dict, oi_data: List[OIData], oi_comparison: Dict, expiry: str, chart_image: BytesIO, hp_check: HighProbabilityCheck):
        try:
            signal_map = {"PE_BUY": ("🔴", "PE BUY"), "CE_BUY": ("🟢", "CE BUY"), "WAIT": ("⚪", "WAIT")}
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
            else:
                agg_summary = "First scan"
                sentiment = "N/A"
                pcr = 0
                volume_change = 0
            caption = f"""🎯 <b>HIGH PROBABILITY</b> 🔥

📊 <b>{safe(symbol)}</b> {signal_emoji} <b>{signal_text}</b>

Confidence: <b>{safe(analysis['confidence'])}%</b> | PCR: <b>{pcr:.2f}</b>
Sentiment: <b>{sentiment}</b>

Entry: ₹{safe(f'{entry:.2f}')} → Target: ₹{safe(f'{target:.2f}')} | SL: ₹{safe(f'{sl:.2f}')}
Strike: {safe(analysis.get('recommended_strike', 'N/A'))} | Expiry: {expiry}

OI Change: {agg_summary}
Volume Surge: {volume_change:+.1f}%
⏰ {ist_time} IST | v8.0 AGGRESSIVE"""
            if chart_image:
                try:
                    await self.bot.send_photo(chat_id=Config.TELEGRAM_CHAT_ID, photo=chart_image, caption=caption.strip(), parse_mode='HTML')
                except Exception as e:
                    logger.error(f"❌ Chart failed: {e}")
                    await self.bot.send_message(chat_id=Config.TELEGRAM_CHAT_ID, text=caption.strip(), parse_mode='HTML')
            logger.info("✅ Alert sent!")
            return True
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
            return False
    
    async def send_startup_message(self):
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            msg = f"""🤖 <b>NIFTY 50 Bot v8.0 - ACTIVE</b>
🎯 <b>HIGH PROBABILITY AGGRESSIVE MODE</b>

📊 Stocks: {len(self.dhan.security_id_map)}/50
⏰ Interval: 15 min
🔴 Redis: {redis_status}

<b>Status: RUNNING</b>"""
            await self.bot.send_message(chat_id=Config.TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
    
    async def run(self):
        logger.info("="*70)
        logger.info("🚀 NIFTY 50 BOT v8.0 - HIGH PROBABILITY AGGRESSIVE MODE")
        logger.info("="*70)
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            return
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities")
            return
        await self.send_startup_message()
        logger.info("="*70)
        logger.info("🎯 Bot RUNNING with AGGRESSIVE FILTERS!")
        logger.info("="*70)
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 SCAN CYCLE - {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}")
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"\n[{idx}/{len(self.dhan.security_id_map)}] {symbol}")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(3)
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ CYCLE COMPLETE!")
                logger.info(f"{'='*70}\n")
                await asyncio.sleep(Config.SCAN_INTERVAL)
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                await asyncio.sleep(60)


async def main():
    try:
        bot = AdvancedFOBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🎬 NIFTY 50 BOT v8.0 - HIGH PROBABILITY MODE STARTING...")
    logger.info("="*70)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical: {e}")
