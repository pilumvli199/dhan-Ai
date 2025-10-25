"""
🤖 ADVANCED NIFTY 50 STOCKS TRADING BOT v7.1 - FIXES APPLIED
✅ Multi-Timeframe Analysis (5m/15m/1h)
✅ Horizontal Large Charts (16x9)
✅ Advanced Candlestick Pattern Detection (15+ patterns)
✅ OI Flow Matrix Analysis (Long/Short Buildup/Unwinding)
✅ Enhanced DeepSeek V3 Prompt
🔧 FIXES: Better JSON parsing, Data validation, 3-day minimum data

Author: Advanced Trading System
Version: 7.1 - CRITICAL FIXES
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

# Telegram
from telegram import Bot

# Charting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ========================
# CONFIGURATION
# ========================
class Config:
    """Bot Configuration"""
    
    # API Credentials
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Dhan API URLs
    DHAN_API_BASE = "https://api.dhan.co"
    DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
    DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
    DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
    DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    # Bot Settings
    SCAN_INTERVAL = 900  # 15 minutes
    CONFIDENCE_THRESHOLD = 70  # Lowered from 75 for better signals
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 3600
    
    # Enhanced Analysis Settings
    LOOKBACK_DAYS = 10  # Increased to ensure 3+ days of trading data
    ATM_STRIKE_RANGE = 11
    MIN_CANDLES_REQUIRED = 50  # Minimum candles for analysis
    
    # NIFTY 50 Stocks
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


# ========================
# DATA MODELS
# ========================
@dataclass
class OIData:
    """Enhanced Option Chain Data Model"""
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
    """Candlestick Pattern Data"""
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


# ========================
# REDIS HANDLER
# ========================
class RedisCache:
    """Redis Cache Manager with OI Flow Tracking"""
    
    def __init__(self):
        try:
            logger.info("🔴 Connecting to Redis...")
            self.redis_client = redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def store_option_chain(self, symbol: str, oi_data: List[OIData], spot_price: float):
        """Store option chain with OI Flow data"""
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
            logger.error(f"❌ Redis store error: {e}")
            return False
    
    def get_oi_comparison(self, symbol: str, current_oi: List[OIData], 
                         current_price: float) -> Dict:
        """Enhanced OI comparison with Flow Matrix classification"""
        try:
            if not self.redis_client:
                return {'change': 'NO_CACHE', 'flow_summary': {}}
            
            key = f"oi_data:{symbol}"
            cached = self.redis_client.get(key)
            
            if not cached:
                return {'change': 'FIRST_SCAN', 'flow_summary': {}}
            
            old_data = json.loads(cached)
            old_strikes = {s['strike']: s for s in old_data['strikes']}
            previous_price = old_data.get('spot_price', current_price)
            
            price_change = current_price - previous_price
            price_direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
            
            flow_summary = {
                'LONG_BUILDUP': [],
                'SHORT_BUILDUP': [],
                'LONG_UNWINDING': [],
                'SHORT_COVERING': []
            }
            
            for curr_oi in current_oi:
                old = old_strikes.get(curr_oi.strike, {})
                
                ce_oi_change = curr_oi.ce_oi - old.get('ce_oi', 0)
                pe_oi_change = curr_oi.pe_oi - old.get('pe_oi', 0)
                
                total_oi_change = ce_oi_change + pe_oi_change
                
                if abs(total_oi_change) > 500:
                    
                    if price_direction == "UP" and total_oi_change > 0:
                        if pe_oi_change > ce_oi_change:
                            curr_oi.oi_flow_type = "LONG_BUILDUP"
                            flow_summary['LONG_BUILDUP'].append({
                                'strike': curr_oi.strike,
                                'ce_oi_change': ce_oi_change,
                                'pe_oi_change': pe_oi_change,
                                'total_change': total_oi_change
                            })
                    
                    elif price_direction == "DOWN" and total_oi_change > 0:
                        if ce_oi_change > pe_oi_change:
                            curr_oi.oi_flow_type = "SHORT_BUILDUP"
                            flow_summary['SHORT_BUILDUP'].append({
                                'strike': curr_oi.strike,
                                'ce_oi_change': ce_oi_change,
                                'pe_oi_change': pe_oi_change,
                                'total_change': total_oi_change
                            })
                    
                    elif price_direction == "DOWN" and total_oi_change < 0:
                        if pe_oi_change < ce_oi_change:
                            curr_oi.oi_flow_type = "LONG_UNWINDING"
                            flow_summary['LONG_UNWINDING'].append({
                                'strike': curr_oi.strike,
                                'ce_oi_change': ce_oi_change,
                                'pe_oi_change': pe_oi_change,
                                'total_change': total_oi_change
                            })
                    
                    elif price_direction == "UP" and total_oi_change < 0:
                        if ce_oi_change < pe_oi_change:
                            curr_oi.oi_flow_type = "SHORT_COVERING"
                            flow_summary['SHORT_COVERING'].append({
                                'strike': curr_oi.strike,
                                'ce_oi_change': ce_oi_change,
                                'pe_oi_change': pe_oi_change,
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
                'time_diff': time_diff,
                'old_spot': previous_price
            }
            
        except Exception as e:
            logger.error(f"❌ Redis comparison error: {e}")
            return {'change': 'ERROR', 'flow_summary': {}}


# ========================
# ADVANCED PATTERN DETECTOR
# ========================
class AdvancedPatternDetector:
    """Advanced Candlestick Pattern Detection"""
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame, lookback: int = 50) -> List[CandlePattern]:
        """Detect 15+ candlestick patterns"""
        patterns = []
        
        # Use all available data if less than lookback
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
                
                # DOJI
                if body_ratio < 0.1:
                    pattern_name = "DOJI"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                
                # HAMMER
                elif lower_wick > body * 2 and upper_wick < body * 0.5 and body_ratio < 0.3:
                    pattern_name = "HAMMER"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                
                # INVERTED HAMMER
                elif upper_wick > body * 2 and lower_wick < body * 0.5 and body_ratio < 0.3:
                    pattern_name = "INVERTED_HAMMER"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                
                # SHOOTING STAR
                elif upper_wick > body * 2 and lower_wick < body * 0.5 and not is_bullish:
                    pattern_name = "SHOOTING_STAR"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                
                # MARUBOZU
                elif upper_wick < body * 0.1 and lower_wick < body * 0.1 and body_ratio > 0.8:
                    pattern_name = "MARUBOZU_BULLISH" if is_bullish else "MARUBOZU_BEARISH"
                    significance = "STRONG" if volume_confirmed else "MODERATE"
                
                # SPINNING TOP
                elif body_ratio < 0.3 and upper_wick > body and lower_wick > body:
                    pattern_name = "SPINNING_TOP"
                    significance = "MODERATE"
                
                # MULTI-CANDLE PATTERNS
                elif i > 0:
                    prev_row = recent_df.iloc[i-1]
                    prev_body = abs(prev_row['close'] - prev_row['open'])
                    prev_is_bullish = prev_row['close'] > prev_row['open']
                    
                    # BULLISH ENGULFING
                    if (is_bullish and not prev_is_bullish and 
                        row['open'] < prev_row['close'] and 
                        row['close'] > prev_row['open'] and
                        body > prev_body * 0.7):
                        pattern_name = "BULLISH_ENGULFING"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    
                    # BEARISH ENGULFING
                    elif (not is_bullish and prev_is_bullish and 
                          row['open'] > prev_row['close'] and 
                          row['close'] < prev_row['open'] and
                          body > prev_body * 0.7):
                        pattern_name = "BEARISH_ENGULFING"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    
                    # PIERCING LINE
                    elif (is_bullish and not prev_is_bullish and
                          row['open'] < prev_row['low'] and
                          row['close'] > (prev_row['open'] + prev_row['close']) / 2):
                        pattern_name = "PIERCING_LINE"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    
                    # DARK CLOUD COVER
                    elif (not is_bullish and prev_is_bullish and
                          row['open'] > prev_row['high'] and
                          row['close'] < (prev_row['open'] + prev_row['close']) / 2):
                        pattern_name = "DARK_CLOUD_COVER"
                        significance = "STRONG" if volume_confirmed else "MODERATE"
                    
                    # HARAMI
                    elif (body < prev_body * 0.5 and
                          row['high'] < prev_row['high'] and
                          row['low'] > prev_row['low']):
                        pattern_name = "HARAMI_BULLISH" if is_bullish else "HARAMI_BEARISH"
                        significance = "MODERATE"
                
                # THREE CANDLE PATTERNS
                if i > 1:
                    prev1 = recent_df.iloc[i-1]
                    prev2 = recent_df.iloc[i-2]
                    
                    # THREE WHITE SOLDIERS
                    if (is_bullish and 
                        prev1['close'] > prev1['open'] and 
                        prev2['close'] > prev2['open'] and
                        row['close'] > prev1['close'] > prev2['close']):
                        pattern_name = "THREE_WHITE_SOLDIERS"
                        significance = "STRONG"
                    
                    # THREE BLACK CROWS
                    elif (not is_bullish and 
                          prev1['close'] < prev1['open'] and 
                          prev2['close'] < prev2['open'] and
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


# ========================
# CHART ANALYZER
# ========================
class ChartAnalyzer:
    """Multi-Timeframe Chart Analysis"""
    
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify trend using SMAs"""
        if len(df) < 20:
            return "INSUFFICIENT_DATA"
        
        # Use available data
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
        """Calculate dynamic support/resistance"""
        # Need minimum data for pivot calculation
        if len(df) < 20:
            current = df['close'].iloc[-1]
            return {
                'nearest_support': current * 0.98,
                'nearest_resistance': current * 1.02,
                'swing_high': df['high'].max(),
                'swing_low': df['low'].min()
            }
        
        # Use available data
        lookback = min(100, len(df))
        recent = df.tail(lookback)
        current = recent['close'].iloc[-1]
        
        pivot_window = min(50, len(recent))
        highs = recent['high'].tail(pivot_window)
        lows = recent['low'].tail(pivot_window)
        
        resistance_levels = []
        support_levels = []
        
        # Simplified pivot detection for smaller datasets
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


# ========================
# DHAN API HANDLER
# ========================
class DhanAPI:
    """Dhan HQ API Integration"""
    
    def __init__(self, redis_cache: RedisCache):
        self.headers = {
            'access-token': Config.DHAN_ACCESS_TOKEN,
            'client-id': Config.DHAN_CLIENT_ID,
            'Content-Type': 'application/json'
        }
        self.security_id_map = {}
        self.redis = redis_cache
        logger.info("✅ DhanAPI initialized")
    
    async def load_security_ids(self):
        """Load security IDs"""
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
                                
                                self.security_id_map[stock_symbol] = {
                                    'security_id': int(sec_id),
                                    'segment': 'NSE_EQ',
                                    'trading_symbol': trading_symbol,
                                    'instrument': 'EQUITY',
                                    'instrument_type': instrument_type
                                }
                                logger.info(f"✅ {stock_symbol}: ID={sec_id} Type={instrument_type}")
                                break
                    except Exception:
                        continue
            
            logger.info(f"🎯 Loaded {len(self.security_id_map)}/50 stocks")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry"""
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
            logger.error(f"❌ Expiry error: {e}")
            return None
    
    def get_multi_timeframe_data(self, security_id: int, segment: str, 
                                 symbol: str, instrument_type: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get multi-timeframe data - FIXED with better data handling"""
        try:
            logger.info(f"📊 Fetching MTF data for {symbol} ({instrument_type})")
            
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
                logger.error(f"API returned {response.status_code}")
                return None
            
            data = response.json()
            
            if 'timestamp' not in data or len(data['open']) == 0:
                logger.warning(f"No candle data in response")
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
            
            logger.info(f"📥 Received {len(df_base)} base candles")
            
            # Check if we have minimum required data
            if len(df_base) < Config.MIN_CANDLES_REQUIRED:
                logger.warning(f"⚠️ Only {len(df_base)} candles, need {Config.MIN_CANDLES_REQUIRED}+ for reliable analysis")
                # Continue anyway but with warning
            
            result = {}
            
            if instrument_type == "INDEX":
                result['5m'] = df_base.copy()
                
                # FIXED: Use 'min' instead of 'T'
                result['15m'] = df_base.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # FIXED: Use 'h' instead of 'H'
                result['1h'] = df_base.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
            else:
                # Calculate time difference to determine base timeframe
                if len(df_base) > 1:
                    time_diff = (df_base.index[1] - df_base.index[0]).seconds / 60
                else:
                    time_diff = 15  # Default assumption
                
                if time_diff <= 5:
                    # Resample to 15-min
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
                
                # Resample to 1-hour
                result['1h'] = df_15m.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            logger.info(f"✅ {symbol}: MTF data ready")
            for tf, df in result.items():
                logger.info(f"   {tf}: {len(df)} candles")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ MTF data error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, 
                        symbol: str, spot_price: float) -> Optional[List[OIData]]:
        """Get option chain data"""
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
            
            logger.info(f"📍 {symbol} ATM: {atm_strike} (Spot: {spot_price:.2f})")
            
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
            
            logger.info(f"✅ {symbol}: {len(oi_list)} strikes fetched")
            return oi_list
            
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None


# ========================
# DEEPSEEK ANALYZER
# ========================
class DeepSeekAnalyzer:
    """DeepSeek V3 Enhanced Analysis - FIXED JSON parsing"""
    
    @staticmethod
    def extract_json_from_response(content: str) -> Optional[Dict]:
        """Enhanced JSON extraction with multiple fallback methods"""
        try:
            # Method 1: Direct JSON parse
            try:
                return json.loads(content)
            except:
                pass
            
            # Method 2: Extract from code blocks
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
            
            # Method 3: Find first complete JSON object
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
        """Enhanced DeepSeek analysis with better error handling"""
        try:
            logger.info(f"🤖 DeepSeek: Analyzing {symbol}...")
            
            base_tf = '5m' if '5m' in mtf_data else '15m'
            entry_tf_patterns = patterns_dict.get(base_tf, [])
            
            # Format patterns (last 10)
            pattern_summary = []
            for i, p in enumerate(entry_tf_patterns[-10:], 1):
                vol_flag = "✓" if p.volume_confirmed else ""
                pattern_summary.append(
                    f"{i}. {p.timestamp} | {p.pattern_name} ({p.significance}) {vol_flag}"
                )
            
            patterns_text = "\n".join(pattern_summary) if pattern_summary else "No significant patterns"
            
            # Count strong patterns
            strong_patterns = [p for p in entry_tf_patterns[-20:] if p.significance == "STRONG"]
            pattern_types = {}
            for p in strong_patterns:
                pattern_types[p.pattern_name] = pattern_types.get(p.pattern_name, 0) + 1
            
            # Format OI data (top 10 strikes)
            oi_data_sorted = sorted(oi_data, key=lambda x: x.strike)
            atm_strike = min(oi_data, key=lambda x: abs(x.strike - spot_price)).strike
            
            oi_table = []
            for oi in oi_data_sorted[:10]:
                marker = " ⭐ATM" if oi.strike == atm_strike else ""
                oi_table.append(
                    f"Strike {oi.strike}{marker} | CE OI:{oi.ce_oi:,} PE OI:{oi.pe_oi:,} | PCR:{oi.pcr_at_strike:.2f}"
                )
            
            oi_text = "\n".join(oi_table)
            
            # OI Flow summary
            flow_summary = oi_comparison.get('flow_summary', {})
            flow_parts = []
            for flow_type in ['LONG_BUILDUP', 'SHORT_BUILDUP', 'LONG_UNWINDING', 'SHORT_COVERING']:
                items = flow_summary.get(flow_type, [])
                if items:
                    flow_parts.append(f"{flow_type}: {len(items)} strikes")
            
            flow_text = ", ".join(flow_parts) if flow_parts else "First scan (no previous data)"
            
            # PCR
            total_ce_oi = sum(oi.ce_oi for oi in oi_data)
            total_pe_oi = sum(oi.pe_oi for oi in oi_data)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            # Levels
            levels_1h = levels_dict.get('1h', {})
            levels_entry = levels_dict.get(base_tf, {})
            
            # Trends
            trend_1h = ChartAnalyzer.identify_trend(mtf_data.get('1h', mtf_data[base_tf]))
            trend_entry = ChartAnalyzer.identify_trend(mtf_data[base_tf])
            
            # DeepSeek API Call
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Analyze {symbol} stock for options trading.

CURRENT DATA:
Spot: Rs {spot_price:.2f} | ATM Strike: {atm_strike}
1H Trend: {trend_1h} | {base_tf.upper()} Trend: {trend_entry}
PCR: {pcr:.2f} | Total CE OI: {total_ce_oi:,} | Total PE OI: {total_pe_oi:,}

PATTERNS (Last 10 {base_tf}):
{patterns_text}

Strong Patterns: {', '.join([f"{k}({v})" for k, v in pattern_types.items()]) if pattern_types else "None"}

OPTION CHAIN (Top 10):
{oi_text}

OI FLOW: {flow_text}

SUPPORT/RESISTANCE:
1H: Support={levels_1h.get('nearest_support', 'N/A')} Resistance={levels_1h.get('nearest_resistance', 'N/A')}
{base_tf.upper()}: Support={levels_entry.get('nearest_support', 'N/A')} Resistance={levels_entry.get('nearest_resistance', 'N/A')}

Analyze and reply STRICTLY in this JSON format (no markdown, no extra text):

{{
  "opportunity": "PE_BUY or CE_BUY or WAIT",
  "confidence": 75,
  "scoring_breakup": {{
    "chart_setup": 22,
    "option_flow": 25,
    "risk_management": 16,
    "probability": 12
  }},
  "recommended_strike": {int(atm_strike)},
  "entry_price": {spot_price:.2f},
  "target": {spot_price * 1.02:.2f},
  "stop_loss": {spot_price * 0.98:.2f},
  "risk_reward": "1:2",
  "timeframe_confluence": "Brief trend alignment",
  "pattern_signal": "Key pattern name",
  "oi_flow_signal": "OI flow summary",
  "key_levels": "Support and resistance",
  "reasoning": "Why this trade setup",
  "probability": "70%",
  "risk_factors": ["Risk 1", "Risk 2"]
}}

CRITICAL: Reply with ONLY the JSON object. No explanations before or after."""

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert trader. Reply ONLY with valid JSON. No markdown, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek API error: {response.status_code}")
                logger.error(f"Response: {response.text[:200]}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Enhanced JSON extraction
            analysis = DeepSeekAnalyzer.extract_json_from_response(content)
            
            if not analysis:
                logger.warning(f"⚠️ Could not parse JSON. Raw response (first 300 chars):")
                logger.warning(content[:300])
                return None
            
            # Validate required fields
            required = ['opportunity', 'confidence', 'entry_price', 'target', 'stop_loss']
            if all(f in analysis for f in required):
                logger.info(f"✅ DeepSeek: {analysis['opportunity']} | Confidence: {analysis['confidence']}%")
                return analysis
            else:
                missing = [f for f in required if f not in analysis]
                logger.warning(f"⚠️ Missing fields in response: {missing}")
                return None
            
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# CHART GENERATOR
# ========================
class ChartGenerator:
    """Generate horizontal large charts"""
    
    @staticmethod
    def create_mtf_chart(mtf_data: Dict, symbol: str, entry: float, 
                        target: float, stop_loss: float, opportunity: str) -> BytesIO:
        """Create multi-timeframe chart (16x9 horizontal)"""
        try:
            logger.info(f"📊 Generating chart for {symbol}")
            
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
                title=f"{symbol} - {opportunity} ({base_tf.upper()})",
                ylabel='Price (₹)',
                volume=True,
                hlines=hlines,
                returnfig=True,
                figsize=(16, 9),
                tight_layout=True
            )
            
            ax = axes[0]
            current_price = chart_df['close'].iloc[-1]
            
            ax.text(len(chart_df), entry, f' Entry: ₹{entry:.2f}', 
                   color='blue', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), target, f' Target: ₹{target:.2f}', 
                   color='green', fontweight='bold', va='center', fontsize=10)
            ax.text(len(chart_df), stop_loss, f' SL: ₹{stop_loss:.2f}', 
                   color='red', fontweight='bold', va='center', fontsize=10)
            ax.axhline(y=current_price, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.text(len(chart_df), current_price, f' Current: ₹{current_price:.2f}', 
                   color='orange', fontweight='bold', va='center', fontsize=10)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            logger.info(f"✅ Chart generated (16x9)")
            return buf
            
        except Exception as e:
            logger.error(f"❌ Chart error: {e}")
            return None


# ========================
# MAIN BOT
# ========================
class AdvancedFOBot:
    """Advanced NIFTY 50 Bot v7.1"""
    
    def __init__(self):
        logger.info("🔧 Initializing Bot v7.1...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.pattern_detector = AdvancedPatternDetector()
        self.chart_analyzer = ChartAnalyzer()
        self.chart_generator = ChartGenerator()
        self.running = True
        logger.info("✅ Bot v7.1 initialized")
    
    def is_market_open(self) -> bool:
        """Check market hours"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            return False
        
        return Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE
    
    def escape_html(self, text: str) -> str:
        """Escape HTML"""
        return html.escape(str(text))
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Comprehensive scan with better error handling"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            instrument_type = info['instrument_type']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 SCANNING: {symbol} ({instrument_type})")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No F&O - SKIP")
                return
            
            # Get MTF data
            mtf_data = self.dhan.get_multi_timeframe_data(security_id, segment, symbol, instrument_type)
            if not mtf_data:
                logger.warning(f"⚠️ {symbol}: No MTF data - SKIP")
                return
            
            # Get spot price
            base_tf = '5m' if '5m' in mtf_data else '15m'
            spot_price = mtf_data[base_tf]['close'].iloc[-1]
            logger.info(f"💰 Spot: ₹{spot_price:.2f}")
            
            # Check minimum data requirement
            if len(mtf_data[base_tf]) < 30:
                logger.warning(f"⚠️ {symbol}: Insufficient data ({len(mtf_data[base_tf])} candles) - SKIP")
                return
            
            # Analyze all timeframes
            patterns_dict = {}
            levels_dict = {}
            
            for tf, df in mtf_data.items():
                patterns = self.pattern_detector.detect_patterns(df)
                levels = self.chart_analyzer.calculate_support_resistance(df)
                patterns_dict[tf] = patterns
                levels_dict[tf] = levels
                
                supp = levels.get('nearest_support', 0)
                logger.info(f"📊 {tf}: {len(patterns)} patterns, Supp={supp:.2f}")
            
            # Get option chain
            oi_data = self.dhan.get_option_chain(security_id, segment, expiry, symbol, spot_price)
            if not oi_data or len(oi_data) < 10:
                logger.warning(f"⚠️ {symbol}: No OI data - SKIP")
                return
            
            # OI comparison
            oi_comparison = self.redis.get_oi_comparison(symbol, oi_data, spot_price)
            self.redis.store_option_chain(symbol, oi_data, spot_price)
            
            flow_summary = oi_comparison.get('flow_summary', {})
            logger.info(f"📊 OI Flow: {len(flow_summary.get('LONG_BUILDUP', []))} Long Buildup, "
                       f"{len(flow_summary.get('SHORT_BUILDUP', []))} Short Buildup")
            
            # DeepSeek analysis
            analysis = DeepSeekAnalyzer.create_analysis(
                symbol, spot_price, mtf_data, patterns_dict, 
                oi_data, oi_comparison, levels_dict
            )
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No analysis - SKIP")
                return
            
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Low confidence ({analysis['confidence']}%) - NO ALERT")
                return
            
            # Generate chart
            chart_image = self.chart_generator.create_mtf_chart(
                mtf_data, symbol,
                analysis.get('entry_price', spot_price),
                analysis.get('target', spot_price * 1.03),
                analysis.get('stop_loss', spot_price * 0.97),
                analysis['opportunity']
            )
            
            # Send alert
            await self.send_alert(symbol, spot_price, analysis, mtf_data, 
                                 oi_data, oi_comparison, expiry, chart_image)
            
            logger.info(f"✅ {symbol}: ALERT SENT! 🎉")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Scan error {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_alert(self, symbol: str, spot_price: float, analysis: Dict,
                        mtf_data: Dict, oi_data: List[OIData], 
                        oi_comparison: Dict, expiry: str, chart_image: BytesIO):
        """Send trading alert"""
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
            
            total_ce = sum(oi.ce_oi for oi in oi_data)
            total_pe = sum(oi.pe_oi for oi in oi_data)
            pcr = total_pe / total_ce if total_ce > 0 else 0
            
            # Chart caption
            caption = f"""
📊 <b>{safe(symbol)}</b> {signal_emoji} <b>{signal_text}</b>

Confidence: {safe(analysis['confidence'])}% | PCR: {pcr:.2f}
Entry: ₹{safe(f'{entry:.2f}')} → Target: ₹{safe(f'{target:.2f}')} | SL: ₹{safe(f'{sl:.2f}')}
Strike: {safe(analysis.get('recommended_strike', 'N/A'))} | Expiry: {expiry}

⏰ {ist_time} IST | v7.1
"""
            
            # Send chart
            if chart_image:
                try:
                    await self.bot.send_photo(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        photo=chart_image,
                        caption=caption.strip(),
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"❌ Chart send failed: {e}")
                    await self.bot.send_message(
                        chat_id=Config.TELEGRAM_CHAT_ID,
                        text=caption.strip(),
                        parse_mode='HTML'
                    )
            
            # Detailed message
            flow_summary = oi_comparison.get('flow_summary', {})
            flow_text = []
            for flow_type in ['LONG_BUILDUP', 'SHORT_BUILDUP']:
                items = flow_summary.get(flow_type, [])
                if items:
                    flow_text.append(f"{flow_type}: {len(items)}")
            
            flow_summary_text = ", ".join(flow_text) if flow_text else "First scan"
            
            detailed = f"""
📈 <b>Analysis Details</b>

🕯️ Pattern: {safe(analysis.get('pattern_signal', 'N/A')[:100])}

⛓️ OI: {safe(flow_summary_text)}
{safe(analysis.get('oi_flow_signal', 'N/A')[:120])}

🎯 MTF: {safe(analysis.get('timeframe_confluence', 'N/A')[:100])}

💡 {safe(analysis.get('reasoning', 'N/A')[:200])}

Score: {analysis.get('scoring_breakup', {}).get('chart_setup', 0)}/30 + 
{analysis.get('scoring_breakup', {}).get('option_flow', 0)}/30 + 
{analysis.get('scoring_breakup', {}).get('risk_management', 0)}/20

🤖 DeepSeek V3 | Phase 1
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=detailed.strip(),
                parse_mode='HTML'
            )
            
            logger.info("✅ Alert sent!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
            return False
    
    async def send_startup_message(self):
        """Startup notification"""
        try:
            redis_status = "✅" if self.redis.redis_client else "❌"
            
            msg = f"""
🤖 <b>NIFTY 50 Bot v7.1 - ACTIVE</b>

🔧 FIXES APPLIED:
✅ Better JSON parsing (3 fallback methods)
✅ Data validation (min 30 candles)
✅ Pandas deprecation fixed
✅ Enhanced error handling
✅ Support/resistance calculation improved

📊 Stocks: {len(self.dhan.security_id_map)}/50
⏰ Interval: 15 min
🎯 Confidence: {Config.CONFIDENCE_THRESHOLD}%+
🔴 Redis: {redis_status}

🚀 Status: <b>RUNNING</b>
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("="*70)
        logger.info("🚀 NIFTY 50 BOT v7.1 - FIXES APPLIED")
        logger.info("="*70)
        
        # Validate
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            return
        
        # Load securities
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities")
            return
        
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("🎯 Bot RUNNING! Scanning every 15 min...")
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
                    await asyncio.sleep(3)  # Rate limiting
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ CYCLE COMPLETE! Next in 15 min")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


# ========================
# MAIN
# ========================
async def main():
    """Entry point"""
    try:
        bot = AdvancedFOBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🎬 NIFTY 50 BOT v7.1 - STARTING...")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical: {e}")
        logger.error(traceback.format_exc())
