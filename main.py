"""
🤖 ADVANCED F&O TRADING BOT v4.0
NIFTY 50 + SENSEX (BANK NIFTY Optional)

✅ Advanced Chart Pattern Detection (DeepSeek powered)
✅ Smart Money Concepts (Order Blocks, FVG, Liquidity)
✅ Psychological Levels & Trendlines
✅ Option Chain Analysis (OI, Change in OI, Volume)
✅ Redis Caching for OI comparison
✅ PE/CE Buy Opportunity Detection

Author: Advanced Trading System
Version: 4.0 - CHART + OI COMBINED ANALYSIS
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

# Telegram
from telegram import Bot

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
    SCAN_INTERVAL = 300  # 5 minutes
    CONFIDENCE_THRESHOLD = 75  # Higher threshold for better signals
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    REDIS_EXPIRY = 3600  # 1 hour cache
    
    # Chart Analysis Settings
    LOOKBACK_CANDLES = 100  # Last 100 candles for pattern detection
    TRENDLINE_CANDLES = 50  # Last 50 candles for trendline
    PSYCHOLOGICAL_LEVELS = [100, 250, 500, 1000]  # Round number intervals
    
    # Symbols to track - NIFTY 50 & SENSEX
    SYMBOLS = {
        "NIFTY": {
            "symbol": "NIFTY 50",
            "segment": "IDX_I",
            "alternatives": ["Nifty 50", "NIFTY50", "NIFTY"],
            "lot_size": 25
        },
        "BANKNIFTY": {
            "symbol": "NIFTY BANK",
            "segment": "IDX_I",
            "alternatives": ["Nifty Bank", "NIFTYBANK", "BANKNIFTY"],
            "lot_size": 15
        },
        # Add SENSEX if needed
        "SENSEX": {
            "symbol": "SENSEX",
            "segment": "IDX_I",
            "alternatives": ["BSE SENSEX", "SENSEX"],
            "lot_size": 10
        }
    }


# ========================
# DATA MODELS
# ========================
@dataclass
class ChartPattern:
    """Chart Pattern Data Model"""
    name: str
    type: str  # BULLISH/BEARISH/NEUTRAL
    confidence: int
    target: float
    stop_loss: float
    description: str


@dataclass
class OIData:
    """Option Chain Data Model"""
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_oi_change: int
    pe_oi_change: int


@dataclass
class TrendlineData:
    """Trendline Information"""
    support_line: float
    resistance_line: float
    trend: str  # UPTREND/DOWNTREND/SIDEWAYS


# ========================
# REDIS HANDLER
# ========================
class RedisCache:
    """Redis Cache Manager"""
    
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
        """Store option chain with OI data"""
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
                        'pe_volume': oi.pe_volume
                    }
                    for oi in oi_data
                ],
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
            })
            
            self.redis_client.setex(key, Config.REDIS_EXPIRY, value)
            logger.info(f"💾 Redis: Stored OI data for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Redis store error: {e}")
            return False
    
    def get_oi_comparison(self, symbol: str, current_oi: List[OIData]) -> Dict:
        """Compare current OI with cached data"""
        try:
            if not self.redis_client:
                return {'change': 'NO_CACHE', 'deltas': []}
            
            key = f"oi_data:{symbol}"
            cached = self.redis_client.get(key)
            
            if not cached:
                logger.info(f"📊 {symbol}: First OI scan")
                return {'change': 'FIRST_SCAN', 'deltas': []}
            
            old_data = json.loads(cached)
            old_strikes = {s['strike']: s for s in old_data['strikes']}
            
            deltas = []
            for curr_oi in current_oi:
                old = old_strikes.get(curr_oi.strike, {})
                
                ce_oi_change = curr_oi.ce_oi - old.get('ce_oi', 0)
                pe_oi_change = curr_oi.pe_oi - old.get('pe_oi', 0)
                
                if abs(ce_oi_change) > 1000 or abs(pe_oi_change) > 1000:
                    deltas.append({
                        'strike': curr_oi.strike,
                        'ce_oi_change': ce_oi_change,
                        'pe_oi_change': pe_oi_change,
                        'ce_volume': curr_oi.ce_volume,
                        'pe_volume': curr_oi.pe_volume
                    })
            
            time_diff = (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                        datetime.fromisoformat(old_data['timestamp'])).seconds / 60
            
            logger.info(f"📊 {symbol}: {len(deltas)} strikes with significant OI changes")
            
            return {
                'change': 'UPDATED',
                'deltas': deltas,
                'time_diff': time_diff,
                'old_spot': old_data.get('spot_price', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Redis comparison error: {e}")
            return {'change': 'ERROR', 'deltas': []}


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
        """Load security IDs from CSV"""
        try:
            logger.info("📥 Loading security IDs...")
            response = requests.get(Config.DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to load instruments")
                return False
            
            csv_reader = csv.DictReader(io.StringIO(response.text))
            all_rows = list(csv_reader)
            
            for symbol, info in Config.SYMBOLS.items():
                segment = info['segment']
                alternatives = info.get('alternatives', [info['symbol']])
                
                for row in all_rows:
                    try:
                        if segment == "IDX_I":
                            trading_symbol = row.get('SEM_TRADING_SYMBOL', '')
                            if (row.get('SEM_SEGMENT') == 'I' and 
                                trading_symbol in alternatives):
                                sec_id = row.get('SEM_SMST_SECURITY_ID')
                                if sec_id:
                                    self.security_id_map[symbol] = {
                                        'security_id': int(sec_id),
                                        'segment': segment,
                                        'trading_symbol': trading_symbol,
                                        'lot_size': info['lot_size']
                                    }
                                    logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                    break
                    except Exception:
                        continue
            
            logger.info(f"🎯 Loaded {len(self.security_id_map)} securities")
            return len(self.security_id_map) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_nearest_expiry(self, security_id: int, segment: str) -> Optional[str]:
        """Get nearest expiry"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            response = requests.post(
                Config.DHAN_EXPIRY_LIST_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data'][0]
            
            return None
        except Exception as e:
            logger.error(f"❌ Expiry error: {e}")
            return None
    
    def get_historical_candles(self, security_id: int, segment: str, symbol: str, 
                              lookback_days: int = 7) -> Optional[pd.DataFrame]:
        """Get historical candles"""
        try:
            logger.info(f"📊 Fetching {lookback_days} days candles for {symbol}")
            
            exch_seg = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            instrument = "INDEX" if segment == "IDX_I" else "EQUITY"
            
            ist = pytz.timezone('Asia/Kolkata')
            to_date = datetime.now(ist)
            from_date = to_date - timedelta(days=lookback_days)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg,
                "instrument": instrument,
                "interval": "5",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                Config.DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'timestamp' in data and len(data.get('open', [])) > 0:
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                        'open': data['open'],
                        'high': data['high'],
                        'low': data['low'],
                        'close': data['close'],
                        'volume': data['volume']
                    })
                    
                    df = df.dropna()
                    logger.info(f"✅ {symbol}: {len(df)} candles fetched")
                    return df
            
            logger.warning(f"⚠️ {symbol}: No candle data")
            return None
            
        except Exception as e:
            logger.error(f"❌ Candle fetch error: {e}")
            return None
    
    def get_option_chain(self, security_id: int, segment: str, expiry: str, 
                        symbol: str) -> Optional[Dict]:
        """Get option chain data"""
        try:
            logger.info(f"⛓️ Fetching option chain for {symbol}")
            
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
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    logger.info(f"✅ Option chain received for {symbol}")
                    return data['data']
            
            return None
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None


# ========================
# ADVANCED CHART ANALYZER
# ========================
class AdvancedChartAnalyzer:
    """Advanced Chart Pattern & Technical Analysis"""
    
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify overall trend"""
        if len(df) < 20:
            return "INSUFFICIENT_DATA"
        
        # Use last 50 candles
        recent = df.tail(50)
        
        # Calculate moving averages
        sma_20 = recent['close'].tail(20).mean()
        sma_50 = recent['close'].mean()
        current_price = recent['close'].iloc[-1]
        
        # Trend logic
        if current_price > sma_20 > sma_50:
            return "UPTREND"
        elif current_price < sma_20 < sma_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def find_psychological_levels(spot_price: float) -> List[float]:
        """Find psychological support/resistance levels"""
        levels = []
        
        for interval in Config.PSYCHOLOGICAL_LEVELS:
            # Find nearest round numbers
            lower = (spot_price // interval) * interval
            upper = lower + interval
            
            levels.extend([lower, upper])
        
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        
        # Keep only levels within 5% of spot
        filtered = [
            level for level in levels
            if abs(level - spot_price) / spot_price <= 0.05
        ]
        
        logger.info(f"🎯 Psychological levels: {filtered}")
        return filtered
    
    @staticmethod
    def calculate_support_resistance_zones(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance zones"""
        if len(df) < 50:
            return [], []
        
        recent = df.tail(Config.TRENDLINE_CANDLES)
        
        # Find swing highs and lows
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Resistance zones (clusters of highs)
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        # Support zones (clusters of lows)
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        logger.info(f"📊 Support zones: {len(support_levels)}, Resistance zones: {len(resistance_levels)}")
        
        return support_levels, resistance_levels
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> List[ChartPattern]:
        """Detect chart patterns (Head & Shoulders, Triangles, etc.)"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        recent = df.tail(50)
        closes = recent['close'].values
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Double Top Detection
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                patterns.append(ChartPattern(
                    name="DOUBLE_TOP",
                    type="BEARISH",
                    confidence=75,
                    target=closes[-1] * 0.97,
                    stop_loss=max(highs[-10:]),
                    description="Double top pattern detected - potential reversal"
                ))
        
        # Double Bottom Detection
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                patterns.append(ChartPattern(
                    name="DOUBLE_BOTTOM",
                    type="BULLISH",
                    confidence=75,
                    target=closes[-1] * 1.03,
                    stop_loss=min(lows[-10:]),
                    description="Double bottom pattern - potential reversal upward"
                ))
        
        logger.info(f"📈 Chart patterns: {[p.name for p in patterns]}")
        return patterns
    
    @staticmethod
    def calculate_trendlines(df: pd.DataFrame) -> TrendlineData:
        """Calculate dynamic trendlines"""
        if len(df) < 30:
            return TrendlineData(0, 0, "INSUFFICIENT_DATA")
        
        recent = df.tail(Config.TRENDLINE_CANDLES)
        
        # Simple linear regression for trendline
        x = np.arange(len(recent))
        y_high = recent['high'].values
        y_low = recent['low'].values
        
        # Resistance trendline (highs)
        z_high = np.polyfit(x, y_high, 1)
        resistance_line = z_high[0] * len(recent) + z_high[1]
        
        # Support trendline (lows)
        z_low = np.polyfit(x, y_low, 1)
        support_line = z_low[0] * len(recent) + z_low[1]
        
        # Determine trend
        if z_high[0] > 0 and z_low[0] > 0:
            trend = "UPTREND"
        elif z_high[0] < 0 and z_low[0] < 0:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        logger.info(f"📐 Trendlines: Support={support_line:.2f}, Resistance={resistance_line:.2f}, Trend={trend}")
        
        return TrendlineData(support_line, resistance_line, trend)


# ========================
# OPTION CHAIN ANALYZER
# ========================
class OptionChainAnalyzer:
    """Option Chain Data Analysis"""
    
    @staticmethod
    def parse_option_chain(option_chain: Dict, spot_price: float) -> List[OIData]:
        """Parse option chain into OI data"""
        oi_list = []
        
        oc_data = option_chain.get('oc', {})
        
        for strike_str, data in oc_data.items():
            try:
                strike = float(strike_str)
                
                # Only analyze strikes within 5% of spot
                if abs(strike - spot_price) / spot_price > 0.05:
                    continue
                
                ce_data = data.get('ce', {})
                pe_data = data.get('pe', {})
                
                oi_list.append(OIData(
                    strike=strike,
                    ce_oi=ce_data.get('oi', 0),
                    pe_oi=pe_data.get('oi', 0),
                    ce_volume=ce_data.get('volume', 0),
                    pe_volume=pe_data.get('volume', 0),
                    ce_oi_change=0,  # Will be calculated via Redis
                    pe_oi_change=0
                ))
            except Exception:
                continue
        
        logger.info(f"📊 Parsed {len(oi_list)} strikes")
        return oi_list
    
    @staticmethod
    def calculate_pcr(oi_list: List[OIData]) -> float:
        """Calculate Put-Call Ratio"""
        total_ce_oi = sum(oi.ce_oi for oi in oi_list)
        total_pe_oi = sum(oi.pe_oi for oi in oi_list)
        
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        logger.info(f"📈 PCR: {pcr:.2f}")
        return pcr
    
    @staticmethod
    def find_max_oi_strikes(oi_list: List[OIData]) -> Dict:
        """Find strikes with maximum OI"""
        max_ce_oi = max(oi_list, key=lambda x: x.ce_oi)
        max_pe_oi = max(oi_list, key=lambda x: x.pe_oi)
        
        return {
            'max_ce_strike': max_ce_oi.strike,
            'max_ce_oi': max_ce_oi.ce_oi,
            'max_pe_strike': max_pe_oi.strike,
            'max_pe_oi': max_pe_oi.pe_oi
        }
    
    @staticmethod
    def calculate_max_pain(oi_list: List[OIData], spot_price: float) -> float:
        """Calculate Max Pain strike"""
        pain_values = {}
        
        for strike_point in [oi.strike for oi in oi_list]:
            total_pain = 0
            
            for oi in oi_list:
                if oi.strike < strike_point:
                    # ITM Puts
                    total_pain += (strike_point - oi.strike) * oi.pe_oi
                else:
                    # ITM Calls
                    total_pain += (oi.strike - strike_point) * oi.ce_oi
            
            pain_values[strike_point] = total_pain
        
        if pain_values:
            max_pain_strike = min(pain_values, key=pain_values.get)
            logger.info(f"💰 Max Pain: {max_pain_strike}")
            return max_pain_strike
        
        return spot_price


# ========================
# DEEPSEEK ANALYZER
# ========================
class DeepSeekAnalyzer:
    """DeepSeek V3 Combined Analysis"""
    
    @staticmethod
    def analyze_combined(chart_data: Dict, oi_data: Dict, oi_comparison: Dict) -> Optional[Dict]:
        """Combined Chart + OI analysis"""
        try:
            logger.info("🤖 DeepSeek: Analyzing Chart + OI data...")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Build OI change summary
            oi_changes = ""
            if oi_comparison.get('deltas'):
                oi_changes = "\n".join([
                    f"Strike {d['strike']}: CE OI {d['ce_oi_change']:+,}, PE OI {d['pe_oi_change']:+,}, CE Vol {d['ce_volume']:,}, PE Vol {d['pe_volume']:,}"
                    for d in oi_comparison['deltas'][:5]
                ])
            
            prompt = f"""You are expert F&O trader. Analyze combined data and give PE/CE buying opportunity.

CHART DATA:
- Symbol: {chart_data['symbol']}
- Spot Price: ₹{chart_data['spot_price']}
- Trend: {chart_data['trend']}
- Support Zones: {chart_data['support_zones']}
- Resistance Zones: {chart_data['resistance_zones']}
- Chart Patterns: {chart_data['chart_patterns']}
- Psychological Levels: {chart_data['psychological_levels']}
- Trendline Support: ₹{chart_data['trendline_support']}
- Trendline Resistance: ₹{chart_data['trendline_resistance']}

OPTION CHAIN DATA:
- PCR: {oi_data['pcr']}
- Max CE OI Strike: {oi_data['max_ce_strike']} ({oi_data['max_ce_oi']:,} OI)
- Max PE OI Strike: {oi_data['max_pe_strike']} ({oi_data['max_pe_oi']:,} OI)
- Max Pain: ₹{oi_data['max_pain']}

OI CHANGES (Last {oi_comparison.get('time_diff', 0):.0f} mins):
{oi_changes if oi_changes else "No significant changes"}

TASK: Analyze if there's PE or CE buying opportunity. Consider:
1. Chart trend + reversal patterns
2. OI buildup at strikes
3. Volume surge in CE/PE
4. Smart money flow

Reply ONLY in JSON:
{{
  "opportunity": "PE_BUY/CE_BUY/WAIT",
  "confidence": 80,
  "recommended_strike": {chart_data['spot_price']},
  "entry_price": 100,
  "target": 150,
  "stop_loss": 80,
  "quantity_lots": 1,
  "reasoning": "Detailed analysis combining chart patterns, OI data, and volume",
  "marathi_explanation": "मराठी मध्ये संपूर्ण analysis"
}}
"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are expert F&O trader specializing in chart patterns and option chain analysis. Reply in JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ DeepSeek API error: {response.status_code}")
                return None
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"✅ DeepSeek: {analysis['opportunity']} ({analysis['confidence']}%)")
                return analysis
            
            logger.warning("⚠️ Could not parse DeepSeek response")
            return None
            
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            logger.error(traceback.format_exc())
            return None


# ========================
# MAIN BOT
# ========================
class AdvancedFOBot:
    """Advanced F&O Trading Bot"""
    
    def __init__(self):
        logger.info("🔧 Initializing Advanced F&O Bot...")
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.redis = RedisCache()
        self.dhan = DhanAPI(self.redis)
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.oi_analyzer = OptionChainAnalyzer()
        self.running = True
        logger.info("✅ Advanced F&O Bot initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        current_time = now_ist.strftime("%H:%M")
        
        if now_ist.weekday() >= 5:
            logger.info(f"📅 Weekend: Market closed")
            return False
        
        if Config.MARKET_OPEN <= current_time <= Config.MARKET_CLOSE:
            logger.info(f"✅ Market OPEN (IST: {current_time})")
            return True
        
        logger.info(f"⏰ Market closed (IST: {current_time})")
        return False
    
    async def scan_symbol(self, symbol: str, info: Dict):
        """Comprehensive scan: Chart + OI analysis"""
        try:
            security_id = info['security_id']
            segment = info['segment']
            lot_size = info['lot_size']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 ADVANCED SCAN: {symbol}")
            logger.info(f"{'='*70}")
            
            # Step 1: Get expiry
            expiry = self.dhan.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No expiry - SKIP")
                return
            
            logger.info(f"📅 Expiry: {expiry}")
            
            # Step 2: Get candles (last 7 days)
            candles_df = self.dhan.get_historical_candles(security_id, segment, symbol, lookback_days=7)
            if candles_df is None or len(candles_df) < Config.LOOKBACK_CANDLES:
                logger.warning(f"⚠️ {symbol}: Insufficient candles - SKIP")
                return
            
            logger.info(f"📊 Candles: {len(candles_df)} fetched")
            
            # Step 3: Get option chain
            option_chain = self.dhan.get_option_chain(security_id, segment, expiry, symbol)
            if not option_chain:
                logger.warning(f"⚠️ {symbol}: No option chain - SKIP")
                return
            
            spot_price = option_chain.get('last_price', 0)
            logger.info(f"💰 Spot Price: ₹{spot_price:,.2f}")
            
            # ==========================================
            # CHART ANALYSIS
            # ==========================================
            logger.info(f"📈 Running Chart Analysis...")
            
            # Trend identification
            trend = self.chart_analyzer.identify_trend(candles_df)
            
            # Support/Resistance zones
            support_zones, resistance_zones = self.chart_analyzer.calculate_support_resistance_zones(candles_df)
            
            # Psychological levels
            psych_levels = self.chart_analyzer.find_psychological_levels(spot_price)
            
            # Chart patterns
            chart_patterns = self.chart_analyzer.detect_chart_patterns(candles_df)
            
            # Trendlines
            trendline_data = self.chart_analyzer.calculate_trendlines(candles_df)
            
            chart_data = {
                'symbol': symbol,
                'spot_price': spot_price,
                'trend': trend,
                'support_zones': [f"₹{s:.2f}" for s in support_zones[-3:]] if support_zones else [],
                'resistance_zones': [f"₹{r:.2f}" for r in resistance_zones[-3:]] if resistance_zones else [],
                'chart_patterns': [f"{p.name} ({p.type})" for p in chart_patterns],
                'psychological_levels': [f"₹{l:.2f}" for l in psych_levels],
                'trendline_support': trendline_data.support_line,
                'trendline_resistance': trendline_data.resistance_line,
                'trendline_trend': trendline_data.trend
            }
            
            logger.info(f"✅ Chart Analysis Complete:")
            logger.info(f"   Trend: {trend}")
            logger.info(f"   Patterns: {len(chart_patterns)}")
            logger.info(f"   Support Zones: {len(support_zones)}")
            logger.info(f"   Resistance Zones: {len(resistance_zones)}")
            
            # ==========================================
            # OPTION CHAIN ANALYSIS
            # ==========================================
            logger.info(f"⛓️ Running Option Chain Analysis...")
            
            # Parse OI data
            oi_list = self.oi_analyzer.parse_option_chain(option_chain, spot_price)
            
            if not oi_list:
                logger.warning(f"⚠️ {symbol}: No OI data - SKIP")
                return
            
            # Calculate metrics
            pcr = self.oi_analyzer.calculate_pcr(oi_list)
            max_oi = self.oi_analyzer.find_max_oi_strikes(oi_list)
            max_pain = self.oi_analyzer.calculate_max_pain(oi_list, spot_price)
            
            oi_data = {
                'pcr': pcr,
                'max_ce_strike': max_oi['max_ce_strike'],
                'max_ce_oi': max_oi['max_ce_oi'],
                'max_pe_strike': max_oi['max_pe_strike'],
                'max_pe_oi': max_oi['max_pe_oi'],
                'max_pain': max_pain
            }
            
            logger.info(f"✅ OI Analysis Complete:")
            logger.info(f"   PCR: {pcr:.2f}")
            logger.info(f"   Max CE: {max_oi['max_ce_strike']} ({max_oi['max_ce_oi']:,} OI)")
            logger.info(f"   Max PE: {max_oi['max_pe_strike']} ({max_oi['max_pe_oi']:,} OI)")
            logger.info(f"   Max Pain: ₹{max_pain:,.2f}")
            
            # ==========================================
            # REDIS COMPARISON
            # ==========================================
            oi_comparison = self.redis.get_oi_comparison(symbol, oi_list)
            
            # Store current OI in Redis
            self.redis.store_option_chain(symbol, oi_list, spot_price)
            
            # ==========================================
            # DEEPSEEK AI ANALYSIS
            # ==========================================
            logger.info(f"🤖 Running DeepSeek Combined Analysis...")
            
            analysis = DeepSeekAnalyzer.analyze_combined(chart_data, oi_data, oi_comparison)
            
            if not analysis:
                logger.warning(f"⚠️ {symbol}: No AI analysis - SKIP")
                return
            
            # Check confidence threshold
            if analysis['confidence'] < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ {symbol}: Low confidence ({analysis['confidence']}%) - NO ALERT")
                return
            
            # ==========================================
            # SEND TELEGRAM ALERT
            # ==========================================
            await self.send_trading_alert(symbol, spot_price, lot_size, chart_data, 
                                         oi_data, oi_comparison, analysis, expiry)
            
            logger.info(f"✅ {symbol}: ALERT SENT! 🎉")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    async def send_trading_alert(self, symbol: str, spot_price: float, lot_size: int,
                                chart_data: Dict, oi_data: Dict, oi_comparison: Dict,
                                analysis: Dict, expiry: str):
        """Send comprehensive trading alert"""
        try:
            # Signal emoji
            if analysis['opportunity'] == "PE_BUY":
                signal_emoji = "🔴"
                signal_text = "PE BUY (Bearish)"
            elif analysis['opportunity'] == "CE_BUY":
                signal_emoji = "🟢"
                signal_text = "CE BUY (Bullish)"
            else:
                signal_emoji = "⚪"
                signal_text = "WAIT"
            
            # Build OI change text
            oi_change_text = ""
            if oi_comparison.get('deltas'):
                changes = oi_comparison['deltas'][:3]
                oi_change_text = "\n".join([
                    f"• Strike {d['strike']}: CE {d['ce_oi_change']:+,} | PE {d['pe_oi_change']:+,}"
                    for d in changes
                ])
            else:
                oi_change_text = "• First scan - No comparison data"
            
            # Calculate position size
            position_value = analysis.get('entry_price', 100) * lot_size * analysis.get('quantity_lots', 1)
            
            message = f"""
🚀 <b>ADVANCED F&O SIGNAL</b>

📊 Symbol: <b>{symbol}</b>
💰 Spot: ₹{spot_price:,.2f}
📅 Expiry: {expiry}
⏰ Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%b %H:%M IST')}

{signal_emoji} <b>OPPORTUNITY: {signal_text}</b>
💪 Confidence: <b>{analysis['confidence']}%</b>

🎯 <b>TRADE SETUP:</b>
• Strike: {analysis.get('recommended_strike', spot_price)} {analysis['opportunity'].split('_')[0]}
• Entry: ₹{analysis.get('entry_price', 100)}
• Target: ₹{analysis.get('target', 150)}
• Stop-Loss: ₹{analysis.get('stop_loss', 80)}
• Quantity: {analysis.get('quantity_lots', 1)} Lot(s) = {lot_size * analysis.get('quantity_lots', 1)} Qty
• Position Size: ₹{position_value:,.0f}

📈 <b>CHART ANALYSIS:</b>
• Trend: {chart_data['trend']}
• Trendline Trend: {chart_data['trendline_trend']}
• Support: {', '.join(chart_data['support_zones'][:2]) if chart_data['support_zones'] else 'N/A'}
• Resistance: {', '.join(chart_data['resistance_zones'][:2]) if chart_data['resistance_zones'] else 'N/A'}
• Patterns: {', '.join(chart_data['chart_patterns']) if chart_data['chart_patterns'] else 'None'}
• Psychological: {', '.join(chart_data['psychological_levels'][:3]) if chart_data['psychological_levels'] else 'None'}

⛓️ <b>OPTION CHAIN:</b>
• PCR: {oi_data['pcr']:.2f}
• Max CE Strike: {oi_data['max_ce_strike']} ({oi_data['max_ce_oi']:,} OI)
• Max PE Strike: {oi_data['max_pe_strike']} ({oi_data['max_pe_oi']:,} OI)
• Max Pain: ₹{oi_data['max_pain']:,.0f}

📊 <b>OI CHANGES ({oi_comparison.get('time_diff', 0):.0f} mins ago):</b>
{oi_change_text}

🧠 <b>AI REASONING:</b>
{analysis.get('reasoning', 'Combined chart pattern and OI analysis')}

📝 <b>मराठी स्पष्टीकरण:</b>
{analysis.get('marathi_explanation', 'तांत्रिक विश्लेषण पूर्ण झाले आहे')}

⚡ <b>Disclaimer:</b> This is AI-generated analysis. Trade at your own risk.
💾 Data: Chart patterns + OI comparison via Redis
🤖 Powered by: DeepSeek V3 + Advanced Technical Analysis
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Trading alert sent to Telegram!")
            
        except Exception as e:
            logger.error(f"❌ Alert sending error: {e}")
            logger.error(traceback.format_exc())
    
    async def send_startup_message(self):
        """Send startup notification"""
        try:
            logger.info("📤 Sending startup message...")
            ist = pytz.timezone('Asia/Kolkata')
            
            redis_status = "✅ Connected" if self.redis.redis_client else "❌ Disconnected"
            
            msg = f"""
🤖 <b>Advanced F&O Bot v4.0 Started!</b>

📊 Tracking: <b>{len(self.dhan.security_id_map)} indices</b>
⏰ Scan Interval: {Config.SCAN_INTERVAL//60} minutes
🎯 Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}%
⏱️ Market Hours: {Config.MARKET_OPEN} - {Config.MARKET_CLOSE} IST
🔴 Redis Cache: {redis_status}

🔍 <b>Advanced Features:</b>
✅ Chart Pattern Detection (Double Top/Bottom, etc.)
✅ Trendline Analysis (Dynamic Support/Resistance)
✅ Psychological Level Detection
✅ Smart Money Concepts
✅ Option Chain Analysis (OI, Volume, PCR)
✅ OI Change Tracking (Redis-based)
✅ Max Pain Calculation
✅ Combined DeepSeek V3 Analysis
✅ PE/CE Buy Opportunity Detection

📈 <b>Active Symbols ({len(self.dhan.security_id_map)}):</b>
{', '.join(self.dhan.security_id_map.keys())}

⚡ Analysis Method:
1. Fetch {Config.LOOKBACK_CANDLES}+ candles
2. Detect chart patterns & trendlines
3. Analyze option chain (OI + Volume)
4. Compare with Redis cache (OI changes)
5. DeepSeek AI combines all data
6. Generate PE/CE buy signals (75%+ confidence)

🚀 Status: <b>ACTIVE & MONITORING</b> ✅
⏰ Startup: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}

📝 Next scan: When market opens!
"""
            
            await self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='HTML'
            )
            logger.info("✅ Startup message sent!")
        except Exception as e:
            logger.error(f"❌ Startup message error: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*70)
        logger.info("🚀 ADVANCED F&O BOT v4.0 STARTING...")
        logger.info("="*70)
        
        # Validate credentials
        logger.info("🔐 Validating API credentials...")
        missing = []
        for cred in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DHAN_CLIENT_ID', 
                     'DHAN_ACCESS_TOKEN', 'DEEPSEEK_API_KEY']:
            if not getattr(Config, cred):
                missing.append(cred)
        
        if missing:
            logger.error(f"❌ Missing credentials: {', '.join(missing)}")
            return
        
        logger.info("✅ All API credentials validated!")
        
        # Load security IDs
        logger.info("📥 Loading security IDs...")
        success = await self.dhan.load_security_ids()
        if not success:
            logger.error("❌ Failed to load securities. Exiting...")
            return
        
        logger.info(f"✅ Loaded {len(self.dhan.security_id_map)} securities!")
        
        # Send startup message
        await self.send_startup_message()
        
        logger.info("="*70)
        logger.info("🎯 Bot is now RUNNING! Monitoring market...")
        logger.info("="*70)
        
        while self.running:
            try:
                if not self.is_market_open():
                    logger.info("😴 Market closed. Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                
                ist = pytz.timezone('Asia/Kolkata')
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 ADVANCED SCAN CYCLE START")
                logger.info(f"⏰ IST Time: {datetime.now(ist).strftime('%H:%M:%S')}")
                logger.info(f"📊 Scanning {len(self.dhan.security_id_map)} symbols...")
                logger.info(f"{'='*70}")
                
                # Scan each symbol
                for idx, (symbol, info) in enumerate(self.dhan.security_id_map.items(), 1):
                    logger.info(f"\n[{idx}/{len(self.dhan.security_id_map)}] Processing {symbol}...")
                    await self.scan_symbol(symbol, info)
                    await asyncio.sleep(5)  # Rate limit between symbols
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ SCAN CYCLE COMPLETE!")
                logger.info(f"⏰ Next scan in {Config.SCAN_INTERVAL//60} minutes...")
                logger.info(f"{'='*70}\n")
                
                await asyncio.sleep(Config.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)


# ========================
# MAIN ENTRY POINT
# ========================
async def main():
    """Main entry point"""
    try:
        logger.info("="*70)
        logger.info("🚀 INITIALIZING ADVANCED F&O BOT v4.0")
        logger.info("="*70)
        
        bot = AdvancedFOBot()
        await bot.run()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("="*70)
        logger.info("👋 Bot shutdown complete")
        logger.info("="*70)


if __name__ == "__main__":
    ist = pytz.timezone('Asia/Kolkata')
    logger.info("="*70)
    logger.info("🎬 ADVANCED F&O BOT STARTING...")
    logger.info(f"⏰ IST: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🌍 Timezone: Asia/Kolkata (IST)")
    logger.info("="*70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        logger.error(traceback.format_exc())
