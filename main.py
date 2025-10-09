import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime
import logging
import csv
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import pandas as pd
import json
import numpy as np

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# CONFIGURATION
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# Gemini API URL
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Stock/Index List
STOCKS_INDICES = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "BAJFINANCE": {"symbol": "BAJFINANCE", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "LTIM": {"symbol": "LTIM", "segment": "NSE_EQ"},
    "ADANIENT": {"symbol": "ADANIENT", "segment": "NSE_EQ"},
    "KOTAKBANK": {"symbol": "KOTAKBANK", "segment": "NSE_EQ"},
    "LT": {"symbol": "LT", "segment": "NSE_EQ"},
    "MARUTI": {"symbol": "MARUTI", "segment": "NSE_EQ"},
    "TECHM": {"symbol": "TECHM", "segment": "NSE_EQ"},
    "LICI": {"symbol": "LICI", "segment": "NSE_EQ"},
    "HINDUNILVR": {"symbol": "HINDUNILVR", "segment": "NSE_EQ"},
    "NTPC": {"symbol": "NTPC", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
    "POWERGRID": {"symbol": "POWERGRID", "segment": "NSE_EQ"},
    "ONGC": {"symbol": "ONGC", "segment": "NSE_EQ"},
    "PERSISTENT": {"symbol": "PERSISTENT", "segment": "NSE_EQ"},
    "DRREDDY": {"symbol": "DRREDDY", "segment": "NSE_EQ"},
    "M&M": {"symbol": "M&M", "segment": "NSE_EQ"},
    "WIPRO": {"symbol": "WIPRO", "segment": "NSE_EQ"},
    "DMART": {"symbol": "DMART", "segment": "NSE_EQ"},
    "TRENT": {"symbol": "TRENT", "segment": "NSE_EQ"},
}

# ========================
# ENHANCED GEMINI AI
# ========================

class EnhancedGeminiAnalyzer:
    """Deep analysis with trade alerts"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = f"{GEMINI_API_URL}?key={api_key}"
    
    def extract_chart_raw_text(self, candles):
        """Chart data ला text format मध्ये convert करतो"""
        try:
            text = "CANDLE DATA (Last 50 candles):\n"
            text += "Timestamp | Open | High | Low | Close | Volume\n"
            text += "=" * 70 + "\n"
            
            # Last 50 candles घेतो
            recent_candles = candles[-50:] if len(candles) > 50 else candles
            
            for c in recent_candles:
                ts = c.get('timestamp', 'N/A')
                o = c.get('open', 0)
                h = c.get('high', 0)
                l = c.get('low', 0)
                cl = c.get('close', 0)
                v = c.get('volume', 0)
                text += f"{ts} | {o:.2f} | {h:.2f} | {l:.2f} | {cl:.2f} | {v}\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting chart text: {e}")
            return ""
    
    def calculate_technical_indicators(self, candles):
        """Technical indicators calculate करतो"""
        try:
            df = pd.DataFrame(candles)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            
            # ATR (14)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean()
            
            # Volume analysis
            vol_sma20 = df['volume'].rolling(window=20).mean()
            
            latest = {
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else None,
                'macd': macd.iloc[-1] if len(macd) > 0 else None,
                'macd_signal': signal.iloc[-1] if len(signal) > 0 else None,
                'bb_upper': bb_upper.iloc[-1] if len(bb_upper) > 0 else None,
                'bb_lower': bb_lower.iloc[-1] if len(bb_lower) > 0 else None,
                'bb_middle': sma20.iloc[-1] if len(sma20) > 0 else None,
                'atr': atr.iloc[-1] if len(atr) > 0 else None,
                'volume_ratio': (df['volume'].iloc[-1] / vol_sma20.iloc[-1]) if len(vol_sma20) > 0 and vol_sma20.iloc[-1] > 0 else None,
                'current_price': df['close'].iloc[-1],
                'price_change': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 0 else 0
            }
            
            return latest
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def deep_analyze_with_alerts(self, chart_buffer, symbol, candle_data, option_data):
        """DEEP ANALYSIS with TRADE ALERTS"""
        try:
            # 1. Chart image encode
            chart_buffer.seek(0)
            image_bytes = chart_buffer.read()
            chart_buffer.seek(0)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # 2. Technical indicators
            indicators = self.calculate_technical_indicators(candle_data)
            
            # 3. Raw chart text
            chart_text = self.extract_chart_raw_text(candle_data)
            
            # 4. Option chain detailed analysis
            spot_price = option_data.get('last_price', 0)
            oc_data = option_data.get('oc', {})
            
            # Strikes data
            strikes = sorted([float(s) for s in oc_data.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            atm_idx = strikes.index(atm_strike)
            
            # ATM ± 5 strikes
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            relevant_strikes = strikes[start_idx:end_idx]
            
            # Detailed option metrics
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_vol = 0
            total_pe_vol = 0
            max_ce_oi_strike = None
            max_pe_oi_strike = None
            max_ce_oi = 0
            max_pe_oi = 0
            
            oc_detailed = []
            for strike in relevant_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc_data.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                ce_oi = ce.get('oi', 0)
                pe_oi = pe.get('oi', 0)
                ce_vol = ce.get('volume', 0)
                pe_vol = pe.get('volume', 0)
                
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
                total_ce_vol += ce_vol
                total_pe_vol += pe_vol
                
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    max_ce_oi_strike = strike
                
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    max_pe_oi_strike = strike
                
                oc_detailed.append({
                    'strike': strike,
                    'is_atm': strike == atm_strike,
                    'ce_ltp': ce.get('last_price', 0),
                    'ce_oi': ce_oi,
                    'ce_vol': ce_vol,
                    'ce_iv': ce.get('implied_volatility', 0),
                    'ce_delta': ce.get('greeks', {}).get('delta', 0),
                    'ce_theta': ce.get('greeks', {}).get('theta', 0),
                    'pe_ltp': pe.get('last_price', 0),
                    'pe_oi': pe_oi,
                    'pe_vol': pe_vol,
                    'pe_iv': pe.get('implied_volatility', 0),
                    'pe_delta': pe.get('greeks', {}).get('delta', 0),
                    'pe_theta': pe.get('greeks', {}).get('theta', 0)
                })
            
            # PCR
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
            
            # ATM data
            atm_data = oc_data.get(f"{atm_strike:.6f}", {})
            ce_atm = atm_data.get('ce', {})
            pe_atm = atm_data.get('pe', {})
            
            # DEEP PROMPT
            prompt = f"""तुम्ही EXPERT OPTIONS TRADER आणि TECHNICAL ANALYST आहात. खालील सर्व data analyze करून DEEP INSIGHTS आणि TRADE ALERTS द्या.

═══════════════════════════════════════════════════════════
📊 SYMBOL: {symbol}
💰 SPOT PRICE: ₹{spot_price:,.2f}
🎯 ATM STRIKE: ₹{atm_strike:,.0f}
═══════════════════════════════════════════════════════════

📈 TECHNICAL INDICATORS:
• RSI(14): {indicators.get('rsi', 0):.2f}
• MACD: {indicators.get('macd', 0):.2f} | Signal: {indicators.get('macd_signal', 0):.2f}
• Bollinger Bands: Upper={indicators.get('bb_upper', 0):.2f}, Middle={indicators.get('bb_middle', 0):.2f}, Lower={indicators.get('bb_lower', 0):.2f}
• ATR(14): {indicators.get('atr', 0):.2f}
• Volume Ratio: {indicators.get('volume_ratio', 0):.2f}x (vs 20-SMA)
• Price Change: {indicators.get('price_change', 0):.2f}%

═══════════════════════════════════════════════════════════
📊 RAW CHART DATA (Last 50 Candles):
{chart_text}
═══════════════════════════════════════════════════════════

🔢 OPTION CHAIN METRICS:
• Total CE OI: {total_ce_oi:,.0f} | Total PE OI: {total_pe_oi:,.0f}
• PCR (OI): {pcr_oi:.3f} | PCR (Volume): {pcr_vol:.3f}
• Total CE Volume: {total_ce_vol:,.0f} | Total PE Volume: {total_pe_vol:,.0f}
• Max CE OI Strike: ₹{max_ce_oi_strike:,.0f} ({max_ce_oi:,.0f})
• Max PE OI Strike: ₹{max_pe_oi_strike:,.0f} ({max_pe_oi:,.0f})

ATM STRIKE DATA (₹{atm_strike:,.0f}):
• CE: LTP=₹{ce_atm.get('last_price', 0):.2f} | OI={ce_atm.get('oi', 0):,.0f} | Vol={ce_atm.get('volume', 0):,.0f} | IV={ce_atm.get('implied_volatility', 0):.1f}% | Δ={ce_atm.get('greeks', {}).get('delta', 0):.3f} | Θ={ce_atm.get('greeks', {}).get('theta', 0):.2f}
• PE: LTP=₹{pe_atm.get('last_price', 0):.2f} | OI={pe_atm.get('oi', 0):,.0f} | Vol={pe_atm.get('volume', 0):,.0f} | IV={pe_atm.get('implied_volatility', 0):.1f}% | Δ={pe_atm.get('greeks', {}).get('delta', 0):.3f} | Θ={pe_atm.get('greeks', {}).get('theta', 0):.2f}

DETAILED STRIKE-WISE DATA:
{json.dumps(oc_detailed, indent=2)}

═══════════════════════════════════════════════════════════

🤖 CRITICAL ANALYSIS REQUIRED (Marathi मध्ये, detailed):

1️⃣ **CHART PATTERN ANALYSIS** (10-12 lines):
   • Chart मध्ये कोणता pattern दिसतो? (Head & Shoulders, Double Top/Bottom, Triangle, Flag, Pennant etc.)
   • Support आणि Resistance levels identify करा (chart + Bollinger Bands वापरून)
   • Trend काय आहे? (Uptrend/Downtrend/Sideways)
   • Volume analysis - breakout/breakdown confirmation?
   • Candlestick patterns (Doji, Hammer, Shooting Star etc.)
   • RSI, MACD, Bollinger च्या basis वर signal काय आहे?

2️⃣ **OPTION CHAIN DEEP DIVE** (10-12 lines):
   • PCR interpretation - Bullish/Bearish/Neutral sentiment
   • Max Pain Theory - कोणत्या strike वर OI buildup सर्वात जास्त?
   • OI vs Volume analysis - Smart money कुठे move करत आहे?
   • IV Skew - CE vs PE IV comparison आणि त्याचा अर्थ
   • Greeks Analysis - Delta, Theta impact
   • Gamma Squeeze possibility check
   • Option writers (sellers) कुठे aggressive आहेत?

3️⃣ **🚨 TRADE ALERTS - CE/PE BUY/SELL SIGNALS** (8-10 lines):
   **IF BULLISH SETUP:**
   • CE BUY: Strike={}, Entry=₹{}, Target=₹{}, SL=₹{}, Risk:Reward=1:{}
   • PE SELL (if premium juicy): Strike={}, Entry=₹{}, SL=₹{}
   • Spot breakout level: ₹{}, confirm होण्यासाठी volume > {}
   
   **IF BEARISH SETUP:**
   • PE BUY: Strike={}, Entry=₹{}, Target=₹{}, SL=₹{}, Risk:Reward=1:{}
   • CE SELL (if premium juicy): Strike={}, Entry=₹{}, SL=₹{}
   • Spot breakdown level: ₹{}, confirm होण्यासाठी volume > {}
   
   **IF NEUTRAL/RANGE-BOUND:**
   • IRON CONDOR / STRADDLE SELL strategy suggest करा
   • Range: ₹{} to ₹{}
   • Time decay benefit कसा घ्यायचा?

4️⃣ **RISK MANAGEMENT & POSITION SIZING** (5-6 lines):
   • Maximum loss per trade: {}% of capital
   • Position size calculate कसा करायचा?
   • Hedge strategy (if any)
   • Exit plan - profit booking levels
   • Avoid करण्याजोगी mistakes

5️⃣ **KEY LEVELS & TARGETS** (5-6 lines):
   • Immediate Support: ₹{}, ₹{}
   • Immediate Resistance: ₹{}, ₹{}
   • Day's range: ₹{} - ₹{}
   • Breakout target (if bullish): ₹{}
   • Breakdown target (if bearish): ₹{}

6️⃣ **MARKET SENTIMENT SUMMARY** (4-5 lines):
   • Overall sentiment (Bullish/Bearish/Neutral) - 0-100 scale वर rating द्या
   • Confidence level: High/Medium/Low
   • Time horizon: Intraday / Swing (1-2 days)
   • Special notes / Red flags (if any)

═══════════════════════════════════════════════════════════

⚠️ IMPORTANT INSTRUCTIONS:
• प्रत्येक section मध्ये specific numbers द्या, vague statements नको
• Trade alerts मध्ये exact strike, entry, target, SL द्या
• जर clear setup नसेल, तर "WAIT & WATCH" recommend करा
• Risk:Reward ratio 1:2 किंवा जास्त असावं
• Chart image पण बघून pattern verify करा
• सर्व analysis data-driven आणि actionable असावं

FORMAT: Marathi मध्ये, numbered sections, bullet points, clear आणि crisp."""

            # Gemini API call
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,  # More focused responses
                    "maxOutputTokens": 4096,  # Increased for detailed analysis
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        analysis_text = candidate['content']['parts'][0].get('text', '')
                        
                        logger.info(f"✅ DEEP Gemini analysis completed for {symbol}")
                        return analysis_text
                
                logger.warning(f"⚠️ Unexpected Gemini response for {symbol}")
                return None
            else:
                logger.error(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error in deep analysis: {e}")
            return None


# ========================
# ENHANCED BOT
# ========================

class DhanOptionChainBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_id_map = {}
        
        if GEMINI_API_KEY:
            self.gemini = EnhancedGeminiAnalyzer(GEMINI_API_KEY)
            logger.info("✅ Enhanced Gemini AI Analyzer initialized")
        else:
            self.gemini = None
            logger.warning("⚠️ GEMINI_API_KEY not found")
        
        logger.info("Bot initialized")
    
    async def load_security_ids(self):
        """Security IDs load"""
        try:
            logger.info("Loading security IDs...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code == 200:
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
                
                for symbol, info in STOCKS_INDICES.items():
                    segment = info['segment']
                    symbol_name = info['symbol']
                    
                    for row in reader:
                        try:
                            if segment == "IDX_I":
                                if (row.get('SEM_SEGMENT') == 'I' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name
                                        }
                                        logger.info(f"✅ {symbol}: {sec_id}")
                                        break
                            else:
                                if (row.get('SEM_SEGMENT') == 'E' and 
                                    row.get('SEM_TRADING_SYMBOL') == symbol_name and
                                    row.get('SEM_EXM_EXCH_ID') == 'NSE'):
                                    sec_id = row.get('SEM_SMST_SECURITY_ID')
                                    if sec_id:
                                        self.security_id_map[symbol] = {
                                            'security_id': int(sec_id),
                                            'segment': segment,
                                            'trading_symbol': symbol_name
                                        }
                                        logger.info(f"✅ {symbol}: {sec_id}")
                                        break
                        except:
                            continue
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"Loaded {len(self.security_id_map)} securities")
                return True
            else:
                logger.error(f"Failed to load instruments: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading security IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Historical 5-min candles"""
        try:
            from datetime import datetime, timedelta
            
            if segment == "IDX_I":
                exch_seg = "IDX_I"
                instrument = "INDEX"
            else:
                exch_seg = "NSE_EQ"
                instrument = "EQUITY"
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg,
                "instrument": instrument,
                "interval": "5",
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
            
            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'open' in data:
                    opens = data.get('open', [])
                    highs = data.get('high', [])
                    lows = data.get('low', [])
                    closes = data.get('close', [])
                    volumes = data.get('volume', [])
                    timestamps = data.get('start_Time', [])
                    
                    candles = []
                    for i in range(len(opens)):
                        candles.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else '',
                            'open': opens[i],
                            'high': highs[i],
                            'low': lows[i],
                            'close': closes[i],
                            'volume': volumes[i]
                        })
                    
                    logger.info(f"{symbol}: {len(candles)} candles")
                    return candles
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def create_enhanced_chart(self, candles, symbol, spot_price):
        """Enhanced high-quality chart"""
        try:
            df_data = []
            for candle in candles:
                timestamp = candle.get('timestamp', '')
                df_data.append({
                    'Date': pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now(),
                    'Open': float(candle.get('open', 0)),
                    'High': float(candle.get('high', 0)),
                    'Low': float(candle.get('low', 0)),
                    'Close': float(candle.get('close', 0)),
                    'Volume': int(float(candle.get('volume', 0)))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            if len(df) < 2:
                return None
            
            # Calculate indicators for overlay
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma20 + (std20 * 2)
            df['BB_Lower'] = sma20 - (std20 * 2)
            
            # Custom style - high quality
            mc = mpf.make_marketcolors(
                up='#00ff88',
                down='#ff4444',
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#404040',
                facecolor='#0a0a0a',
                figcolor='#0a0a0a',
                gridaxis='both',
                y_on_right=False,
                rc={
                    'font.size': 10,
                    'axes.labelsize': 11,
                    'axes.titlesize': 13,
                    'xtick.labelsize': 9,
                    'ytick.labelsize': 9,
                    'legend.fontsize': 9
                }
            )
            
            # Additional plots for indicators
            apds = [
                mpf.make_addplot(df['SMA20'], color='#ffa500', width=1.5, secondary_y=False),
                mpf.make_addplot(df['SMA50'], color='#00bfff', width=1.5, secondary_y=False),
                mpf.make_addplot(df['BB_Upper'], color='#9370db', width=1, linestyle='--', secondary_y=False),
                mpf.make_addplot(df['BB_Lower'], color='#9370db', width=1, linestyle='--', secondary_y=False),
            ]
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                addplot=apds,
                title=f'\n{symbol} | Spot: ₹{spot_price:,.2f} | Candles: {len(candles)}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(14, 10),
                returnfig=True,
                tight_layout=True,
                panel_ratios=(3, 1)
            )
            
            # Title styling
            axes[0].set_title(
                f'{symbol} | Spot: ₹{spot_price:,.2f} | Candles: {len(candles)}',
                color='white',
                fontsize=15,
                fontweight='bold',
                pad=25
            )
            
            # Legend
            axes[0].legend(['SMA20', 'SMA50', 'BB Upper', 'BB Lower'], 
                          loc='upper left', 
                          facecolor='#1a1a1a', 
                          edgecolor='white',
                          labelcolor='white')
            
            # Axis styling
            for ax in axes:
                ax.tick_params(colors='white', which='both')
                ax.spines['bottom'].set_color('#606060')
                ax.spines['top'].set_color('#606060')
                ax.spines['left'].set_color('#606060')
                ax.spines['right'].set_color('#606060')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Save high-quality image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#0a0a0a', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Nearest expiry"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment
            }
            
            response = requests.post(
                DHAN_EXPIRY_LIST_URL,
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
            logger.error(f"Error getting expiry: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Option chain data"""
        try:
            payload = {
                "UnderlyingScrip": security_id,
                "UnderlyingSeg": segment,
                "Expiry": expiry
            }
            
            response = requests.post(
                DHAN_OPTION_CHAIN_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    def format_option_chain_message(self, symbol, data, expiry):
        """Enhanced option chain formatting"""
        try:
            spot_price = data.get('last_price', 0)
            oc_data = data.get('oc', {})
            
            if not oc_data:
                return None
            
            strikes = sorted([float(s) for s in oc_data.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 7)
            end_idx = min(len(strikes), atm_idx + 8)
            selected_strikes = strikes[start_idx:end_idx]
            
            # Calculate PCR
            total_ce_oi = sum([oc_data.get(f"{s:.6f}", {}).get('ce', {}).get('oi', 0) for s in selected_strikes])
            total_pe_oi = sum([oc_data.get(f"{s:.6f}", {}).get('pe', {}).get('oi', 0) for s in selected_strikes])
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            msg = f"📊 *{symbol} OPTION CHAIN*\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📅 Expiry: `{expiry}`\n"
            msg += f"💰 Spot: `₹{spot_price:,.2f}`\n"
            msg += f"🎯 ATM: `₹{atm_strike:,.0f}`\n"
            msg += f"📈 PCR (OI): `{pcr:.3f}`\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += "```\n"
            msg += "Strike    CE-LTP   CE-OI   CE-Vol   PE-LTP   PE-OI   PE-Vol\n"
            msg += "═══════════════════════════════════════════════════════════\n"
            
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc_data.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                ce_ltp = ce.get('last_price', 0)
                ce_oi = ce.get('oi', 0)
                ce_vol = ce.get('volume', 0)
                
                pe_ltp = pe.get('last_price', 0)
                pe_oi = pe.get('oi', 0)
                pe_vol = pe.get('volume', 0)
                
                atm_mark = "🎯" if strike == atm_strike else "  "
                
                msg += f"{atm_mark}{strike:7.0f}  {ce_ltp:7.1f} {ce_oi/1000:7.0f}K {ce_vol/1000:6.0f}K  {pe_ltp:7.1f} {pe_oi/1000:7.0f}K {pe_vol/1000:6.0f}K\n"
            
            msg += "```\n\n"
            
            # ATM Greeks
            atm_data = oc_data.get(f"{atm_strike:.6f}", {})
            if atm_data:
                ce_atm = atm_data.get('ce', {})
                pe_atm = atm_data.get('pe', {})
                
                ce_greeks = ce_atm.get('greeks', {})
                pe_greeks = pe_atm.get('greeks', {})
                ce_iv = ce_atm.get('implied_volatility', 0)
                pe_iv = pe_atm.get('implied_volatility', 0)
                
                msg += "📊 *ATM Greeks & Volatility:*\n"
                msg += f"• CE: Δ=`{ce_greeks.get('delta', 0):.3f}` | Θ=`{ce_greeks.get('theta', 0):.2f}` | IV=`{ce_iv:.1f}%`\n"
                msg += f"• PE: Δ=`{pe_greeks.get('delta', 0):.3f}` | Θ=`{pe_greeks.get('theta', 0):.2f}` | IV=`{pe_iv:.1f}%`\n"
            
            return msg
            
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return None
    
    async def send_option_chain_batch(self, symbols_batch):
        """Batch processing with deep AI analysis"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    logger.warning(f"Skipping {symbol}")
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                # Get expiry
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    logger.warning(f"{symbol}: No expiry")
                    continue
                
                logger.info(f"📊 Processing {symbol} (Expiry: {expiry})...")
                
                # Option chain
                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"{symbol}: No option chain data")
                    continue
                
                spot_price = oc_data.get('last_price', 0)
                
                # Historical candles
                logger.info(f"📈 Fetching candles for {symbol}...")
                candles = self.get_historical_data(security_id, segment, symbol)
                
                # Enhanced chart
                chart_buf = None
                if candles:
                    logger.info(f"🎨 Creating enhanced chart for {symbol}...")
                    chart_buf = self.create_enhanced_chart(candles, symbol, spot_price)
                
                # 🤖 DEEP AI ANALYSIS
                ai_analysis = None
                if self.gemini and chart_buf and candles:
                    logger.info(f"🤖 Running DEEP Gemini analysis for {symbol}...")
                    ai_analysis = self.gemini.deep_analyze_with_alerts(
                        chart_buf, 
                        symbol, 
                        candles, 
                        oc_data
                    )
                
                # 1️⃣ Send chart
                if chart_buf:
                    chart_buf.seek(0)
                    await self.bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID,
                        photo=chart_buf,
                        caption=f"📊 {symbol} - Enhanced Chart with Indicators\n🕐 Last {len(candles)} Candles (5-min)"
                    )
                    logger.info(f"✅ Chart sent for {symbol}")
                    await asyncio.sleep(1.5)
                
                # 2️⃣ Send option chain
                message = self.format_option_chain_message(symbol, oc_data, expiry)
                if message:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=message,
                        parse_mode='Markdown'
                    )
                    logger.info(f"✅ Option chain sent for {symbol}")
                    await asyncio.sleep(1.5)
                
                # 3️⃣ Send AI analysis with alerts
                if ai_analysis:
                    # Split if too long
                    max_length = 4096
                    if len(ai_analysis) > max_length:
                        parts = [ai_analysis[i:i+max_length] for i in range(0, len(ai_analysis), max_length)]
                        for idx, part in enumerate(parts, 1):
                            header = f"🤖 *DEEP AI ANALYSIS - {symbol}* (Part {idx}/{len(parts)})\n\n"
                            await self.bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                text=header + part,
                                parse_mode='Markdown'
                            )
                            await asyncio.sleep(1.5)
                    else:
                        ai_msg = f"🤖 *DEEP AI ANALYSIS - {symbol}*\n\n{ai_analysis}"
                        await self.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=ai_msg,
                            parse_mode='Markdown'
                        )
                    
                    logger.info(f"✅ AI analysis sent for {symbol}")
                
                # Rate limit
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(3)
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Enhanced Bot started!")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("Failed to load security IDs")
            return
        
        await self.send_startup_message()
        
        # Batches
        all_symbols = list(self.security_id_map.keys())
        batch_size = 5
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"Total {len(all_symbols)} symbols in {len(batches)} batches")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Update cycle at {timestamp}")
                logger.info(f"{'='*60}")
                
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    await self.send_option_chain_batch(batch)
                    
                    if batch_num < len(batches):
                        logger.info("⏱️ Waiting 5 seconds...")
                        await asyncio.sleep(5)
                
                logger.info("\n✅ All batches completed!")
                logger.info("⏳ Next cycle in 5 minutes...\n")
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup message"""
        try:
            msg = "🚀 *Enhanced Dhan Option Bot + Deep AI*\n\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📊 Tracking: `{len(self.security_id_map)}` stocks/indices\n"
            msg += "⏱️ Update: Every 5 minutes\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            msg += "✨ *Features:*\n"
            msg += "• 📈 Enhanced Charts (SMA, Bollinger Bands)\n"
            msg += "• 📉 Complete Option Chain (CE/PE)\n"
            msg += "• 🎯 Greeks & IV Analysis\n"
            msg += "• 📊 Technical Indicators (RSI, MACD, ATR)\n\n"
            
            if self.gemini:
                msg += "🤖 *AI-Powered Analysis:*\n"
                msg += "• Chart Pattern Recognition\n"
                msg += "• Deep Option Chain Insights\n"
                msg += "• PCR & OI Analysis\n"
                msg += "• 🚨 *Trade Alerts (CE/PE Buy/Sell)*\n"
                msg += "• Entry, Target, Stop Loss Levels\n"
                msg += "• Risk Management Tips\n"
                msg += "• Support/Resistance Levels\n"
            else:
                msg += "⚠️ AI Analysis: Disabled\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            msg += "⚡ Powered by:\n"
            msg += "• DhanHQ API v2\n"
            msg += "• Google Gemini 2.0 Flash\n"
            msg += "• Railway.app Deployment\n\n"
            msg += "🕐 Market: 9:15 AM - 3:30 PM (Mon-Fri)\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup: {e}")


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    try:
        required_vars = [
            TELEGRAM_BOT_TOKEN, 
            TELEGRAM_CHAT_ID, 
            DHAN_CLIENT_ID, 
            DHAN_ACCESS_TOKEN
        ]
        
        if not all(required_vars):
            logger.error("❌ Missing environment variables!")
            exit(1)
        
        if not GEMINI_API_KEY:
            logger.warning("⚠️ GEMINI_API_KEY not set - AI disabled")
        
        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
