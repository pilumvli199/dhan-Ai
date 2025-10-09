import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime
import logging
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import pandas as pd
import google.generativeai as genai
from PIL import Image

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

# Stock/Index List
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    
    # Stocks
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
# TECHNICAL ANALYSIS HELPER
# ========================
class TechnicalAnalyzer:
    """Candlestick data analyze करतो"""
    
    @staticmethod
    def calculate_indicators(candles):
        """Technical indicators calculate करतो"""
        try:
            if not candles or len(candles) < 20:
                return None
            
            closes = [c['close'] for c in candles[-50:]]
            highs = [c['high'] for c in candles[-50:]]
            lows = [c['low'] for c in candles[-50:]]
            volumes = [c['volume'] for c in candles[-50:]]
            
            # Simple Moving Averages
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            
            # RSI (14 period)
            rsi = TechnicalAnalyzer._calculate_rsi(closes, 14)
            
            # Support/Resistance (last 50 candles)
            resistance = max(highs[-50:]) if len(highs) >= 50 else max(highs)
            support = min(lows[-50:]) if len(lows) >= 50 else min(lows)
            
            # Volume analysis
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            volume_spike = current_volume > (avg_volume * 1.5)
            
            # Trend detection
            if sma_20 and sma_50:
                trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            return {
                'sma_20': round(sma_20, 2) if sma_20 else None,
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'rsi': round(rsi, 2) if rsi else None,
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'trend': trend,
                'volume_spike': volume_spike,
                'avg_volume': int(avg_volume)
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """RSI calculate करतो"""
        try:
            if len(prices) < period + 1:
                return None
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return None
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return None
    
    @staticmethod
    def detect_candlestick_patterns(candles):
        """Candlestick patterns detect करतो"""
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3] if len(candles) >= 3 else None
        
        # Doji
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        if body < (range_size * 0.1) and range_size > 0:
            patterns.append("🔵 DOJI (Indecision)")
        
        # Hammer
        if last['close'] > last['open']:
            lower_wick = last['open'] - last['low']
            upper_wick = last['high'] - last['close']
            body = last['close'] - last['open']
            if lower_wick > (body * 2) and upper_wick < body:
                patterns.append("🔨 HAMMER (Bullish Reversal)")
        
        # Shooting Star
        if last['close'] < last['open']:
            upper_wick = last['high'] - last['open']
            lower_wick = last['close'] - last['low']
            body = last['open'] - last['close']
            if upper_wick > (body * 2) and lower_wick < body:
                patterns.append("⭐ SHOOTING STAR (Bearish Reversal)")
        
        # Bullish Engulfing
        if prev['close'] < prev['open'] and last['close'] > last['open']:
            if last['open'] < prev['close'] and last['close'] > prev['open']:
                patterns.append("🟢 BULLISH ENGULFING (Strong Buy)")
        
        # Bearish Engulfing
        if prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['open'] > prev['close'] and last['close'] < prev['open']:
                patterns.append("🔴 BEARISH ENGULFING (Strong Sell)")
        
        # Morning Star (3 candles)
        if prev2:
            if (prev2['close'] < prev2['open'] and 
                abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.3 and
                last['close'] > last['open'] and
                last['close'] > (prev2['open'] + prev2['close']) / 2):
                patterns.append("🌅 MORNING STAR (Bullish Reversal)")
        
        # Evening Star (3 candles)
        if prev2:
            if (prev2['close'] > prev2['open'] and
                abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.3 and
                last['close'] < last['open'] and
                last['close'] < (prev2['open'] + prev2['close']) / 2):
                patterns.append("🌆 EVENING STAR (Bearish Reversal)")
        
        return patterns

# ========================
# GEMINI VISION ANALYZER
# ========================
class GeminiChartAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.request_count = 0
        self.last_reset_time = datetime.now()
        self.max_rpm = 15
        
    async def analyze_comprehensive(self, image_buffer, symbol, spot_price, 
                                   option_data, technical_data, candlestick_patterns):
        """सर्व data एकत्र analyze करतो"""
        try:
            await self._check_rate_limit()
            
            image_buffer.seek(0)
            image = Image.open(image_buffer)
            
            # Option chain summary तयार करतो
            oc_summary = self._prepare_option_chain_summary(option_data)
            
            # Technical indicators summary
            tech_summary = self._prepare_technical_summary(technical_data)
            
            # Candlestick patterns
            pattern_summary = "\n".join(candlestick_patterns) if candlestick_patterns else "No significant patterns"
            
            # Comprehensive prompt
            prompt = f"""
You are an EXPERT Indian stock market trader analyzing {symbol} (₹{spot_price:,.2f}).

📊 **CHART ANALYSIS:**
Analyze the candlestick chart image for visual patterns like:
- Head & Shoulders, Double Top/Bottom, Triangles, Wedges, Channels
- Trend lines, breakouts, breakdowns
- Volume patterns and price action

💹 **TECHNICAL DATA PROVIDED:**
{tech_summary}

🕯️ **CANDLESTICK PATTERNS DETECTED:**
{pattern_summary}

📉 **OPTION CHAIN DATA:**
{oc_summary}

**GIVE YOUR ANALYSIS IN THIS FORMAT:**

🎯 **MARKET OUTLOOK:** [Bullish/Bearish/Neutral with reasoning]

📊 **CHART PATTERN:** [What visual pattern you see in chart + significance]

📈 **TECHNICAL CONFIRMATION:**
- Trend: [Align with SMA/RSI data]
- Momentum: [Based on RSI + Volume]
- Support/Resistance: [Key levels]

🔥 **OPTION MARKET SENTIMENT:**
[Analysis based on CE/PE OI, volume, IV changes]

🎯 **TRADE SETUP (If tradeable):**
✅ **Signal:** BUY/SELL/HOLD
💰 **Entry:** ₹[price]
🎯 **Target 1:** ₹[price] 
🎯 **Target 2:** ₹[price]
🛑 **Stop Loss:** ₹[price]
📊 **Risk:Reward:** [ratio]
⏰ **Timeframe:** [Intraday/Swing/Positional]

⚠️ **RISK LEVEL:** [Low/Medium/High] - [Why?]

🔮 **KEY POINTS:**
- [Important observation 1]
- [Important observation 2]
- [Important observation 3]

Keep it CONCISE, ACTIONABLE, and in HINDI+ENGLISH mix if needed for clarity.
Only give trade setup if confidence > 70%.
"""
            
            response = self.model.generate_content([prompt, image])
            
            self.request_count += 1
            logger.info(f"✅ Gemini comprehensive analysis for {symbol} (#{self.request_count})")
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini analysis error for {symbol}: {e}")
            return None
    
    def _prepare_option_chain_summary(self, option_data):
        """Option chain data summarize करतो"""
        try:
            if not option_data or 'oc' not in option_data:
                return "Option data not available"
            
            spot = option_data.get('last_price', 0)
            oc = option_data.get('oc', {})
            
            # ATM strike शोधतो
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce = atm_data.get('ce', {})
            pe = atm_data.get('pe', {})
            
            # CE/PE Ratio
            ce_oi = ce.get('oi', 0)
            pe_oi = pe.get('oi', 0)
            pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
            
            # IV
            ce_iv = ce.get('implied_volatility', 0)
            pe_iv = pe.get('implied_volatility', 0)
            
            summary = f"""
Spot: ₹{spot:,.2f} | ATM: ₹{atm_strike:,.0f}
CE OI: {ce_oi/1000:.0f}K | PE OI: {pe_oi/1000:.0f}K
PCR Ratio: {pcr} (>1=Bullish, <1=Bearish)
CE IV: {ce_iv:.1f}% | PE IV: {pe_iv:.1f}%
CE LTP: ₹{ce.get('last_price', 0):.1f} | PE LTP: ₹{pe.get('last_price', 0):.1f}
"""
            return summary.strip()
        except Exception as e:
            return "Option summary error"
    
    def _prepare_technical_summary(self, tech_data):
        """Technical data format करतो"""
        if not tech_data:
            return "Technical data not available"
        
        summary = f"""
SMA 20: ₹{tech_data.get('sma_20', 'N/A')} | SMA 50: ₹{tech_data.get('sma_50', 'N/A')}
RSI(14): {tech_data.get('rsi', 'N/A')} (>70=Overbought, <30=Oversold)
Trend: {tech_data.get('trend', 'N/A')}
Support: ₹{tech_data.get('support', 'N/A')} | Resistance: ₹{tech_data.get('resistance', 'N/A')}
Volume Spike: {'YES ⚠️' if tech_data.get('volume_spike') else 'No'}
Avg Volume: {tech_data.get('avg_volume', 0):,}
"""
        return summary.strip()
    
    async def _check_rate_limit(self):
        """Rate limit manage करतो"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_reset_time).total_seconds()
        
        if time_diff >= 60:
            self.request_count = 0
            self.last_reset_time = current_time
            logger.info("🔄 Rate limit reset")
        
        if self.request_count >= self.max_rpm:
            wait_time = 60 - time_diff
            if wait_time > 0:
                logger.warning(f"⏳ Rate limit hit. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time + 1)
                self.request_count = 0
                self.last_reset_time = datetime.now()
        
        await asyncio.sleep(4)


# ========================
# BOT CODE
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
        self.gemini_analyzer = GeminiChartAnalyzer(GEMINI_API_KEY)
        self.tech_analyzer = TechnicalAnalyzer()
        logger.info("🤖 Bot initialized with comprehensive analysis")
    
    async def load_security_ids(self):
        """Security IDs load करतो"""
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
                
                logger.info(f"✅ {len(self.security_id_map)} securities loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Historical candles घेतो"""
        try:
            from datetime import timedelta
            
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
                    
                    logger.info(f"{symbol}: {len(candles)} candles fetched")
                    return candles
            
            return None
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None
    
    def create_candlestick_chart(self, candles, symbol, spot_price):
        """Enhanced chart"""
        try:
            df_data = []
            for candle in candles:
                timestamp = candle.get('timestamp', '')
                df_data.append({
                    'Date': pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now(),
                    'Open': float(candle['open']),
                    'High': float(candle['high']),
                    'Low': float(candle['low']),
                    'Close': float(candle['close']),
                    'Volume': int(float(candle['volume']))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            if len(df) < 2:
                return None
            
            mc = mpf.make_marketcolors(
                up='#00ff88',
                down='#ff3366',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#2a2a2a',
                facecolor='#0a0a0a',
                figcolor='#0a0a0a',
                y_on_right=False
            )
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} | ₹{spot_price:,.2f} | {len(candles)} Candles',
                ylabel='Price (₹)',
                ylabel_lower='Vol',
                figsize=(14, 9),
                returnfig=True,
                tight_layout=True
            )
            
            axes[0].set_title(
                f'{symbol} | ₹{spot_price:,.2f} | {len(candles)} Candles (5min)',
                color='#00ff88',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            
            for ax in axes:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('#444')
                ax.spines['top'].set_color('#444')
                ax.spines['left'].set_color('#444')
                ax.spines['right'].set_color('#444')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0a')
            buf.seek(0)
            plt.close(fig)
            
            return buf
        except Exception as e:
            logger.error(f"Chart error for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Expiry घेतो"""
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
            logger.error(f"Expiry error: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Option chain घेतो"""
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
            logger.error(f"Option chain error: {e}")
            return None
    
    def format_option_chain_message(self, symbol, data, expiry):
        """Option chain format करतो"""
        try:
            spot_price = data.get('last_price', 0)
            oc_data = data.get('oc', {})
            
            if not oc_data:
                return None
            
            strikes = sorted([float(s) for s in oc_data.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            selected_strikes = strikes[start_idx:end_idx]
            
            msg = f"📊 *{symbol} OPTION CHAIN*\n"
            msg += f"📅 Expiry: {expiry}\n"
            msg += f"💰 Spot: ₹{spot_price:,.2f}\n"
            msg += f"🎯 ATM: ₹{atm_strike:,.0f}\n\n"
            
            msg += "```\n"
            msg += "Strike   CE-LTP  CE-OI  PE-LTP  PE-OI\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc_data.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                atm_mark = "🔸" if strike == atm_strike else "  "
                
                msg += f"{atm_mark}{strike:6.0f}  {ce.get('last_price', 0):6.1f} {ce.get('oi', 0)/1000:5.0f}K  {pe.get('last_price', 0):6.1f} {pe.get('oi', 0)/1000:5.0f}K\n"
            
            msg += "```"
            
            return msg
        except Exception as e:
            logger.error(f"Format error for {symbol}: {e}")
            return None
    
    async def send_option_chain_batch(self, symbols_batch):
        """Comprehensive analysis सह batch process करतो"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    logger.warning(f"⚠️ {symbol} - No security ID")
                    continue
                
                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']
                
                logger.info(f"\n{'='*50}")
                logger.info(f"🔍 Analyzing {symbol}...")
                logger.info(f"{'='*50}")
                
                # 1. Expiry fetch करतो
                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    logger.warning(f"{symbol}: No expiry found")
                    continue
                
                # 2. Option chain data
                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"{symbol}: No option chain")
                    continue
                
                spot_price = oc_data.get('last_price', 0)
                
                # 3. Historical candles fetch करतो
                logger.info(f"📊 Fetching candles for {symbol}...")
                candles = self.get_historical_data(security_id, segment, symbol)
                
                if not candles or len(candles) < 20:
                    logger.warning(f"{symbol}: Insufficient candle data")
                    continue
                
                # 4. Technical Analysis करतो
                logger.info(f"📈 Calculating technical indicators...")
                technical_data = self.tech_analyzer.calculate_indicators(candles)
                
                # 5. Candlestick Pattern Detection
                logger.info(f"🕯️ Detecting candlestick patterns...")
                patterns = self.tech_analyzer.detect_candlestick_patterns(candles)
                
                # 6. Chart बनवतो
                logger.info(f"📊 Creating chart...")
                chart_buf = self.create_candlestick_chart(candles, symbol, spot_price)
                
                if not chart_buf:
                    logger.warning(f"{symbol}: Chart creation failed")
                    continue
                
                # 7. Chart पाठवतो (पहिले)
                await self.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=chart_buf,
                    caption=f"📊 *{symbol}* - Chart Analysis\n💰 Spot: ₹{spot_price:,.2f}"
                )
                logger.info(f"✅ Chart sent for {symbol}")
                await asyncio.sleep(1)
                
                # 8. Technical Summary पाठवतो
                if technical_data:
                    tech_msg = f"📈 *TECHNICAL ANALYSIS - {symbol}*\n\n"
                    tech_msg += f"💰 Price: ₹{spot_price:,.2f}\n"
                    tech_msg += f"📊 Trend: *{technical_data['trend']}*\n"
                    tech_msg += f"📉 SMA(20): ₹{technical_data['sma_20']}\n"
                    tech_msg += f"📉 SMA(50): ₹{technical_data['sma_50']}\n"
                    tech_msg += f"⚡ RSI(14): {technical_data['rsi']}\n"
                    tech_msg += f"🔼 Resistance: ₹{technical_data['resistance']:,.2f}\n"
                    tech_msg += f"🔽 Support: ₹{technical_data['support']:,.2f}\n"
                    tech_msg += f"📊 Volume Spike: {'YES ⚠️' if technical_data['volume_spike'] else 'No'}\n"
                    
                    if patterns:
                        tech_msg += f"\n🕯️ *Patterns Detected:*\n"
                        for pattern in patterns:
                            tech_msg += f"  {pattern}\n"
                    
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=tech_msg,
                        parse_mode='Markdown'
                    )
                    logger.info(f"✅ Technical analysis sent")
                    await asyncio.sleep(1)
                
                # 9. Option Chain पाठवतो
                oc_message = self.format_option_chain_message(symbol, oc_data, expiry)
                if oc_message:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=oc_message,
                        parse_mode='Markdown'
                    )
                    logger.info(f"✅ Option chain sent")
                    await asyncio.sleep(1)
                
                # 10. 🤖 GEMINI AI COMPREHENSIVE ANALYSIS
                logger.info(f"🤖 Running Gemini AI comprehensive analysis...")
                chart_buf.seek(0)  # Buffer reset
                
                ai_analysis = await self.gemini_analyzer.analyze_comprehensive(
                    chart_buf, 
                    symbol, 
                    spot_price,
                    oc_data,
                    technical_data,
                    patterns
                )
                
                if ai_analysis:
                    # AI Analysis पाठवतो
                    ai_msg = f"🤖 *GEMINI AI ANALYSIS - {symbol}*\n"
                    ai_msg += f"{'='*40}\n\n"
                    ai_msg += ai_analysis
                    
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=ai_msg,
                        parse_mode='Markdown'
                    )
                    logger.info(f"✅ AI comprehensive analysis sent for {symbol}")
                else:
                    logger.warning(f"⚠️ AI analysis failed for {symbol}")
                
                # Final separator
                separator_msg = f"\n{'━'*40}\n✅ *{symbol} Analysis Complete*\n{'━'*40}"
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=separator_msg,
                    parse_mode='Markdown'
                )
                
                # Rate limiting (Dhan + Gemini)
                logger.info(f"⏳ Cooling down 5 seconds...")
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Comprehensive Trading Bot...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        batch_size = 2  # Small batches for detailed analysis
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
        
        logger.info(f"📊 Total: {len(all_symbols)} symbols in {len(batches)} batches")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 NEW CYCLE: {timestamp}")
                logger.info(f"{'='*60}")
                
                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Batch {batch_num}/{len(batches)}: {batch}")
                    await self.send_option_chain_batch(batch)
                    
                    if batch_num < len(batches):
                        logger.info(f"⏳ Inter-batch wait: 15 seconds...")
                        await asyncio.sleep(15)
                
                logger.info("\n" + "="*60)
                logger.info("✅ CYCLE COMPLETED!")
                logger.info("⏳ Next cycle in 5 minutes...")
                logger.info("="*60 + "\n")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def send_startup_message(self):
        """Startup notification"""
        try:
            msg = "🤖 *COMPREHENSIVE TRADING BOT ACTIVATED!*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"📊 *Tracking:* {len(self.security_id_map)} Stocks/Indices\n"
            msg += f"⏱️ *Update Frequency:* Every 5 minutes\n\n"
            
            msg += "🎯 *ANALYSIS FEATURES:*\n"
            msg += "  ✅ Candlestick Chart (5min)\n"
            msg += "  ✅ Technical Indicators (SMA, RSI)\n"
            msg += "  ✅ Support/Resistance Levels\n"
            msg += "  ✅ Candlestick Patterns\n"
            msg += "  ✅ Volume Analysis\n"
            msg += "  ✅ Option Chain (OI, IV, PCR)\n"
            msg += "  ✅ AI Chart Pattern Recognition\n"
            msg += "  ✅ Trade Setup Recommendations\n"
            msg += "  ✅ Entry/Target/Stop Loss\n"
            msg += "  ✅ Risk Assessment\n\n"
            
            msg += "⚡ *POWERED BY:*\n"
            msg += "  • DhanHQ API v2\n"
            msg += "  • Google Gemini 1.5 Flash AI\n"
            msg += "  • Railway.app Hosting\n\n"
            
            msg += "📈 *MARKET HOURS:*\n"
            msg += "  Monday-Friday: 9:15 AM - 3:30 PM IST\n\n"
            
            msg += "🔔 *Bot Status:* ACTIVE ✅\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"Startup message error: {e}")


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    try:
        # Environment variables check
        required_vars = {
            'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
            'DHAN_CLIENT_ID': DHAN_CLIENT_ID,
            'DHAN_ACCESS_TOKEN': DHAN_ACCESS_TOKEN,
            'GEMINI_API_KEY': GEMINI_API_KEY
        }
        
        missing_vars = [k for k, v in required_vars.items() if not v]
        
        if missing_vars:
            logger.error("❌ MISSING ENVIRONMENT VARIABLES!")
            logger.error(f"Missing: {', '.join(missing_vars)}")
            logger.error("\nPlease set these in Railway.app:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            exit(1)
        
        logger.info("✅ All environment variables present")
        logger.info("🚀 Starting bot...")
        
        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
