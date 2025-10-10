import asyncio
import os
from telegram import Bot
import requests
from datetime import datetime, timedelta
import logging
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import google.generativeai as genai
from PIL import Image
import time

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
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# Stock/Index List
STOCKS_INDICES = {
    # Top Indices
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "SENSEX": {"symbol": "SENSEX", "segment": "IDX_I"},
    
    # High Volume Stocks
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "TCS": {"symbol": "TCS", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
    "TATAMOTORS": {"symbol": "TATAMOTORS", "segment": "NSE_EQ"},
    "AXISBANK": {"symbol": "AXISBANK", "segment": "NSE_EQ"},
    "BHARTIARTL": {"symbol": "BHARTIARTL", "segment": "NSE_EQ"},
}

# ========================
# GEMINI 2.5 FLASH ANALYZER
# ========================
class GeminiFlash25Analyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Gemini 2.5 Flash model use करतोय
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Free tier limits: 10 RPM (Requests Per Minute)
        self.max_rpm = 10
        self.request_times = []
        self.min_request_interval = 6.5  # 60/10 = 6 seconds + buffer
        
        logger.info("🤖 Gemini 2.5 Flash initialized (Free Tier: 10 RPM)")
    
    async def analyze_trade_setup(self, image_buffer, symbol, spot_price, 
                                  candles, technical_data, patterns, option_data):
        """
        Chart + Technical + Option Chain यांचं comprehensive analysis करतो
        आणि trade setup recommend करतो
        """
        try:
            # Rate limiting check
            await self._rate_limit_check()
            
            # Image prepare
            image_buffer.seek(0)
            image = Image.open(image_buffer)
            
            # Last 10 candles घेतो (recent price action)
            recent_candles = candles[-10:] if len(candles) >= 10 else candles
            candle_summary = self._format_candle_data(recent_candles)
            
            # Technical summary
            tech_summary = self._format_technical_data(technical_data, spot_price)
            
            # Pattern summary
            pattern_text = "\n".join(patterns) if patterns else "No major patterns detected"
            
            # Option chain summary
            option_summary = self._format_option_data(option_data, spot_price)
            
            # Comprehensive prompt for Gemini 2.5 Flash
            prompt = f"""
You are an EXPERT Indian stock market trader analyzing {symbol}.

📊 **CURRENT MARKET DATA:**
Spot Price: ₹{spot_price:,.2f}
Timestamp: {datetime.now().strftime('%d-%m-%Y %H:%M IST')}

📈 **TECHNICAL INDICATORS:**
{tech_summary}

🕯️ **CANDLESTICK PATTERNS:**
{pattern_text}

📉 **RECENT PRICE ACTION (Last 10 Candles):**
{candle_summary}

💹 **OPTION CHAIN ANALYSIS:**
{option_summary}

🎯 **CHART IMAGE:**
[Analyze the candlestick chart image for visual patterns, trend lines, support/resistance zones]

---

**YOUR TASK:**
Analyze ALL the data above (chart + technicals + patterns + options) and provide:

1️⃣ **MARKET SENTIMENT** (Bullish/Bearish/Neutral with confidence %)

2️⃣ **KEY OBSERVATIONS** (3-4 critical points from chart + data)

3️⃣ **TRADE RECOMMENDATION:**
   IF tradeable setup exists:
   ✅ **ACTION:** BUY/SELL
   💰 **ENTRY:** ₹[exact price]
   🎯 **TARGET 1:** ₹[price] ([%] profit)
   🎯 **TARGET 2:** ₹[price] ([%] profit)
   🛑 **STOP LOSS:** ₹[price] ([%] risk)
   📊 **RISK:REWARD:** [ratio like 1:2]
   ⏰ **TIMEFRAME:** Intraday/Swing (1-2 days)/Positional (>3 days)
   🔥 **CONFIDENCE:** [%]
   
   IF NO clear setup:
   ⏸️ **ACTION:** WAIT/AVOID - [reason]

4️⃣ **OPTION STRATEGY** (if applicable):
   [CE/PE to buy/sell with strike and reasoning]

5️⃣ **RISK FACTORS:** [What can go wrong]

**FORMATTING RULES:**
- Keep it CONCISE (under 30 lines)
- Use emojis for readability
- Give specific prices, not ranges
- Only recommend trade if confidence > 65%
- Focus on ACTIONABLE insights
- Mix Hindi-English for clarity if needed

Analyze NOW! 🚀
"""
            
            # Gemini API call
            response = self.model.generate_content([prompt, image])
            
            # Track request
            self.request_times.append(time.time())
            
            logger.info(f"✅ Gemini 2.5 Flash analysis done for {symbol}")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Gemini analysis error for {symbol}: {e}")
            return None
    
    async def _rate_limit_check(self):
        """
        Free tier ka 10 RPM limit manage करतो
        """
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at limit
        if len(self.request_times) >= self.max_rpm:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                logger.warning(f"⏳ Rate limit reached! Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time + 1)
                self.request_times = []
        
        # Always wait minimum interval between requests
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_request_interval:
                wait = self.min_request_interval - time_since_last
                logger.info(f"⏳ Throttling: waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
    
    def _format_candle_data(self, candles):
        """Recent candles format करतो"""
        lines = []
        for i, c in enumerate(candles[-5:], 1):  # Last 5 candles
            change = c['close'] - c['open']
            pct = (change / c['open']) * 100
            candle_type = "🟢 GREEN" if change > 0 else "🔴 RED"
            lines.append(
                f"{i}. {candle_type} | O:{c['open']:.2f} H:{c['high']:.2f} "
                f"L:{c['low']:.2f} C:{c['close']:.2f} ({pct:+.2f}%)"
            )
        return "\n".join(lines)
    
    def _format_technical_data(self, tech, price):
        """Technical indicators format करतो"""
        if not tech:
            return "Technical data unavailable"
        
        rsi_status = "Overbought" if tech.get('rsi', 50) > 70 else "Oversold" if tech.get('rsi', 50) < 30 else "Neutral"
        
        return f"""
Current: ₹{price:,.2f}
Trend: {tech.get('trend', 'N/A')}
SMA(20): ₹{tech.get('sma_20', 'N/A')} | SMA(50): ₹{tech.get('sma_50', 'N/A')}
RSI(14): {tech.get('rsi', 'N/A')} - {rsi_status}
Support: ₹{tech.get('support', 'N/A')} | Resistance: ₹{tech.get('resistance', 'N/A')}
Volume: {'HIGH SPIKE ⚡' if tech.get('volume_spike') else 'Normal'}
"""
    
    def _format_option_data(self, oc_data, spot):
        """Option chain data summarize करतो"""
        try:
            if not oc_data or 'oc' not in oc_data:
                return "Option data not available"
            
            oc = oc_data.get('oc', {})
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce = atm_data.get('ce', {})
            pe = atm_data.get('pe', {})
            
            ce_oi = ce.get('oi', 0)
            pe_oi = pe.get('oi', 0)
            pcr = round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0
            
            sentiment = "BULLISH 🟢" if pcr > 1.2 else "BEARISH 🔴" if pcr < 0.8 else "NEUTRAL 🟡"
            
            return f"""
ATM Strike: ₹{atm_strike:,.0f}
CE: OI={ce_oi/1000:.0f}K | LTP=₹{ce.get('last_price', 0):.1f} | IV={ce.get('implied_volatility', 0):.1f}%
PE: OI={pe_oi/1000:.0f}K | LTP=₹{pe.get('last_price', 0):.1f} | IV={pe.get('implied_volatility', 0):.1f}%
PCR Ratio: {pcr} → {sentiment}
"""
        except:
            return "Option summary error"


# ========================
# TECHNICAL ANALYZER (Same as before)
# ========================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(candles):
        try:
            if not candles or len(candles) < 20:
                return None
            
            closes = [c['close'] for c in candles[-50:]]
            highs = [c['high'] for c in candles[-50:]]
            lows = [c['low'] for c in candles[-50:]]
            volumes = [c['volume'] for c in candles[-50:]]
            
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            
            rsi = TechnicalAnalyzer._calculate_rsi(closes, 14)
            
            resistance = max(highs[-50:]) if len(highs) >= 50 else max(highs)
            support = min(lows[-50:]) if len(lows) >= 50 else min(lows)
            
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            volume_spike = current_volume > (avg_volume * 1.5)
            
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
                patterns.append("🟢 BULLISH ENGULFING")
        
        # Bearish Engulfing
        if prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['open'] > prev['close'] and last['close'] < prev['open']:
                patterns.append("🔴 BEARISH ENGULFING")
        
        return patterns


# ========================
# MAIN BOT CLASS
# ========================
class TradingBot:
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
        self.gemini_analyzer = GeminiFlash25Analyzer(GEMINI_API_KEY)
        self.tech_analyzer = TechnicalAnalyzer()
        logger.info("🤖 Trading Bot initialized with Gemini 2.5 Flash")
    
    async def load_security_ids(self):
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
        try:
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
                title=f'\n{symbol} | ₹{spot_price:,.2f} | 5min Chart',
                ylabel='Price (₹)',
                ylabel_lower='Vol',
                figsize=(14, 9),
                returnfig=True,
                tight_layout=True
            )
            
            axes[0].set_title(
                f'{symbol} | ₹{spot_price:,.2f} | 5min Chart',
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
    
    async def analyze_and_send(self, symbol):
        """Single stock analyze करतो आणि alerts पाठवतो"""
        try:
            if symbol not in self.security_id_map:
                logger.warning(f"⚠️ {symbol} - No security ID")
                return
            
            info = self.security_id_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 Analyzing {symbol}...")
            logger.info(f"{'='*50}")
            
            # 1. Expiry
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"{symbol}: No expiry")
                return
            
            # 2. Option Chain
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"{symbol}: No option chain")
                return
            
            spot_price = oc_data.get('last_price', 0)
            
            # 3. Candles
            candles = self.get_historical_data(security_id, segment, symbol)
            if not candles or len(candles) < 20:
                logger.warning(f"{symbol}: Insufficient candles")
                return
            
            # 4. Technical Analysis
            technical_data = self.tech_analyzer.calculate_indicators(candles)
            
            # 5. Patterns
            patterns = self.tech_analyzer.detect_candlestick_patterns(candles)
            
            # 6. Chart
            chart_buf = self.create_candlestick_chart(candles, symbol, spot_price)
            if not chart_buf:
                logger.warning(f"{symbol}: Chart failed")
                return
            
            # 7. 🤖 GEMINI 2.5 FLASH ANALYSIS
            logger.info(f"🤖 Running Gemini 2.5 Flash analysis...")
            
            ai_analysis = await self.gemini_analyzer.analyze_trade_setup(
                chart_buf,
                symbol,
                spot_price,
                candles,
                technical_data,
                patterns,
                oc_data
            )
            
            # 8. Send Chart
            chart_buf.seek(0)
            await self.bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=chart_buf,
                caption=f"📊 *{symbol}* Chart\n💰 Spot: ₹{spot_price:,.2f}",
                parse_mode='Markdown'
            )
            
            # 9. Send AI Analysis (if available)
            if ai_analysis:
                # Split karnar jar message lamba asel
                if len(ai_analysis) > 4000:
                    parts = [ai_analysis[i:i+4000] for i in range(0, len(ai_analysis), 4000)]
                    for part in parts:
                        await self.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=f"🤖 *AI ANALYSIS - {symbol}*\n\n{part}",
                            parse_mode='Markdown'
                        )
                        await asyncio.sleep(1)
                else:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"🤖 *AI TRADE ANALYSIS - {symbol}*\n{'='*40}\n\n{ai_analysis}",
                        parse_mode='Markdown'
                    )
            else:
                logger.warning(f"⚠️ AI analysis unavailable for {symbol}")
            
            logger.info(f"✅ {symbol} analysis complete!")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Trading Bot with Gemini 2.5 Flash...")
        
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        
        logger.info(f"📊 Total symbols: {len(all_symbols)}")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 NEW SCAN CYCLE: {timestamp}")
                logger.info(f"{'='*60}\n")
                
                for idx, symbol in enumerate(all_symbols, 1):
                    logger.info(f"📊 [{idx}/{len(all_symbols)}] Processing {symbol}...")
                    
                    await self.analyze_and_send(symbol)
                    
                    # Inter-symbol delay (Gemini rate limit + Dhan API)
                    if idx < len(all_symbols):
                        logger.info(f"⏳ Waiting 8 seconds before next symbol...")
                        await asyncio.sleep(8)
                
                logger.info("\n" + "="*60)
                logger.info("✅ SCAN CYCLE COMPLETED!")
                logger.info("⏳ Next cycle in 10 minutes...")
                logger.info("="*60 + "\n")
                
                await asyncio.sleep(600)  # 10 minutes
                
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
            msg = "🤖 *TRADING BOT ACTIVATED!*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            msg += f"📊 *Tracking:* {len(self.security_id_map)} Stocks/Indices\n"
            msg += f"⏱️ *Scan Frequency:* Every 10 minutes\n"
            msg += f"📈 *Timeframe:* 5-minute candles\n\n"
            
            msg += "🎯 *FEATURES:*\n"
            msg += "  ✅ Live Candlestick Charts\n"
            msg += "  ✅ Technical Indicators (SMA, RSI)\n"
            msg += "  ✅ Support/Resistance Levels\n"
            msg += "  ✅ Candlestick Pattern Detection\n"
            msg += "  ✅ Volume Analysis\n"
            msg += "  ✅ Option Chain (PCR, OI, IV)\n"
            msg += "  ✅ AI Chart Pattern Recognition\n"
            msg += "  ✅ Trade Setup Recommendations\n"
            msg += "  ✅ Entry/Target/Stop Loss Levels\n"
            msg += "  ✅ Risk:Reward Calculation\n"
            msg += "  ✅ Option Strategy Suggestions\n\n"
            
            msg += "⚡ *POWERED BY:*\n"
            msg += "  • Google Gemini 2.5 Flash AI\n"
            msg += "  • DhanHQ API v2\n"
            msg += "  • Railway.app Cloud Hosting\n\n"
            
            msg += "📋 *SYMBOLS TRACKED:*\n"
            symbols_list = ", ".join(list(self.security_id_map.keys()))
            msg += f"  {symbols_list}\n\n"
            
            msg += "⚙️ *RATE LIMITS:*\n"
            msg += "  • Gemini: 10 requests/min (Free Tier)\n"
            msg += "  • Auto-throttling enabled ✅\n\n"
            
            msg += "📈 *MARKET HOURS:*\n"
            msg += "  Mon-Fri: 9:15 AM - 3:30 PM IST\n\n"
            
            msg += "🔔 *Status:* ACTIVE ✅\n"
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
            logger.error("\n⚙️ Please set these in Railway.app:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            exit(1)
        
        logger.info("✅ All environment variables present")
        logger.info("🚀 Initializing Trading Bot...")
        logger.info("🤖 Using Gemini 2.5 Flash (Free Tier: 10 RPM)")
        
        bot = TradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
