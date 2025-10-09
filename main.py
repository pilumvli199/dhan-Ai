# Simplified 3-Layer AI Trading Bot
# Works WITHOUT Option Chain data - Only Chart Analysis
# Dependencies: python-telegram-bot, requests, matplotlib, mplfinance, pandas, numpy, pytz
"""

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
import base64
import json

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TEST MODE: Set to True for testing outside market hours
TEST_MODE = os.getenv("TEST_MODE", "False").upper() == "TRUE"

# Dhan URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"

# AI URLs
GEMINI_FLASH_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_PRO_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
OPENAI_GPT4_URL = "https://api.openai.com/v1/chat/completions"

# Stocks to track (High liquidity)
STOCKS = {
    "NIFTY 50": {"symbol": "NIFTY 50", "segment": "IDX_I"},
    "NIFTY BANK": {"symbol": "NIFTY BANK", "segment": "IDX_I"},
    "RELIANCE": {"symbol": "RELIANCE", "segment": "NSE_EQ"},
    "HDFCBANK": {"symbol": "HDFCBANK", "segment": "NSE_EQ"},
    "ICICIBANK": {"symbol": "ICICIBANK", "segment": "NSE_EQ"},
    "INFY": {"symbol": "INFY", "segment": "NSE_EQ"},
    "SBIN": {"symbol": "SBIN", "segment": "NSE_EQ"},
}

class SimpleChartAIBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.running = True
        self.headers = {
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.security_map = {}
        self.trade_signals = []
        self.test_mode = TEST_MODE
        logger.info(f"🤖 Simple Chart AI Bot initialized (TEST_MODE={self.test_mode})")
    
    def is_market_open(self):
        """Check if market is open (IST)"""
        from datetime import datetime, time
        import pytz
        
        # Get IST time
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Check weekend
        if now.weekday() >= 5:  # Sat=5, Sun=6
            return False
        
        # Market hours: 9:15 AM - 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)
        current_time = now.time()
        
        is_open = market_open <= current_time <= market_close
        
        if not is_open:
            logger.warning(f"⚠️ Market closed (Current time: {now.strftime('%H:%M IST')})")
        
        return is_open
    
    def generate_mock_data(self, symbol, base_price=1000):
        """Generate mock data for testing"""
        import random
        
        logger.info(f"🧪 Generating mock data for {symbol}")
        
        # Generate 100 candles
        candles = []
        price = base_price
        
        for i in range(100):
            open_price = price
            change = random.uniform(-2, 2)  # ±2% change
            close_price = price + (price * change / 100)
            high_price = max(open_price, close_price) * random.uniform(1.001, 1.01)
            low_price = min(open_price, close_price) * random.uniform(0.99, 0.999)
            volume = random.randint(100000, 1000000)
            
            timestamp = (datetime.now() - timedelta(minutes=(100-i)*5)).strftime("%Y-%m-%d %H:%M:%S")
            
            candles.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            price = close_price
        
        spot_price = candles[-1]['close']
        
        return candles, spot_price
    
    async def load_securities(self):
        """Load security IDs"""
        try:
            logger.info("📥 Loading security IDs...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            
            if response.status_code != 200:
                return False
            
            csv_data = response.text.split('\n')
            reader = csv.DictReader(csv_data)
            
            for symbol, info in STOCKS.items():
                segment = info['segment']
                symbol_name = info['symbol']
                
                for row in reader:
                    try:
                        if segment == "IDX_I":
                            if (row.get('SEM_SEGMENT') == 'I' and 
                                row.get('SEM_TRADING_SYMBOL') == symbol_name):
                                sec_id = row.get('SEM_SMST_SECURITY_ID')
                                if sec_id:
                                    self.security_map[symbol] = {
                                        'security_id': int(sec_id),
                                        'segment': segment,
                                        'trading_symbol': symbol_name
                                    }
                                    logger.info(f"✅ {symbol}: ID={sec_id}")
                                    break
                        else:
                            if (row.get('SEM_SEGMENT') == 'E' and 
                                row.get('SEM_TRADING_SYMBOL') == symbol_name and
                                row.get('SEM_EXM_EXCH_ID') == 'NSE'):
                                sec_id = row.get('SEM_SMST_SECURITY_ID')
                                if sec_id:
                                    self.security_map[symbol] = {
                                        'security_id': int(sec_id),
                                        'segment': segment,
                                        'trading_symbol': symbol_name
                                    }
                                    logger.info(f"✅ {symbol}: ID={sec_id}")
                                    break
                    except:
                        continue
                
                csv_data = response.text.split('\n')
                reader = csv.DictReader(csv_data)
            
            logger.info(f"✅ Loaded {len(self.security_map)} securities")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading securities: {e}")
            return False
    
    def get_spot_price(self, security_id, segment):
        """Get current spot price with detailed logging"""
        try:
            exch_seg = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exch_seg
            }
            
            logger.info(f"OHLC API Request: {payload}")
            
            response = requests.post(
                DHAN_OHLC_URL,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            logger.info(f"OHLC Response: Status={response.status_code}")
            logger.info(f"OHLC Body: {response.text[:300]}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"OHLC Data: {data}")
                
                if data.get('data'):
                    ltp = data['data'].get('last_price', 0)
                    if ltp > 0:
                        logger.info(f"✅ Spot Price: {ltp}")
                        return ltp
                    else:
                        logger.warning(f"⚠️ LTP is 0 or missing")
                else:
                    logger.warning(f"⚠️ No 'data' key in response")
            else:
                logger.error(f"❌ OHLC API failed: {response.status_code}")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error getting spot price: {e}")
            return 0
    
    def get_intraday_candles(self, security_id, segment):
        """Get 5-min candles"""
        try:
            exch_seg = "IDX_I" if segment == "IDX_I" else "NSE_EQ"
            instrument = "INDEX" if segment == "IDX_I" else "EQUITY"
            
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
                    candles = []
                    for i in range(len(data['open'])):
                        candles.append({
                            'timestamp': data['start_Time'][i] if i < len(data['start_Time']) else '',
                            'open': data['open'][i],
                            'high': data['high'][i],
                            'low': data['low'][i],
                            'close': data['close'][i],
                            'volume': data['volume'][i]
                        })
                    
                    logger.info(f"📊 Got {len(candles)} candles")
                    return candles
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting candles: {e}")
            return None
    
    def create_chart(self, candles, symbol, spot_price):
        """Create chart"""
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
                return None, None
            
            mc = mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#333333',
                facecolor='#1e1e1e',
                figcolor='#1e1e1e'
            )
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} - Spot: ₹{spot_price:,.2f}',
                figsize=(12, 8),
                returnfig=True
            )
            
            # For Telegram
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf.seek(0)
            
            # For AI (base64)
            buf_copy = io.BytesIO()
            fig.savefig(buf_copy, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf_copy.seek(0)
            base64_img = base64.b64encode(buf_copy.read()).decode('utf-8')
            
            plt.close(fig)
            
            return buf, base64_img
            
        except Exception as e:
            logger.error(f"❌ Chart error: {e}")
            return None, None
    
    def prepare_candle_text(self, candles):
        """Convert candles to text"""
        text = "Timestamp | Open | High | Low | Close | Volume\n"
        text += "-" * 70 + "\n"
        
        for candle in candles[-30:]:  # Last 30
            time = candle['timestamp'][-8:] if candle['timestamp'] else ''
            text += f"{time} | {candle['open']:.2f} | {candle['high']:.2f} | {candle['low']:.2f} | {candle['close']:.2f} | {candle['volume']}\n"
        
        return text
    
    async def gemini_flash_scan(self, symbol, candles, spot_price):
        """Layer 1: Quick scan"""
        try:
            logger.info(f"⚡ Gemini Flash scanning {symbol}...")
            
            candle_text = self.prepare_candle_text(candles)
            
            prompt = f"""Analyze {symbol} chart for F&O trading opportunity.

**Spot Price:** ₹{spot_price:,.2f}

**Last 30 Candles (5-min):**
{candle_text}

**Task:** Quick scan - Is there a trading opportunity?
Look for: Trend, momentum, volume surge, breakout/breakdown

Response Format: YES/NO - [1 line reason]

Example: "YES - Strong bullish breakout with volume surge"
"""
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 100
                }
            }
            
            response = requests.post(
                f"{GEMINI_FLASH_URL}?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"⚡ Flash: {ai_response}")
                
                approved = ai_response.upper().startswith('YES')
                return {'approved': approved, 'response': ai_response}
            
            return {'approved': False, 'response': 'API Error'}
            
        except Exception as e:
            logger.error(f"❌ Flash error: {e}")
            return {'approved': False, 'response': str(e)}
    
    async def gemini_pro_strategy(self, symbol, candles, spot_price, flash_result):
        """Layer 2: Strategy"""
        try:
            logger.info(f"🎯 Gemini Pro analyzing {symbol}...")
            
            candle_text = self.prepare_candle_text(candles[-50:])
            
            prompt = f"""Deep analysis for {symbol} F&O trading.

**Spot:** ₹{spot_price:,.2f}

**Chart Data (50 candles):**
{candle_text}

**Flash AI:** {flash_result['response']}

**Your Task:**
1. Confirm trade viability (YES/NO)
2. Suggest strategy:
   - Direction: BULLISH/BEARISH
   - Entry zone
   - Target & Stop loss levels
   - Risk assessment

Format:
YES/NO
Direction: [BULLISH/BEARISH]
Entry: ₹[price]
Target: ₹[price]
Stop Loss: ₹[price]
Risk: [comment]
"""
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 300
                }
            }
            
            response = requests.post(
                f"{GEMINI_PRO_URL}?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"🎯 Pro: {ai_response[:200]}...")
                
                approved = ai_response.upper().startswith('YES')
                return {'approved': approved, 'response': ai_response}
            
            return {'approved': False, 'response': 'API Error'}
            
        except Exception as e:
            logger.error(f"❌ Pro error: {e}")
            return {'approved': False, 'response': str(e)}
    
    async def gpt4o_final_call(self, symbol, candles, chart_base64, spot_price, flash_result, pro_result):
        """Layer 3: Final call with vision"""
        try:
            logger.info(f"🤖 GPT-4o final call for {symbol}...")
            
            candle_text = self.prepare_candle_text(candles[-50:])
            
            prompt = f"""FINAL TRADING DECISION for {symbol}.

**Spot:** ₹{spot_price:,.2f}

**Chart (last 50 candles):**
{candle_text}

**AI Layer 1:** {flash_result['response']}

**AI Layer 2:** {pro_result['response']}

**YOUR DECISION:**
Analyze chart image + data. Give FINAL TRADE SIGNAL.

Format:
TRADE: YES/NO
DIRECTION: BULLISH/BEARISH/NEUTRAL
ENTRY: ₹[price]
TARGET: ₹[price] (+[%])
STOP_LOSS: ₹[price] (-[%])
RISK_REWARD: [ratio]
CONFIDENCE: [%]
REASON: [2-3 lines]

If NO: TRADE: NO - [reason]
"""
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{chart_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            response = requests.post(
                OPENAI_GPT4_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                logger.info(f"🤖 GPT-4o: {ai_response[:200]}...")
                
                # Parse response
                trade_signal = self.parse_signal(ai_response, symbol, spot_price)
                return trade_signal
            
            return {'trade': False, 'reason': 'API Error'}
            
        except Exception as e:
            logger.error(f"❌ GPT-4o error: {e}")
            return {'trade': False, 'reason': str(e)}
    
    def parse_signal(self, text, symbol, spot):
        """Parse AI response"""
        lines = text.strip().split('\n')
        signal = {
            'symbol': symbol,
            'spot_price': spot,
            'trade': False
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('TRADE:'):
                signal['trade'] = 'YES' in line.upper()
            elif line.startswith('DIRECTION:'):
                signal['direction'] = line.split(':', 1)[1].strip()
            elif line.startswith('ENTRY:'):
                signal['entry'] = line.split(':', 1)[1].strip()
            elif line.startswith('TARGET:'):
                signal['target'] = line.split(':', 1)[1].strip()
            elif line.startswith('STOP_LOSS:'):
                signal['stop_loss'] = line.split(':', 1)[1].strip()
            elif line.startswith('RISK_REWARD:'):
                signal['risk_reward'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                signal['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('REASON:'):
                signal['reason'] = line.split(':', 1)[1].strip()
        
        return signal
    
    async def analyze_stock(self, symbol):
        """Complete analysis with test mode support"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 ANALYZING: {symbol}")
            logger.info(f"{'='*60}")
            
            if symbol not in self.security_map:
                return None
            
            info = self.security_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            # Check if test mode or market closed
            if self.test_mode or not self.is_market_open():
                logger.info(f"🧪 Using MOCK DATA for {symbol}")
                
                # Mock data with realistic prices
                mock_prices = {
                    "NIFTY 50": 24500,
                    "NIFTY BANK": 51000,
                    "RELIANCE": 2850,
                    "HDFCBANK": 1650,
                    "ICICIBANK": 1250,
                    "INFY": 1850,
                    "SBIN": 820
                }
                
                base_price = mock_prices.get(symbol, 1000)
                candles, spot_price = self.generate_mock_data(symbol, base_price)
                
            else:
                # Real market data
                spot_price = self.get_spot_price(security_id, segment)
                if spot_price == 0:
                    logger.warning(f"⚠️ {symbol}: No spot price")
                    return None
                
                candles = self.get_intraday_candles(security_id, segment)
                if not candles or len(candles) < 50:
                    logger.warning(f"⚠️ {symbol}: Insufficient candles")
                    return None
            
            # Create chart
            chart_buf, chart_base64 = self.create_chart(candles, symbol, spot_price)
            if not chart_base64:
                logger.warning(f"⚠️ {symbol}: Chart failed")
                return None
            
            # Layer 1: Flash
            await asyncio.sleep(1)
            flash_result = await self.gemini_flash_scan(symbol, candles, spot_price)
            
            if not flash_result['approved']:
                logger.info(f"❌ {symbol}: Flash rejected - {flash_result['response']}")
                return None
            
            # Layer 2: Pro
            await asyncio.sleep(1)
            pro_result = await self.gemini_pro_strategy(symbol, candles, spot_price, flash_result)
            
            if not pro_result['approved']:
                logger.info(f"❌ {symbol}: Pro rejected")
                return None
            
            # Layer 3: GPT-4o
            await asyncio.sleep(1)
            trade_signal = await self.gpt4o_final_call(symbol, candles, chart_base64, spot_price, flash_result, pro_result)
            
            if not trade_signal.get('trade'):
                logger.info(f"❌ {symbol}: GPT-4o rejected")
                return None
            
            logger.info(f"🎉 {symbol}: TRADE SIGNAL!")
            
            return {
                'symbol': symbol,
                'spot_price': spot_price,
                'chart_buf': chart_buf,
                'flash_result': flash_result,
                'pro_result': pro_result,
                'trade_signal': trade_signal,
                'timestamp': datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                'test_mode': self.test_mode or not self.is_market_open()
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis error for {symbol}: {e}")
            return None
    
    async def send_trade_signal(self, result):
        """Send to Telegram"""
        try:
            symbol = result['symbol']
            signal = result['trade_signal']
            is_test = result.get('test_mode', False)
            
            # Chart
            if result.get('chart_buf'):
                caption = f"📊 {symbol} Technical Chart"
                if is_test:
                    caption += " [TEST MODE - Mock Data]"
                
                await self.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=result['chart_buf'],
                    caption=caption
                )
                await asyncio.sleep(1)
            
            # Signal message
            msg = f"🎯 *TRADE SIGNAL: {symbol}*\n"
            if is_test:
                msg += "🧪 *[TEST MODE - Mock Data]*\n"
            msg += "\n"
            msg += f"💰 Spot: ₹{result['spot_price']:,.2f}\n"
            msg += f"🎲 Direction: {signal.get('direction', 'N/A')}\n\n"
            msg += f"📈 *ENTRY:* {signal.get('entry', 'N/A')}\n"
            msg += f"🎯 *TARGET:* {signal.get('target', 'N/A')}\n"
            msg += f"🛑 *STOP LOSS:* {signal.get('stop_loss', 'N/A')}\n"
            msg += f"📊 *R:R:* {signal.get('risk_reward', 'N/A')}\n"
            msg += f"🎲 *CONFIDENCE:* {signal.get('confidence', 'N/A')}\n\n"
            msg += f"💡 *REASON:*\n{signal.get('reason', 'N/A')}\n\n"
            msg += f"┌─── AI LAYERS ───┐\n"
            msg += f"│ ⚡ Flash: ✅\n"
            msg += f"│ 🎯 Pro: ✅\n"
            msg += f"│ 🤖 GPT-4o: ✅\n"
            msg += f"└─────────────────┘\n\n"
            msg += f"🕐 {result['timestamp']}"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Signal sent for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Send error: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("🚀 Simple Chart AI Bot starting...")
        
        if not await self.load_securities():
            logger.error("❌ Failed to load securities")
            return
        
        # Startup message
        msg = "🤖 *Chart AI Bot Started!*\n\n"
        msg += f"📊 Tracking: {len(self.security_map)} stocks\n"
        msg += "⚡ AI: Gemini + GPT-4o\n"
        msg += "📈 Chart-only analysis\n"
        msg += "⏱️ Scan: Every 5 min"
        
        await self.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode='Markdown'
        )
        
        all_symbols = list(self.security_map.keys())
        
        while self.running:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 SCAN CYCLE: {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"{'='*70}\n")
                
                scanned = 0
                approved = 0
                rejected = 0
                
                for idx, symbol in enumerate(all_symbols, 1):
                    logger.info(f"\n[{idx}/{len(all_symbols)}] {symbol}")
                    
                    result = await self.analyze_stock(symbol)
                    scanned += 1
                    
                    if result:
                        approved += 1
                        await self.send_trade_signal(result)
                        self.trade_signals.append(result)
                    else:
                        rejected += 1
                    
                    # 5 sec delay between stocks (Gemini rate limit)
                    if idx < len(all_symbols):
                        logger.info("⏳ 5 sec wait...")
                        await asyncio.sleep(5)
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ Scan done: {scanned} | ✅ {approved} | ❌ {rejected}")
                logger.info(f"{'='*70}\n")
                
                # Summary message
                summary = f"📊 *Scan Completed*\n\n"
                summary += f"Scanned: {scanned}\n"
                summary += f"Signals: {approved}\n"
                summary += f"Rejected: {rejected}\n\n"
                summary += "⏳ Next scan in 5 minutes"
                
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=summary,
                    parse_mode='Markdown'
                )
                
                # Wait 5 minutes
                logger.info("⏸️ Waiting 5 minutes...\n")
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        # Check env vars
        required = [
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
            DHAN_CLIENT_ID,
            DHAN_ACCESS_TOKEN,
            GEMINI_API_KEY,
            OPENAI_API_KEY
        ]
        
        if not all(required):
            logger.error("❌ Missing environment variables!")
            logger.error("Required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY")
            exit(1)
        
        bot = SimpleChartAIBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        exit(1)
