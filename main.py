import asyncio
import os
import sys
import requests
import logging
import csv
import io
from datetime import datetime, timedelta
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# Third-party libraries
from telegram import Bot, InputFile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import google.generativeai as genai

# Force stdout flush for deployment logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- CONFIGURATION ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        logger.warning("GEMINI_API_KEY not found. AI analysis will be disabled.")
except Exception as e:
    logger.critical(f"Failed to configure AI clients: {e}")
    GEMINI_API_KEY = None

DHAN_API_BASE = "https://api.dhan.co"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v1/optionchain"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v1/charts/intraday"

STOCKS_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "BAJFINANCE", "INFY", 
    "TATAMOTORS", "AXISBANK", "SBIN", "ADANIENT", "KOTAKBANK",
    "LT", "MARUTI", "NTPC", "BHARTIARTL", "POWERGRID", "M&M", "WIPRO"
]
# Note: Indices like NIFTY/SENSEX use a different security ID logic and are excluded for stability.

# --- GEMINI AI ANALYZER ---
class GeminiAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash') if GEMINI_API_KEY else None

    def calculate_technical_indicators(self, candles_df):
        try:
            df = candles_df.copy()
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            latest = df.iloc[-1]
            return {'rsi': latest['RSI'], 'macd': latest['MACD'], 'macd_signal': latest['MACD_Signal']}
        except Exception:
            return {}

    def format_data_for_ai(self, symbol, oc_data, candles_df, indicators):
        spot_price = oc_data.get('spotPrice', 0)
        atm_strike = min([d['strikePrice'] for d in oc_data['optionChainDetails']], key=lambda x: abs(x - spot_price))
        
        # Format Option Chain
        oc_text = "STRIKE | CE OI (Lakhs) | PE OI (Lakhs)\n"
        for detail in sorted(oc_data['optionChainDetails'], key=lambda x: x['strikePrice']):
            if abs(detail['strikePrice'] - atm_strike) <= (atm_strike * 0.05):
                oc_text += f"{detail['strikePrice']:<6.0f} | {detail.get('ce_openInterest', 0) / 100000:<15.2f} | {detail.get('pe_openInterest', 0) / 100000:<15.2f}\n"

        # Format Candle Data
        candle_text = "Time | Open | High | Low | Close | Volume\n"
        for _, row in candles_df.tail(20).iterrows():
            candle_text += f"{row.name.strftime('%H:%M')} | {row.Open:.2f} | {row.High:.2f} | {row.Low:.2f} | {row.Close:.2f} | {row.Volume:,}\n"

        # Build Final Prompt
        return f"""You are an expert F&O technical analyst. Analyze all data for {symbol} and create a detailed trade setup in Marathi.

## 1. Summary
- **Symbol:** {symbol} | **Spot Price:** ₹{spot_price:,.2f} | **ATM:** ₹{atm_strike:,.0f}
- **RSI(14):** {indicators.get('rsi', 0):.2f} | **MACD:** {indicators.get('macd', 0):.2f}

## 2. Option Chain (OI in Lakhs)
{oc_text}

## 3. Price Action
{candle_text}

---
## Analysis & Trade Alert (Marathi मध्ये):
1.  **Technical Analysis:** Chart image आणि price action नुसार, trend काय आहे (Uptrend/Downtrend/Sideways)? Key support/resistance levels ओळखा. कोणता chart pattern (e.g., breakout, flag) दिसतोय का?
2.  **Option Chain Analysis:** Option chain नुसार, सर्वाधिक Call OI (resistance) आणि Put OI (support) कुठे आहे? एकूण sentiment (Bullish/Bearish) काय आहे?
3.  **🚨 Final Verdict & Trade Alert:**
    - **Trade Setup:** (Yes/No)?
    - **Action:** (BUY CE / BUY PE)?
    - **Strike:** ₹[STRIKE]
    - **Entry:** ₹[ENTRY]
    - **Target:** ₹[TARGET]
    - **Stop Loss:** ₹[SL]
    - **Reasoning:** (2-3 ओळीत ट्रेड का घ्यावा).
    - **No Trade:** जर trade नसेल, तर "No clear setup, wait and watch" असे लिहा.
"""

    async def analyze(self, symbol, chart_buf, oc_data, candles_df):
        if not self.model: return "AI analysis is disabled."
        try:
            logger.info(f"🤖 Running Gemini Flash analysis for {symbol}...")
            indicators = self.calculate_technical_indicators(candles_df)
            prompt = self.format_data_for_ai(symbol, oc_data, candles_df, indicators)
            
            chart_buf.seek(0)
            image_part = {"mime_type": "image/png", "data": chart_buf.read()}
            
            response = await self.model.generate_content_async([prompt, image_part], generation_config={"temperature": 0.3})
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemini analysis for {symbol}: {e}")
            return "AI analysis failed."

# --- DHAN TRADING BOT ---
class DhanTradingBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.headers = {'access-token': DHAN_ACCESS_TOKEN}
        self.security_id_map = {}
        self.gemini_analyzer = GeminiAnalyzer()
        logger.info("🚀 AI Trading Bot Initialized")

    async def load_security_ids(self):
        try:
            logger.info("Downloading Dhan instruments file...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            response.raise_for_status()
            
            all_rows = list(csv.DictReader(io.StringIO(response.text)))
            today = datetime.now()

            for symbol in STOCKS_WATCHLIST:
                futures = []
                for row in all_rows:
                    try:
                        if (row.get('SEM_TRADING_SYMBOL') == symbol and 
                            row.get('SEM_INSTRUMENT_TYPE') == 'FUTSTK' and
                            row.get('SEM_EXM_EXCH_ID') == 'NFO'):
                            
                            expiry_date = datetime.strptime(row.get('SEM_EXPIRY_DATE', '').split(' ')[0], '%Y-%m-%d')
                            if expiry_date > today:
                                futures.append({'expiry': expiry_date, 'fno_id': int(row.get('SEM_SMST_SECURITY_ID')), 'equity_id': int(row.get('SEM_UNDERLYING_SECURITY_ID'))})
                    except (ValueError, TypeError): continue
                
                if futures:
                    nearest = min(futures, key=lambda x: x['expiry'])
                    self.security_id_map[symbol] = nearest
                    logger.info(f"✅ {symbol}: Loaded F&O Security ID = {nearest['fno_id']}")
            
            logger.info(f"Total {len(self.security_id_map)} F&O securities loaded.")
            return True
        except Exception as e:
            logger.error(f"CRITICAL: Error loading security IDs: {e}", exc_info=True)
            return False

    def get_api_data(self, url, payload, method='POST'):
        try:
            if method == 'POST':
                response = requests.post(url, json=payload, headers=self.headers, timeout=15)
            else: # GET
                response = requests.get(url, params=payload, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('data')
        except Exception as e:
            logger.error(f"API call to {url} failed: {e}")
            return None

    def create_chart(self, df, symbol, spot_price):
        try:
            df_chart = df.copy(); df_chart.set_index('Date', inplace=True)
            df_chart['SMA20'] = df_chart['Close'].rolling(window=20).mean()
            df_chart['SMA50'] = df_chart['Close'].rolling(window=50).mean()
            sma20 = df_chart['SMA20']; std20 = df_chart['Close'].rolling(window=20).std()
            df_chart['BB_Upper'] = sma20 + (std20 * 2)
            df_chart['BB_Lower'] = sma20 - (std20 * 2)

            mc = mpf.make_marketcolors(up='#00ff88', down='#ff4444', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', rc={'font.size': 10})
            apds = [
                mpf.make_addplot(df_chart['SMA20'], color='#ffa500', width=1.2),
                mpf.make_addplot(df_chart['SMA50'], color='#00bfff', width=1.2),
                mpf.make_addplot(df_chart['BB_Upper'], color='#9370db', width=0.8, linestyle='--'),
                mpf.make_addplot(df_chart['BB_Lower'], color='#9370db', width=0.8, linestyle='--'),
            ]
            fig, axes = mpf.plot(df_chart.tail(100), type='candle', style=s, volume=True, addplot=apds,
                                 title=f'\n{symbol} | Spot: ₹{spot_price:,.2f}', figsize=(15, 8), returnfig=True)
            axes[0].legend(['SMA20', 'SMA50', 'BB Upper', 'BB Lower'])
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); buf.seek(0)
            return buf
        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return None

    async def process_stock(self, symbol):
        try:
            if symbol not in self.security_id_map: return
            info = self.security_id_map[symbol]
            logger.info(f"--- Processing {symbol} ---")

            oc_data = self.get_api_data(DHAN_OPTION_CHAIN_URL, {'securityId': str(info['fno_id']), 'exchangeSegment': 'NSE_FNO'})
            if not oc_data: return

            to_date, from_date = datetime.now(), datetime.now() - timedelta(days=7)
            hist_payload = {"securityId": str(info['equity_id']), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "fromDate": from_date.strftime("%Y-%m-%d"), "toDate": to_date.strftime("%Y-%m-%d"), "interval": "FIVE_MINUTE"}
            hist_data = self.get_api_data(DHAN_INTRADAY_URL, hist_payload)
            if not hist_data: return

            candles_df = pd.DataFrame({'Date': pd.to_datetime(hist_data['start_Time'], unit='s'), 'Open': hist_data['open'], 'High': hist_data['high'], 'Low': hist_data['low'], 'Close': hist_data['close'], 'Volume': hist_data['volume']})
            if len(candles_df) < 50: return # Need enough data for indicators
            
            spot_price = oc_data.get('spotPrice', 0)
            chart_buf = self.create_chart(candles_df, symbol, spot_price)
            if not chart_buf: return

            ai_analysis_text = await self.gemini_analyzer.analyze(symbol, chart_buf, oc_data, candles_df)

            caption = f"🤖 ***AI Analysis for {symbol}***\n\n{ai_analysis_text}"
            chart_buf.seek(0)
            await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=InputFile(chart_buf, filename=f"{symbol}.png"), caption=caption, parse_mode='Markdown')
            logger.info(f"✅ Analysis sent for {symbol}")

        except Exception as e:
            logger.error(f"FATAL error processing {symbol}: {e}", exc_info=True)

# --- HTTP SERVER & MAIN EXECUTION ---
class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-type', 'text/plain'); self.end_headers()
        self.wfile.write(b"Bot is alive!")

def run_server():
    port = int(os.environ.get("PORT", 8080))
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, KeepAliveHandler)
        logger.info(f"Starting keep-alive server on port {port}...")
        httpd.serve_forever()
    except Exception: pass

async def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
        logger.critical("❌ Missing critical environment variables. Exiting.")
        return
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    bot = DhanTradingBot()

    if await bot.load_security_ids():
        await bot.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"✅ **AI Trading Bot ONLINE**\nTracking {len(bot.security_id_map)} F&O stocks.", parse_mode='Markdown')
        while True:
            logger.info("============== NEW SCAN CYCLE ==============")
            for symbol in bot.security_id_map.keys():
                await bot.process_stock(symbol)
                logger.info("--- Waiting 3 seconds before next stock ---")
                await asyncio.sleep(3) # Rate limit
            
            logger.info("Scan cycle complete. Waiting for 10 minutes...")
            await asyncio.sleep(600)
    else:
        await bot.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="❌ **Bot failed to start.** Could not load F&O security IDs.", parse_mode='Markdown')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
