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
import pytz

# Third-party libraries
from telegram import Bot, InputFile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import google.generativeai as genai

# --- CONFIGURATION ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Timezone to India Standard Time
IST = pytz.timezone('Asia/Kolkata')

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
    "TATAMOTORS", "AXISBANK", "SBIN", "ADANIENT", "KOTAKBANK"
]

# --- GEMINI AI ANALYZER ---
class GeminiPriceActionAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash') if GEMINI_API_KEY else None

    def format_data_for_ai(self, symbol, oc_data, candles_df):
        spot_price = oc_data.get('spotPrice', 0)
        
        oc_text = "STRIKE | CE OI (Lakhs) | PE OI (Lakhs)\n"
        all_strikes = sorted(oc_data['optionChainDetails'], key=lambda x: x['strikePrice'])
        max_ce_oi = max(all_strikes, key=lambda x: x.get('ce_openInterest', 0))
        max_pe_oi = max(all_strikes, key=lambda x: x.get('pe_openInterest', 0))

        oc_text += f"Max CE OI at: {max_ce_oi['strikePrice']} ({max_ce_oi.get('ce_openInterest',0)/100000:.2f}L) -> Resistance\n"
        oc_text += f"Max PE OI at: {max_pe_oi['strikePrice']} ({max_pe_oi.get('pe_openInterest',0)/100000:.2f}L) -> Support\n"
        
        candle_text = "Time | Open | High | Low | Close | Volume\n"
        for _, row in candles_df.tail(30).iterrows():
            candle_text += f"{row.name.strftime('%H:%M')} | {row.Open:.2f} | {row.High:.2f} | {row.Low:.2f} | {row.Close:.2f} | {row.Volume:,}\n"

        return f"""You are an expert Price Action trader. Your goal is to analyze the provided chart image, option chain data, and raw candle data to find a high-probability trade. Do not use complex indicators. Focus only on what you see.

## Market Data for {symbol}
- **Spot Price:** ₹{spot_price:,.2f}
- **Option Chain Key Levels:**
{oc_text}
- **Recent Price Action (Raw Data):**
{candle_text}

---
## Analysis & Trade Alert (in simple Marathi):

**1. Price Action & Chart Pattern Analysis:**
   - Chart image आणि raw data बघून सध्याचा ट्रेंड काय आहे (Uptrend/Downtrend/Sideways)?
   - महत्त्वाचे Support आणि Resistance levels कोणते आहेत?
   - कोणता Chart Pattern दिसत आहे का? (उदा. Breakout, Triangle, Channel, Double Top/Bottom)
   - महत्त्वाचे Candlestick Patterns (उदा. Engulfing, Doji, Hammer) कोणते दिसत आहेत?

**2. Option Chain Sentiment:**
   - Option Chain नुसार, सर्वात मोठा Support (Max PE OI) आणि Resistance (Max CE OI) कुठे आहे?
   - यावरून बाजाराचा कल (sentiment) काय वाटतो (Bullish/Bearish)?

**3. 🚨 Final Verdict & Trade Alert:**
   - **Trade Setup आहे का?** (Yes/No)
   - **Action:** (BUY CE / BUY PE)?
   - **Strike:** ₹[STRIKE]
   - **Entry Price (Option Premium):** ₹[ENTRY]
   - **Target:** ₹[TARGET]
   - **Stop Loss:** ₹[SL]
   - **Reason (थोडक्यात):** (हा ट्रेड का घ्यावा याची २-३ कारणे).
   - **No Trade:** जर स्पष्ट trade नसेल, तर "सध्या कोणताही स्पष्ट ट्रेड नाही. Breakout/Breakdown ची वाट पहा." असे लिहा.
"""

    async def analyze(self, symbol, chart_buf, oc_data, candles_df):
        if not self.model: return "AI analysis is disabled."
        try:
            logger.info(f"🤖 Running Price Action analysis for {symbol}...")
            prompt = self.format_data_for_ai(symbol, oc_data, candles_df)
            
            chart_buf.seek(0)
            image_part = {"mime_type": "image/png", "data": chart_buf.read()}
            
            response = await self.model.generate_content_async([prompt, image_part], generation_config={"temperature": 0.2})
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
        self.gemini_analyzer = GeminiPriceActionAnalyzer()
        logger.info("🚀 Price Action AI Bot Initialized (IST Timezone Fixed)")

    async def load_security_ids(self):
        try:
            logger.info("Downloading Dhan instruments file...")
            response = requests.get(DHAN_INSTRUMENTS_URL, timeout=30)
            response.raise_for_status()
            
            all_rows = list(csv.DictReader(io.StringIO(response.text)))
            today = datetime.now(IST) # USE IST TIMEZONE
            self.security_id_map.clear()

            for symbol in STOCKS_WATCHLIST:
                futures = []
                for row in all_rows:
                    try:
                        if (row.get('SEM_TRADING_SYMBOL') == symbol and 
                            row.get('SEM_INSTRUMENT_TYPE') == 'FUTSTK' and
                            row.get('SEM_EXM_EXCH_ID') == 'NFO'):
                            
                            expiry_date_str = row.get('SEM_EXPIRY_DATE', '').split(' ')[0]
                            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
                            expiry_date_ist = IST.localize(expiry_date) # Make expiry date timezone-aware
                            
                            if expiry_date_ist > today:
                                futures.append({'expiry': expiry_date_ist, 'fno_id': int(row.get('SEM_SMST_SECURITY_ID')), 'equity_id': int(row.get('SEM_UNDERLYING_SECURITY_ID'))})
                    except (ValueError, TypeError): continue
                
                if futures:
                    nearest = min(futures, key=lambda x: x['expiry'])
                    self.security_id_map[symbol] = nearest
                    logger.info(f"✅ {symbol}: Loaded F&O Security ID = {nearest['fno_id']}")
            
            logger.info(f"Total {len(self.security_id_map)} F&O securities loaded.")
            return len(self.security_id_map) > 0
        except Exception as e:
            logger.error(f"CRITICAL: Error loading security IDs: {e}", exc_info=True)
            return False

    def get_api_data(self, url, payload):
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json().get('data')
        except Exception as e:
            logger.error(f"API call to {url} failed: {e}")
            return None

    def create_chart(self, df, symbol, spot_price):
        try:
            df_chart = df.copy(); df_chart.set_index('Date', inplace=True)
            df_chart['SMA20'] = df_chart['Close'].rolling(window=20).mean()
            df_chart['SMA50'] = df_chart['Close'].rolling(window=50).mean()
            mc = mpf.make_marketcolors(up='#10FF70', down='#FF3333', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', rc={'font.size': 12})
            apds = [mpf.make_addplot(df_chart['SMA20'], color='#FFD700'), mpf.make_addplot(df_chart['SMA50'], color='#00BFFF')]
            fig, axes = mpf.plot(df_chart.tail(120), type='candle', style=s, volume=True, addplot=apds, title=f'\n{symbol} | Spot: ₹{spot_price:,.2f}', figsize=(16, 8), returnfig=True)
            axes[0].legend(['SMA20', 'SMA50'])
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
            return buf
        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return None

    async def process_stock(self, symbol):
        try:
            info = self.security_id_map[symbol]
            logger.info(f"--- Processing {symbol} ---")
            oc_data = self.get_api_data(DHAN_OPTION_CHAIN_URL, {'securityId': str(info['fno_id']), 'exchangeSegment': 'NSE_FNO'})
            if not oc_data: return
            
            # USE IST TIMEZONE FOR DATES
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=7)
            hist_payload = {"securityId": str(info['equity_id']), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "fromDate": from_date.strftime("%Y-%m-%d"), "toDate": to_date.strftime("%Y-%m-%d"), "interval": "FIVE_MINUTE"}
            hist_data = self.get_api_data(DHAN_INTRADAY_URL, hist_payload)
            if not hist_data: return

            candles_df = pd.DataFrame({'Date': pd.to_datetime(hist_data['start_Time'], unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata'), 'Open': hist_data['open'], 'High': hist_data['high'], 'Low': hist_data['low'], 'Close': hist_data['close'], 'Volume': hist_data['volume']})
            if len(candles_df) < 50: return
            
            spot_price = oc_data.get('spotPrice', 0)
            chart_buf = self.create_chart(candles_df, symbol, spot_price)
            if not chart_buf: return

            ai_analysis_text = await self.gemini_analyzer.analyze(symbol, chart_buf, oc_data, candles_df)
            alert_header = f"🚨 **TRADE ALERT: {symbol}** 🚨" if "BUY CE" in ai_analysis_text or "BUY PE" in ai_analysis_text else f"📉 *Analysis for {symbol}*"
            caption = f"{alert_header}\n\n{ai_analysis_text}"
            
            chart_buf.seek(0)
            await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=InputFile(chart_buf, filename=f"{symbol}.png"), caption=caption, parse_mode='Markdown')
            logger.info(f"✅ Analysis sent for {symbol}")
        except Exception as e:
            logger.error(f"FATAL error processing {symbol}: {e}", exc_info=True)

# --- HTTP SERVER & MAIN EXECUTION ---
class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.send_header('Content-type', 'text/plain'); self.end_headers(); self.wfile.write(b"Bot is alive!")

def run_server():
    port = int(os.environ.get("PORT", 8080))
    try:
        httpd = HTTPServer(('', port), KeepAliveHandler); logger.info(f"Starting server on port {port}..."); httpd.serve_forever()
    except Exception: pass

async def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
        logger.critical("❌ Missing critical environment variables. Exiting."); return
    
    server_thread = Thread(target=run_server, daemon=True); server_thread.start()
    bot = DhanTradingBot()
    
    while True: # Auto-retry loop
        # Check if it's within market hours (approx 9 AM to 4 PM IST)
        now_ist = datetime.now(IST)
        if 9 <= now_ist.hour < 16:
            if await bot.load_security_ids():
                logger.info("Successfully loaded F&O securities. Starting main loop.")
                break
            else:
                logger.warning("Failed to load F&O securities during market hours. Retrying in 5 mins...")
                await asyncio.sleep(300)
        else:
            logger.info(f"Market is closed. Current IST: {now_ist.strftime('%H:%M:%S')}. Waiting for market to open...")
            await asyncio.sleep(600) # Wait 10 minutes

    await bot.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"✅ **Price Action AI Bot ONLINE**\nTracking {len(bot.security_id_map)} F&O stocks.", parse_mode='Markdown')
    while True:
        logger.info("============== NEW SCAN CYCLE ==============")
        for symbol in bot.security_id_map.keys():
            await bot.process_stock(symbol)
            logger.info("--- Waiting 3 seconds before next stock ---")
            await asyncio.sleep(3)
        
        logger.info("Scan cycle complete. Waiting for 10 minutes...")
        await asyncio.sleep(600)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
