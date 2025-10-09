#!/usr/bin/env python3
# main.py - Dhan Option Chain Bot with Gemini vision analysis (1 req/sec)
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
import mplfinance as mpf
import pandas as pd
import base64
import json
import time
from PIL import Image

# Gemini client (google-genai)
try:
    from google import genai
except Exception:
    genai = None

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # preferred for free-tier API key use

# Dhan API URLs (as in your original)
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# Stock/Index list (same as you)
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
        logger.info("Bot initialized successfully")

        # Gemini related
        self.gemini_client = None
        self.gemini_semaphore = asyncio.Semaphore(1)  # allow 1 concurrent Gemini call
        self.gemini_min_interval = 1.0  # seconds between calls
        self._last_gemini_call = 0.0

        # init genai client if key present
        if GEMINI_API_KEY:
            if genai is None:
                logger.warning("google-genai library not installed; Gemini integration won't work until installed.")
            else:
                try:
                    # instantiate client with API key (works for free-tier API key)
                    self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                    logger.info("Gemini client initialized with API key")
                except Exception as e:
                    logger.error(f"Failed to init Gemini client: {e}")
        else:
            logger.info("No GEMINI_API_KEY found - Gemini analysis disabled")

    # --------------------
    # Security ID loader
    # --------------------
    async def load_security_ids(self):
        """Dhan मधून security IDs load करतो (without pandas)"""
        try:
            logger.info("Loading security IDs from Dhan...")
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
                                        logger.info(f"✅ {symbol}: Security ID = {sec_id}")
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
                                        logger.info(f"✅ {symbol}: Security ID = {sec_id}")
                                        break
                        except Exception:
                            continue

                    # Reset CSV reader for next symbol
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)

                logger.info(f"Total {len(self.security_id_map)} securities loaded")
                return True
            else:
                logger.error(f"Failed to load instruments: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error loading security IDs: {e}")
            return False

    # --------------------
    # Historical candles
    # --------------------
    def get_historical_data(self, security_id, segment, symbol):
        """Last 5 days चे सर्व 5-minute candles घेतो"""
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

            logger.info(f"Intraday API call for {symbol}: {payload}")

            response = requests.post(
                DHAN_INTRADAY_URL,
                json=payload,
                headers=self.headers,
                timeout=15
            )

            logger.info(f"{symbol} Intraday response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if 'open' in data and 'high' in data and 'low' in data and 'close' in data:
                    opens = data.get('open', [])
                    highs = data.get('high', [])
                    lows = data.get('low', [])
                    closes = data.get('close', [])
                    volumes = data.get('volume', [])
                    timestamps = data.get('start_Time', [])

                    logger.info(f"{symbol}: Total arrays length - Open:{len(opens)}, Time:{len(timestamps)}")

                    candles = []
                    for i in range(len(opens)):
                        candles.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else '',
                            'open': opens[i] if i < len(opens) else 0,
                            'high': highs[i] if i < len(highs) else 0,
                            'low': lows[i] if i < len(lows) else 0,
                            'close': closes[i] if i < len(closes) else 0,
                            'volume': volumes[i] if i < len(volumes) else 0
                        })

                    logger.info(f"{symbol}: Returning ALL {len(candles)} candles from last 5 days (5 min)")
                    return candles
                else:
                    logger.warning(f"{symbol}: Invalid response format - {str(data)[:200]}")
                    return None

            logger.warning(f"{symbol}: Historical data नाही मिळाला - Status: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    # --------------------
    # Candlestick chart creation
    # --------------------
    def create_candlestick_chart(self, candles, symbol, spot_price):
        """Candlestick chart तयार करतो and return io.BytesIO"""
        try:
            df_data = []
            for candle in candles:
                timestamp = candle.get('timestamp', candle.get('start_Time', ''))
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
                logger.warning(f"{symbol}: Not enough candles ({len(df)}) for chart")
                return None

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
                figcolor='#1e1e1e',
                gridaxis='both',
                y_on_right=False
            )

            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} - Last {len(candles)} Candles | Spot: ₹{spot_price:,.2f}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(12, 8),
                returnfig=True,
                tight_layout=True
            )

            axes[0].set_title(
                f'{symbol} - Last {len(candles)} Candles | Spot: ₹{spot_price:,.2f}',
                color='white',
                fontsize=14,
                fontweight='bold',
                pad=20
            )

            for ax in axes:
                ax.tick_params(colors='white', which='both')
                for spine in ax.spines.values():
                    spine.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf.seek(0)
            plt.close(fig)

            return buf

        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return None

    # --------------------
    # Option chain helpers (same as before)
    # --------------------
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
                    expiries = data['data']
                    if expiries:
                        return expiries[0]
            return None

        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
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
            logger.error(f"Error getting option chain: {e}")
            return None

    def format_option_chain_message(self, symbol, data, expiry):
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
            msg += "Strike   CE-LTP  CE-OI  CE-Vol  PE-LTP  PE-OI  PE-Vol\n"
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

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

                atm_mark = "🔸" if strike == atm_strike else "  "

                msg += f"{atm_mark}{strike:6.0f}  {ce_ltp:6.1f} {ce_oi/1000:6.0f}K {ce_vol/1000:6.0f}K  {pe_ltp:6.1f} {pe_oi/1000:6.0f}K {pe_vol/1000:6.0f}K\n"

            msg += "```\n\n"

            atm_data = oc_data.get(f"{atm_strike:.6f}", {})
            if atm_data:
                ce_greeks = atm_data.get('ce', {}).get('greeks', {})
                pe_greeks = atm_data.get('pe', {}).get('greeks', {})
                ce_iv = atm_data.get('ce', {}).get('implied_volatility', 0)
                pe_iv = atm_data.get('pe', {}).get('implied_volatility', 0)

                msg += "📈 *ATM Greeks & IV:*\n"
                msg += f"CE: Δ={ce_greeks.get('delta', 0):.3f} Θ={ce_greeks.get('theta', 0):.2f} IV={ce_iv:.1f}%\n"
                msg += f"PE: Δ={pe_greeks.get('delta', 0):.3f} Θ={pe_greeks.get('theta', 0):.2f} IV={pe_iv:.1f}%\n"

            return msg

        except Exception as e:
            logger.error(f"Error formatting message for {symbol}: {e}")
            return None

    # --------------------
    # Gemini analyze (async wrapper)
    # --------------------
    async def analyze_image_with_gemini_async(self, image_buf, prompt_text):
        """
        Async wrapper that ensures:
          - only 1 call per self.gemini_min_interval seconds
          - calls blocking genai client in a thread (asyncio.to_thread)
        Returns parsed result dict: {'raw_text':..., 'json':...} or {'error':...}
        """
        if not self.gemini_client:
            return {"error": "Gemini client not configured"}

        # Rate limit: ensure at least gemini_min_interval seconds between calls
        async with self.gemini_semaphore:
            now = time.time()
            elapsed = now - self._last_gemini_call
            if elapsed < self.gemini_min_interval:
                to_wait = self.gemini_min_interval - elapsed
                await asyncio.sleep(to_wait)

            # run blocking call in thread
            result = await asyncio.to_thread(self._call_gemini_sync, image_buf, prompt_text)

            # record last call time
            self._last_gemini_call = time.time()

            # enforce at least 1 second between calls by sleeping a small amount (redundant with above)
            await asyncio.sleep(0)  # just yield

            return result

    def _call_gemini_sync(self, image_buf, prompt_text, model="gemini-2.5-flash-image"):
        """Blocking synchronous call to google-genai client. Returns dict."""
        try:
            if genai is None:
                return {"error": "google-genai library not installed"}

            # image bytes -> PIL -> base64 optional (genai supports passing a PIL Image directly in some clients)
            img = Image.open(image_buf)
            # Ensure buffer pointer
            image_buf.seek(0)

            # Build contents: prompt text then image
            # Many genai client examples show: client.generate_content(model=..., contents=[prompt, image])
            try:
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=[prompt_text, img],
                    # you can tune temperature, max_output_tokens if needed
                )
            except Exception as e:
                # fallback to older entrypoint if library version differs
                try:
                    response = self.gemini_client.generate(
                        model=model,
                        input=[{"content": prompt_text, "image": {"mime": "image/png", "data": base64.b64encode(image_buf.getvalue()).decode()}}]
                    )
                except Exception as e2:
                    return {"error": f"Gemini call failure: {e}; fallback failed: {e2}"}

            # Parse response: defensive
            raw_text = ""
            json_obj = None
            try:
                # many responses: response.candidates[0].content.parts etc.
                # Try common paths:
                if hasattr(response, "candidates") and len(response.candidates) > 0:
                    parts = getattr(response.candidates[0], "content", None) or getattr(response.candidates[0], "content", None)
                    # sometimes content is string
                    if isinstance(parts, str):
                        raw_text = parts
                    else:
                        # try to concatenate text parts
                        try:
                            if hasattr(parts, "parts"):
                                text_parts = []
                                for p in parts.parts:
                                    if getattr(p, "text", None):
                                        text_parts.append(p.text)
                                raw_text = "\n".join(text_parts)
                        except Exception:
                            raw_text = str(response.candidates[0])
                elif hasattr(response, "output"):
                    raw_text = str(response.output)
                elif hasattr(response, "text"):
                    raw_text = response.text
                else:
                    raw_text = str(response)
            except Exception:
                raw_text = str(response)

            # Try extract JSON object inside raw_text
            m = None
            try:
                import re
                m = re.search(r'(\{[\s\S]*\})', raw_text)
                if m:
                    try:
                        json_obj = json.loads(m.group(1))
                    except Exception:
                        json_obj = None
            except Exception:
                json_obj = None

            return {"raw_text": raw_text, "json": json_obj}

        except Exception as e:
            return {"error": str(e)}

    # --------------------
    # send batch (modified to call Gemini)
    # --------------------
    async def send_option_chain_batch(self, symbols_batch):
        """एका batch चे option chain data + chart पाठवतो and Gemini analysis करते"""
        for symbol in symbols_batch:
            try:
                if symbol not in self.security_id_map:
                    logger.warning(f"Skipping {symbol} - No security ID")
                    continue

                info = self.security_id_map[symbol]
                security_id = info['security_id']
                segment = info['segment']

                expiry = self.get_nearest_expiry(security_id, segment)
                if not expiry:
                    logger.warning(f"{symbol}: Expiry नाही मिळाला")
                    continue

                logger.info(f"Fetching data for {symbol} (Expiry: {expiry})...")

                oc_data = self.get_option_chain(security_id, segment, expiry)
                if not oc_data:
                    logger.warning(f"{symbol}: Option chain data नाही मिळाला")
                    continue

                spot_price = oc_data.get('last_price', 0)

                logger.info(f"Fetching historical candles for {symbol}...")
                candles = self.get_historical_data(security_id, segment, symbol)

                chart_buf = None
                if candles:
                    logger.info(f"Creating candlestick chart for {symbol}...")
                    chart_buf = self.create_candlestick_chart(candles, symbol, spot_price)

                # Send chart image first (if available)
                if chart_buf:
                    try:
                        await self.bot.send_photo(
                            chat_id=TELEGRAM_CHAT_ID,
                            photo=chart_buf,
                            caption=f"📊 {symbol} - Last {len(candles)} Candles Chart"
                        )
                        logger.info(f"✅ {symbol} chart sent")
                        # reset buffer pointer for Gemini usage
                        chart_buf.seek(0)
                    except Exception as e:
                        logger.error(f"Error sending chart to Telegram for {symbol}: {e}")

                # Prepare option chain message and send
                message = self.format_option_chain_message(symbol, oc_data, expiry)
                if message:
                    try:
                        await self.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"✅ {symbol} option chain sent")
                    except Exception as e:
                        logger.error(f"Error sending option chain message for {symbol}: {e}")

                # Gemini analysis: if configured, send chart + structured prompt
                if chart_buf and self.gemini_client:
                    try:
                        # Build a concise prompt that includes option-chain context and candles summary
                        # For speed, include only a short summary: spot, ATM, top OI strikes
                        oc = oc_data.get('oc', {})
                        spot = spot_price
                        # compute top 3 OI strikes on CE and PE (best-effort)
                        try:
                            strike_items = []
                            for k, v in oc.items():
                                try:
                                    strike_val = float(k)
                                    ce_oi = v.get('ce', {}).get('oi', 0) or 0
                                    pe_oi = v.get('pe', {}).get('oi', 0) or 0
                                    strike_items.append((strike_val, ce_oi, pe_oi))
                                except Exception:
                                    continue
                            # sort by max OI
                            strike_items_sorted = sorted(strike_items, key=lambda x: max(x[1], x[2]), reverse=True)
                            top_oi = strike_items_sorted[:3]
                            top_oi_summary = ", ".join([f"{int(s[0])}(CE_OI={int(s[1])},PE_OI={int(s[2])})" for s in top_oi])
                        except Exception:
                            top_oi_summary = ""
                        prompt_text = (
                            "You are an experienced technical market analyst. "
                            "Analyze the attached 5-minute candlestick chart (image) along with the following option-chain context. "
                            f"Spot={spot:.2f}. Top OI strikes: {top_oi_summary}. "
                            "Please return a concise JSON with keys: trend (up/down/sideways), "
                            "supports (list of approx prices), resistances (list), notable_patterns (list), "
                            "volume_spikes (list), last_visible_price (approx), watch_levels (list of 2), "
                            "and a short human readable summary under 'summary'. Do not provide financial advice."
                        )

                        logger.info(f"Sending to Gemini for {symbol} (rate-limited 1/s)...")
                        gemini_result = await self.analyze_image_with_gemini_async(chart_buf, prompt_text)
                        logger.info(f"Gemini result for {symbol}: {str(gemini_result)[:300]}")

                        if gemini_result:
                            if gemini_result.get('error'):
                                logger.error(f"Gemini error for {symbol}: {gemini_result.get('error')}")
                            else:
                                # Prepare message: if json present, pretty-print; else raw_text
                                if gemini_result.get('json'):
                                    pretty = json.dumps(gemini_result['json'], indent=2)
                                    txt = f"🧠 *Gemini Vision Analysis — {symbol}*\n```\n{pretty}\n```\n"
                                    # also add summary if available
                                    try:
                                        summ = gemini_result['json'].get('summary', None)
                                        if summ:
                                            txt += f"\n*Summary:* {summ}"
                                    except Exception:
                                        pass
                                else:
                                    raw = gemini_result.get('raw_text', '')
                                    txt = f"🧠 *Gemini Vision Analysis — {symbol}*\n{raw}\n"
                                # send to Telegram
                                await self.bot.send_message(
                                    chat_id=TELEGRAM_CHAT_ID,
                                    text=txt,
                                    parse_mode='Markdown'
                                )
                    except Exception as e:
                        logger.error(f"Error during Gemini analysis for {symbol}: {e}")

                # Respect Dhan recommended rate (3s) per original code
                await asyncio.sleep(3)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(3)

    # --------------------
    # Main loop
    # --------------------
    async def run(self):
        logger.info("🚀 Bot started! Loading security IDs...")

        success = await self.load_security_ids()
        if not success:
            logger.error("Failed to load security IDs. Exiting...")
            return

        await self.send_startup_message()

        all_symbols = list(self.security_id_map.keys())
        batch_size = 5
        batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]

        logger.info(f"Total {len(all_symbols)} symbols in {len(batches)} batches")

        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting update cycle at {timestamp}")
                logger.info(f"{'='*50}")

                for batch_num, batch in enumerate(batches, 1):
                    logger.info(f"\n📦 Processing Batch {batch_num}/{len(batches)}: {batch}")
                    await self.send_option_chain_batch(batch)
                    if batch_num < len(batches):
                        await asyncio.sleep(5)

                logger.info("\n✅ All batches completed!")
                logger.info("⏳ Waiting 5 minutes for next cycle...\n")
                await asyncio.sleep(300)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

    async def send_startup_message(self):
        try:
            msg = "🤖 *Dhan Option Chain Bot Started!*\n\n"
            msg += f"📊 Tracking {len(self.security_id_map)} stocks/indices\n"
            msg += "⏱️ Updates every 5 minutes\n"
            msg += "📈 Features:\n"
            msg += "  • Candlestick Charts (Last candles)\n"
            msg += "  • Option Chain: CE/PE LTP, OI, Volume\n"
            msg += "  • Gemini Vision Analysis (if GEMINI_API_KEY set) - max 1 req/sec\n\n"
            msg += "✅ Powered by DhanHQ API v2\n\n"
            msg += "_Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)_"

            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


# ========================
# BOT RUN करा
# ========================
if __name__ == "__main__":
    try:
        if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
            logger.error("❌ Missing environment variables!")
            logger.error("Please set: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN")
            exit(1)

        bot = DhanOptionChainBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
