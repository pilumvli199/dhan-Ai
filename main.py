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
import numpy as np
import base64
from typing import Dict, List, Optional
import json

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

# AI API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dhan API URLs
DHAN_API_BASE = "https://api.dhan.co"
DHAN_OHLC_URL = f"{DHAN_API_BASE}/v2/marketfeed/ohlc"
DHAN_OPTION_CHAIN_URL = f"{DHAN_API_BASE}/v2/optionchain"
DHAN_EXPIRY_LIST_URL = f"{DHAN_API_BASE}/v2/optionchain/expirylist"
DHAN_INSTRUMENTS_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL = f"{DHAN_API_BASE}/v2/charts/historical"
DHAN_INTRADAY_URL = f"{DHAN_API_BASE}/v2/charts/intraday"

# AI API URLs
GEMINI_FLASH_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_PRO_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
OPENAI_GPT4_URL = "https://api.openai.com/v1/chat/completions"

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
# THREE LAYER AI TRADING BOT
# ========================

class ThreeLayerAITradingBot:
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
        self.trade_signals = []  # Store trade signals
        logger.info("🤖 Three-Layer AI Trading Bot initialized")
    
    # ========================
    # DHAN API FUNCTIONS
    # ========================
    
    async def load_security_ids(self):
        """Load security IDs from Dhan"""
        try:
            logger.info("📥 Loading security IDs from Dhan...")
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
                        except Exception as e:
                            continue
                    
                    csv_data_reset = response.text.split('\n')
                    reader = csv.DictReader(csv_data_reset)
                
                logger.info(f"✅ Total {len(self.security_id_map)} securities loaded")
                return True
            else:
                logger.error(f"❌ Failed to load instruments: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading security IDs: {e}")
            return False
    
    def get_historical_data(self, security_id, segment, symbol):
        """Get last 5 days 5-minute candles"""
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
                
                if 'open' in data and 'high' in data:
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
                            'open': opens[i] if i < len(opens) else 0,
                            'high': highs[i] if i < len(highs) else 0,
                            'low': lows[i] if i < len(lows) else 0,
                            'close': closes[i] if i < len(closes) else 0,
                            'volume': volumes[i] if i < len(volumes) else 0
                        })
                    
                    logger.info(f"📊 {symbol}: Got {len(candles)} candles")
                    return candles
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return None
    
    def get_nearest_expiry(self, security_id, segment):
        """Get nearest expiry"""
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
            logger.error(f"❌ Error getting expiry: {e}")
            return None
    
    def get_option_chain(self, security_id, segment, expiry):
        """Get option chain data"""
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
            logger.error(f"❌ Error getting option chain: {e}")
            return None
    
    def create_candlestick_chart(self, candles, symbol, spot_price):
        """Create candlestick chart and return as base64"""
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
                figcolor='#1e1e1e',
                gridaxis='both',
                y_on_right=False
            )
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f'\n{symbol} - Spot: ₹{spot_price:,.2f}',
                ylabel='Price (₹)',
                ylabel_lower='Volume',
                figsize=(12, 8),
                returnfig=True,
                tight_layout=True
            )
            
            # Save to buffer for Telegram
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf.seek(0)
            
            # Convert to base64 for AI analysis
            buf_copy = io.BytesIO()
            fig.savefig(buf_copy, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            buf_copy.seek(0)
            base64_image = base64.b64encode(buf_copy.read()).decode('utf-8')
            
            plt.close(fig)
            
            return buf, base64_image
            
        except Exception as e:
            logger.error(f"❌ Error creating chart for {symbol}: {e}")
            return None, None
    
    # ========================
    # PRE-FILTERS (OI, PCR, IV Analysis)
    # ========================
    
    def apply_prefilters(self, symbol, oc_data, expiry):
        """Pre-filter based on OI, PCR, IV"""
        try:
            spot_price = oc_data.get('last_price', 0)
            oc = oc_data.get('oc', {})
            
            if not oc:
                return False, "No option chain data"
            
            # Calculate metrics
            strikes = sorted([float(s) for s in oc.keys()])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_vol = 0
            total_pe_vol = 0
            
            # Get ATM ± 5 strikes
            atm_idx = strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(strikes), atm_idx + 6)
            selected_strikes = strikes[start_idx:end_idx]
            
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                strike_data = oc.get(strike_key, {})
                
                ce = strike_data.get('ce', {})
                pe = strike_data.get('pe', {})
                
                total_ce_oi += ce.get('oi', 0)
                total_pe_oi += pe.get('oi', 0)
                total_ce_vol += ce.get('volume', 0)
                total_pe_vol += pe.get('volume', 0)
            
            # PCR Calculation
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
            
            # ATM IV
            atm_data = oc.get(f"{atm_strike:.6f}", {})
            ce_iv = atm_data.get('ce', {}).get('implied_volatility', 0)
            pe_iv = atm_data.get('pe', {}).get('implied_volatility', 0)
            avg_iv = (ce_iv + pe_iv) / 2
            
            # Filter Criteria
            filter_passed = True
            reason = []
            
            # 1. PCR should be between 0.7 to 1.3 (neutral to bullish range)
            if not (0.7 <= pcr_oi <= 1.3):
                filter_passed = False
                reason.append(f"PCR OI={pcr_oi:.2f} (Out of range)")
            
            # 2. IV should be > 15% (volatility present)
            if avg_iv < 15:
                filter_passed = False
                reason.append(f"Avg IV={avg_iv:.1f}% (Too low)")
            
            # 3. OI should be significant (> 100K combined)
            total_oi = total_ce_oi + total_pe_oi
            if total_oi < 100000:
                filter_passed = False
                reason.append(f"Total OI={total_oi/1000:.0f}K (Too low)")
            
            filter_result = {
                'passed': filter_passed,
                'pcr_oi': pcr_oi,
                'pcr_vol': pcr_vol,
                'avg_iv': avg_iv,
                'total_oi': total_oi,
                'reason': ', '.join(reason) if reason else 'All filters passed'
            }
            
            logger.info(f"🔍 {symbol} Pre-Filter: {'✅ PASS' if filter_passed else '❌ FAIL'} - {filter_result['reason']}")
            
            return filter_passed, filter_result
            
        except Exception as e:
            logger.error(f"❌ Error in prefilter for {symbol}: {e}")
            return False, {"error": str(e)}
    
    # ========================
    # LAYER 1: GEMINI FLASH (Quick Scan - 30s)
    # ========================
    
    async def gemini_flash_scan(self, symbol, candles, chart_base64, oc_data, expiry, prefilter_data):
        """Layer 1: Gemini Flash - Quick pattern recognition"""
        try:
            logger.info(f"⚡ Layer 1: Gemini Flash scanning {symbol}...")
            
            # Prepare candlestick data text
            candles_text = self.prepare_candles_text(candles[-50:])  # Last 50 candles
            
            # Prepare option chain text
            oc_text = self.prepare_option_chain_text(symbol, oc_data, expiry)
            
            # Gemini Flash prompt
            prompt = f"""You are a F&O trading expert. Analyze this {symbol} data quickly (30 seconds max).

**Chart Data (Last 50 candles - 5min):**
{candles_text}

**Option Chain:**
{oc_text}

**Pre-Filter Results:**
- PCR OI: {prefilter_data['pcr_oi']:.2f}
- Avg IV: {prefilter_data['avg_iv']:.1f}%
- Total OI: {prefilter_data['total_oi']/1000:.0f}K

**Task:** Identify if this stock has potential for F&O trade.
- Look for: Trend strength, volume surge, support/resistance, breakout patterns
- Output: YES/NO + 1 line reason
- Format: "YES - Strong bullish breakout with volume" OR "NO - Sideways, no clear trend"

Response:"""
            
            # Call Gemini Flash API
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 100
                }
            }
            
            response = requests.post(
                f"{GEMINI_FLASH_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                logger.info(f"⚡ Gemini Flash: {ai_response}")
                
                # Check if approved
                approved = ai_response.upper().startswith('YES')
                
                return {
                    'approved': approved,
                    'response': ai_response,
                    'layer': 'Gemini Flash'
                }
            else:
                logger.error(f"❌ Gemini Flash API error: {response.status_code}")
                return {'approved': False, 'response': 'API Error', 'layer': 'Gemini Flash'}
                
        except Exception as e:
            logger.error(f"❌ Gemini Flash error for {symbol}: {e}")
            return {'approved': False, 'response': str(e), 'layer': 'Gemini Flash'}
    
    # ========================
    # LAYER 2: GEMINI PRO (Strategy Analysis - 1 min)
    # ========================
    
    async def gemini_pro_strategy(self, symbol, candles, chart_base64, oc_data, expiry, prefilter_data, flash_result):
        """Layer 2: Gemini Pro - Deep strategy analysis"""
        try:
            logger.info(f"🎯 Layer 2: Gemini Pro analyzing {symbol}...")
            
            # Prepare data
            candles_text = self.prepare_candles_text(candles[-100:])  # Last 100 candles
            oc_text = self.prepare_option_chain_text(symbol, oc_data, expiry)
            
            # Gemini Pro prompt
            prompt = f"""You are a professional F&O trader. Analyze {symbol} deeply for option trading strategy.

**Chart Data (Last 100 candles - 5min):**
{candles_text}

**Option Chain:**
{oc_text}

**Pre-Filter:**
- PCR OI: {prefilter_data['pcr_oi']:.2f}
- Avg IV: {prefilter_data['avg_iv']:.1f}%

**Gemini Flash Opinion:** {flash_result['response']}

**Your Task:**
1. Confirm if this is a HIGH-PROBABILITY trade (YES/NO)
2. If YES, suggest:
   - Strike price (CE/PE)
   - Entry logic
   - Key levels (support/resistance)
   - Risk factors

Format:
YES/NO
Strike: [strike] CE/PE
Entry: [reason]
Levels: Support [X], Resistance [Y]
Risk: [main risk]

Response:"""
            
            # Call Gemini Pro API
            headers = {'Content-Type': 'application/json'}
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.5,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 300
                }
            }
            
            response = requests.post(
                f"{GEMINI_PRO_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                logger.info(f"🎯 Gemini Pro: {ai_response[:200]}...")
                
                # Check if approved
                approved = ai_response.upper().startswith('YES')
                
                return {
                    'approved': approved,
                    'response': ai_response,
                    'layer': 'Gemini Pro'
                }
            else:
                logger.error(f"❌ Gemini Pro API error: {response.status_code}")
                return {'approved': False, 'response': 'API Error', 'layer': 'Gemini Pro'}
                
        except Exception as e:
            logger.error(f"❌ Gemini Pro error for {symbol}: {e}")
            return {'approved': False, 'response': str(e), 'layer': 'Gemini Pro'}
    
    # ========================
    # LAYER 3: GPT-4o (Final Call - 30s)
    # ========================
    
    async def gpt4o_final_call(self, symbol, candles, chart_base64, oc_data, expiry, prefilter_data, flash_result, pro_result):
        """Layer 3: GPT-4o with Vision - Final trading signal"""
        try:
            logger.info(f"🤖 Layer 3: GPT-4o making final call for {symbol}...")
            
            # Prepare data
            candles_text = self.prepare_candles_text(candles[-100:])
            oc_text = self.prepare_option_chain_text(symbol, oc_data, expiry)
            spot_price = oc_data.get('last_price', 0)
            
            # GPT-4o prompt
            prompt = f"""You are an expert F&O trader making the FINAL TRADING DECISION for {symbol}.

**Market Data:**
Spot Price: ₹{spot_price:,.2f}
Expiry: {expiry}

**Chart Analysis (Last 100 candles - 5min):**
{candles_text}

**Option Chain:**
{oc_text}

**Pre-Filter Metrics:**
- PCR OI: {prefilter_data['pcr_oi']:.2f}
- Avg IV: {prefilter_data['avg_iv']:.1f}%
- Total OI: {prefilter_data['total_oi']/1000:.0f}K

**AI Layer 1 (Gemini Flash):** {flash_result['response']}

**AI Layer 2 (Gemini Pro):** {pro_result['response']}

**YOUR FINAL DECISION:**
Analyze the chart image + data and give TRADE SIGNAL in this exact format:

TRADE: YES/NO
TYPE: CE/PE
STRIKE: [strike price]
ENTRY: ₹[price]
TARGET: ₹[price] (+[%])
STOP_LOSS: ₹[price] (-[%])
EXIT_TIME: [time]
RISK_REWARD: [ratio]
CONFIDENCE: [%]
REASON: [1-2 lines]

If NO trade, just say: TRADE: NO - [reason]"""
            
            # Call GPT-4o with Vision
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {OPENAI_API_KEY}'
            }
            
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
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                logger.info(f"🤖 GPT-4o: {ai_response[:200]}...")
                
                # Parse trade signal
                trade_signal = self.parse_trade_signal(ai_response, symbol, spot_price, expiry)
                
                return trade_signal
            else:
                logger.error(f"❌ GPT-4o API error: {response.status_code} - {response.text}")
                return {'trade': False, 'reason': 'API Error', 'layer': 'GPT-4o'}
                
        except Exception as e:
            logger.error(f"❌ GPT-4o error for {symbol}: {e}")
            return {'trade': False, 'reason': str(e), 'layer': 'GPT-4o'}
    
    # ========================
    # HELPER FUNCTIONS
    # ========================
    
    def prepare_candles_text(self, candles):
        """Convert candles to readable text format"""
        text = "Time | Open | High | Low | Close | Volume\n"
        text += "-" * 60 + "\n"
        
        for candle in candles[-20:]:  # Last 20 for brevity
            time = candle.get('timestamp', '')[-8:] if candle.get('timestamp') else ''
            text += f"{time} | {candle['open']:.2f} | {candle['high']:.2f} | {candle['low']:.2f} | {candle['close']:.2f} | {candle['volume']}\n"
        
        return text
    
    def prepare_option_chain_text(self, symbol, oc_data, expiry):
        """Convert option chain to readable text"""
        spot_price = oc_data.get('last_price', 0)
        if not oc:
            return "No option chain data available"
        
        strikes = sorted([float(s) for s in oc.keys()])
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        atm_idx = strikes.index(atm_strike)
        start_idx = max(0, atm_idx - 5)
        end_idx = min(len(strikes), atm_idx + 6)
        selected_strikes = strikes[start_idx:end_idx]
        
        text = f"Spot: ₹{spot_price:,.2f} | Expiry: {expiry}\n"
        text += "Strike | CE-LTP | CE-OI | CE-Vol | PE-LTP | PE-OI | PE-Vol | CE-IV | PE-IV\n"
        text += "-" * 100 + "\n"
        
        for strike in selected_strikes:
            strike_key = f"{strike:.6f}"
            strike_data = oc.get(strike_key, {})
            
            ce = strike_data.get('ce', {})
            pe = strike_data.get('pe', {})
            
            atm_mark = "🎯" if strike == atm_strike else "  "
            
            text += f"{atm_mark}{strike:.0f} | {ce.get('last_price', 0):.2f} | {ce.get('oi', 0)/1000:.0f}K | {ce.get('volume', 0)/1000:.0f}K | "
            text += f"{pe.get('last_price', 0):.2f} | {pe.get('oi', 0)/1000:.0f}K | {pe.get('volume', 0)/1000:.0f}K | "
            text += f"{ce.get('implied_volatility', 0):.1f}% | {pe.get('implied_volatility', 0):.1f}%\n"
        
        return text
    
    def parse_trade_signal(self, ai_response, symbol, spot_price, expiry):
        """Parse GPT-4o response into structured trade signal"""
        try:
            lines = ai_response.strip().split('\n')
            signal = {
                'symbol': symbol,
                'spot_price': spot_price,
                'expiry': expiry,
                'trade': False,
                'layer': 'GPT-4o'
            }
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('TRADE:'):
                    signal['trade'] = 'YES' in line.upper()
                elif line.startswith('TYPE:'):
                    signal['type'] = line.split(':')[1].strip()
                elif line.startswith('STRIKE:'):
                    signal['strike'] = line.split(':')[1].strip()
                elif line.startswith('ENTRY:'):
                    signal['entry'] = line.split(':')[1].strip()
                elif line.startswith('TARGET:'):
                    signal['target'] = line.split(':')[1].strip()
                elif line.startswith('STOP_LOSS:'):
                    signal['stop_loss'] = line.split(':')[1].strip()
                elif line.startswith('EXIT_TIME:'):
                    signal['exit_time'] = line.split(':')[1].strip()
                elif line.startswith('RISK_REWARD:'):
                    signal['risk_reward'] = line.split(':')[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    signal['confidence'] = line.split(':')[1].strip()
                elif line.startswith('REASON:'):
                    signal['reason'] = line.split(':', 1)[1].strip()
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error parsing trade signal: {e}")
            return {'trade': False, 'reason': 'Parse error', 'symbol': symbol}
    
    # ========================
    # MAIN SCAN & ANALYSIS PIPELINE
    # ========================
    
    async def analyze_stock(self, symbol):
        """Complete 3-layer AI analysis for one stock"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 ANALYZING: {symbol}")
            logger.info(f"{'='*60}")
            
            if symbol not in self.security_id_map:
                logger.warning(f"⚠️ Skipping {symbol} - No security ID")
                return None
            
            info = self.security_id_map[symbol]
            security_id = info['security_id']
            segment = info['segment']
            
            # Step 1: Get market data
            logger.info(f"📥 Step 1: Fetching market data...")
            
            expiry = self.get_nearest_expiry(security_id, segment)
            if not expiry:
                logger.warning(f"⚠️ {symbol}: No expiry found")
                return None
            
            oc_data = self.get_option_chain(security_id, segment, expiry)
            if not oc_data:
                logger.warning(f"⚠️ {symbol}: No option chain data")
                return None
            
            candles = self.get_historical_data(security_id, segment, symbol)
            if not candles or len(candles) < 50:
                logger.warning(f"⚠️ {symbol}: Insufficient candle data")
                return None
            
            spot_price = oc_data.get('last_price', 0)
            
            # Create chart
            chart_buf, chart_base64 = self.create_candlestick_chart(candles, symbol, spot_price)
            if not chart_base64:
                logger.warning(f"⚠️ {symbol}: Chart creation failed")
                return None
            
            # Step 2: Pre-filters
            logger.info(f"🔍 Step 2: Applying pre-filters...")
            filter_passed, prefilter_data = self.apply_prefilters(symbol, oc_data, expiry)
            
            if not filter_passed:
                logger.info(f"❌ {symbol}: Pre-filter FAILED - {prefilter_data.get('reason', 'Unknown')}")
                return None
            
            logger.info(f"✅ {symbol}: Pre-filter PASSED")
            await asyncio.sleep(1)  # Rate limit
            
            # Step 3: Layer 1 - Gemini Flash
            logger.info(f"⚡ Step 3: Layer 1 - Gemini Flash analysis...")
            flash_result = await self.gemini_flash_scan(symbol, candles, chart_base64, oc_data, expiry, prefilter_data)
            
            if not flash_result['approved']:
                logger.info(f"❌ {symbol}: Gemini Flash REJECTED - {flash_result['response']}")
                return None
            
            logger.info(f"✅ {symbol}: Gemini Flash APPROVED")
            await asyncio.sleep(1)  # Rate limit
            
            # Step 4: Layer 2 - Gemini Pro
            logger.info(f"🎯 Step 4: Layer 2 - Gemini Pro analysis...")
            pro_result = await self.gemini_pro_strategy(symbol, candles, chart_base64, oc_data, expiry, prefilter_data, flash_result)
            
            if not pro_result['approved']:
                logger.info(f"❌ {symbol}: Gemini Pro REJECTED - {pro_result['response']}")
                return None
            
            logger.info(f"✅ {symbol}: Gemini Pro APPROVED")
            await asyncio.sleep(1)  # Rate limit
            
            # Step 5: Layer 3 - GPT-4o Final Call
            logger.info(f"🤖 Step 5: Layer 3 - GPT-4o final call...")
            trade_signal = await self.gpt4o_final_call(symbol, candles, chart_base64, oc_data, expiry, prefilter_data, flash_result, pro_result)
            
            if not trade_signal.get('trade'):
                logger.info(f"❌ {symbol}: GPT-4o REJECTED - {trade_signal.get('reason', 'Unknown')}")
                return None
            
            logger.info(f"🎉 {symbol}: TRADE SIGNAL GENERATED!")
            
            # Compile complete result
            result = {
                'symbol': symbol,
                'spot_price': spot_price,
                'expiry': expiry,
                'chart_buf': chart_buf,
                'prefilter_data': prefilter_data,
                'flash_result': flash_result,
                'pro_result': pro_result,
                'trade_signal': trade_signal,
                'timestamp': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    # ========================
    # TELEGRAM MESSAGING
    # ========================
    
    async def send_trade_signal(self, result):
        """Send trade signal to Telegram"""
        try:
            symbol = result['symbol']
            signal = result['trade_signal']
            
            # Send chart first
            if result.get('chart_buf'):
                await self.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=result['chart_buf'],
                    caption=f"📊 {symbol} Technical Chart"
                )
                await asyncio.sleep(1)
            
            # Format trade signal message
            msg = f"🎯 *TRADE SIGNAL: {symbol}*\n\n"
            msg += f"┌─────────────────────────────┐\n"
            msg += f"│  💰 Spot: ₹{result['spot_price']:,.2f}\n"
            msg += f"│  📅 Expiry: {result['expiry']}\n"
            msg += f"│  🎲 Type: {signal.get('type', 'N/A')}\n"
            msg += f"│  🎯 Strike: {signal.get('strike', 'N/A')}\n"
            msg += f"└─────────────────────────────┘\n\n"
            
            msg += f"📈 *ENTRY:* {signal.get('entry', 'N/A')}\n"
            msg += f"🎯 *TARGET:* {signal.get('target', 'N/A')}\n"
            msg += f"🛑 *STOP LOSS:* {signal.get('stop_loss', 'N/A')}\n"
            msg += f"⏰ *EXIT TIME:* {signal.get('exit_time', 'N/A')}\n"
            msg += f"📊 *RISK:REWARD:* {signal.get('risk_reward', 'N/A')}\n"
            msg += f"🎲 *CONFIDENCE:* {signal.get('confidence', 'N/A')}\n\n"
            
            msg += f"💡 *REASON:*\n{signal.get('reason', 'N/A')}\n\n"
            
            msg += f"┌─────── AI ANALYSIS ───────┐\n"
            msg += f"│ ⚡ Gemini Flash: ✅ PASS\n"
            msg += f"│ 🎯 Gemini Pro: ✅ PASS\n"
            msg += f"│ 🤖 GPT-4o: ✅ TRADE\n"
            msg += f"└───────────────────────────┘\n\n"
            
            msg += f"📊 *Pre-Filter Metrics:*\n"
            msg += f"• PCR OI: {result['prefilter_data']['pcr_oi']:.2f}\n"
            msg += f"• Avg IV: {result['prefilter_data']['avg_iv']:.1f}%\n"
            msg += f"• Total OI: {result['prefilter_data']['total_oi']/1000:.0f}K\n\n"
            
            msg += f"🕐 Generated: {result['timestamp']}\n"
            msg += f"⚡ Powered by: Gemini + GPT-4o"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Trade signal sent for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error sending trade signal: {e}")
    
    async def send_startup_message(self):
        """Send bot startup message"""
        try:
            msg = "🤖 *F&O TRADING BOT v2.0 STARTED!*\n\n"
            msg += "┌─────────────────────────────┐\n"
            msg += "│  📡 Data: DhanHQ API\n"
            msg += "│  🔍 Pre-Filters: OI, PCR, IV\n"
            msg += "│  ⚡ AI Layer 1: Gemini Flash\n"
            msg += "│  🎯 AI Layer 2: Gemini Pro\n"
            msg += "│  🤖 AI Layer 3: GPT-4o Vision\n"
            msg += "└─────────────────────────────┘\n\n"
            msg += f"📊 Tracking: {len(self.security_id_map)} stocks/indices\n"
            msg += f"⏱️ Scan Interval: 5 minutes\n"
            msg += f"🎯 Analysis Time: ~5 sec/stock\n\n"
            msg += "✅ Bot is now scanning for trades...\n"
            msg += "🚂 Deployed on Railway.app"
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup message sent")
        except Exception as e:
            logger.error(f"❌ Error sending startup message: {e}")
    
    async def send_scan_summary(self, scanned, approved, rejected):
        """Send scan cycle summary"""
        try:
            msg = f"📊 *SCAN CYCLE COMPLETED*\n\n"
            msg += f"🔍 Scanned: {scanned} stocks\n"
            msg += f"✅ Approved: {approved} trades\n"
            msg += f"❌ Rejected: {rejected} stocks\n\n"
            msg += f"⏱️ Next scan in 5 minutes..."
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Error sending scan summary: {e}")
    
    # ========================
    # MAIN RUN LOOP
    # ========================
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Three-Layer AI Trading Bot starting...")
        
        # Load security IDs
        success = await self.load_security_ids()
        if not success:
            logger.error("❌ Failed to load security IDs. Exiting...")
            return
        
        await self.send_startup_message()
        
        all_symbols = list(self.security_id_map.keys())
        
        logger.info(f"✅ Ready to scan {len(all_symbols)} symbols")
        logger.info(f"⏱️ Analysis time per stock: ~5 seconds (due to AI rate limits)")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 STARTING SCAN CYCLE AT {timestamp}")
                logger.info(f"{'='*70}\n")
                
                scanned = 0
                approved = 0
                rejected = 0
                
                # Scan each symbol
                for idx, symbol in enumerate(all_symbols, 1):
                    logger.info(f"\n[{idx}/{len(all_symbols)}] Processing {symbol}...")
                    
                    result = await self.analyze_stock(symbol)
                    scanned += 1
                    
                    if result:
                        # Trade signal generated!
                        approved += 1
                        await self.send_trade_signal(result)
                        self.trade_signals.append(result)
                    else:
                        rejected += 1
                    
                    # Rate limit: 5 seconds per stock (to respect Gemini free tier: 1 req/sec)
                    logger.info(f"⏳ Waiting 5 seconds before next stock...")
                    await asyncio.sleep(5)
                
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ SCAN CYCLE COMPLETED")
                logger.info(f"📊 Scanned: {scanned} | Approved: {approved} | Rejected: {rejected}")
                logger.info(f"{'='*70}\n")
                
                # Send summary
                await self.send_scan_summary(scanned, approved, rejected)
                
                # Wait 5 minutes before next cycle
                logger.info("⏳ Waiting 5 minutes for next scan cycle...\n")
                await asyncio.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)


# ========================
# MAIN ENTRY POINT
# ========================
if __name__ == "__main__":
    try:
        # Check environment variables
        required_vars = [
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
            DHAN_CLIENT_ID,
            DHAN_ACCESS_TOKEN,
            GEMINI_API_KEY,
            OPENAI_API_KEY
        ]
        
        if not all(required_vars):
            logger.error("❌ Missing environment variables!")
            logger.error("Required:")
            logger.error("  - TELEGRAM_BOT_TOKEN")
            logger.error("  - TELEGRAM_CHAT_ID")
            logger.error("  - DHAN_CLIENT_ID")
            logger.error("  - DHAN_ACCESS_TOKEN")
            logger.error("  - GEMINI_API_KEY")
            logger.error("  - OPENAI_API_KEY")
            exit(1)
        
        # Start bot
        bot = ThreeLayerAITradingBot()
        asyncio.run(bot.run())
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        exit(1)
        oc = oc_data.get('oc', {})
