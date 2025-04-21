import asyncio
import websockets
import json
import datetime
import requests
import time
import logging
import os
from collections import deque
import numpy as np
from pybit.unified_trading import HTTP

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8150510214:AAHpr2CpTsArAJfz2Ro6URSedm_A5WsiMQo"
TELEGRAM_CHAT_ID = "-4665308596"

# Logging Setup
LOG_DIR = "/home/ubuntu/Rheza/local-share/06_trades_and_orderbooks"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/bybit_ft_bot_v1_1_doge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

def send_telegram_message(message):
    """Send a message to a Telegram group using the bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to send Telegram message: {response.text}")
    else:
        logging.info("Telegram message sent successfully!")

import logging
from pybit.unified_trading import HTTP

class BybitAPI:
    def __init__(self, api_key, api_secret, testnet=False):
        """Initialize the Bybit API session."""
        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)

    def market_buy(self, symbol, side, qty):
        """Place a market buy/sell order."""
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,  # Buy / Sell
            orderType="Market",
            qty=qty,
            timeInForce="GoodTillCancel"
        )

    def tp_limit(self, symbol, qty, tp_price):
        """Place a Take Profit Limit Sell order."""
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side="Sell",  # Sell to close long position
            orderType="Limit",
            qty=qty,
            price=tp_price,  # Set TP price (e.g., 110)
            timeInForce="GoodTillCancel",  # Order stays until filled/canceled
            reduceOnly=True  # Ensures it only reduces an existing position
        )

    def sl_market(self, symbol, side, qty, sl_market_price):
        """Place a stop-loss market order."""
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,  
            orderType="Market",    
            qty=qty,              
            triggerPrice=sl_market_price,     
            triggerBy="LastPrice",  
            triggerDirection=2,  # Trigger when price drops below `triggerPrice`
            reduceOnly=True
        )

    def sl_limit(self, symbol, side, qty, sl_limit_price):
        """Place a stop-loss limit order."""
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=qty,
            price=sl_limit_price,
            stopPx=0.17,  # Stop trigger price
            triggerBy="LastPrice",
            timeInForce="PostOnly"
        )

    def get_mark_price(self, symbol):
        """Retrieve the mark price for a given symbol."""
        response = self.session.get_tickers(category="linear", symbol=symbol)
        if response["retCode"] == 0 and "result" in response:
            return float(response["result"]["list"][0]["markPrice"])
        return None  # Return None if request fails

    def get_position_risk(self, symbol):
        """Retrieve the position quantity and entry price."""
        response = self.session.get_positions(category="linear", symbol=symbol)
        if response["retCode"] == 0 and "result" in response:
            position_data = response["result"]["list"][0]  # Assuming single position per symbol
            return {
                "quantity": float(position_data["size"]),
                "entry_price": float(position_data["avgPrice"])
            }
        return None

    def cancel_all_orders(self, symbol):
        """Cancel all orders with proper error handling."""
        try:
            # Cancel active orders
            active_orders = self.session.cancel_all_orders(
                category="linear", 
                symbol=symbol,
                orderFilter="Order"
            )
            
            # Cancel conditional orders
            conditional_orders = self.session.cancel_all_orders(
                category="linear",
                symbol=symbol,
                orderFilter="StopOrder"
            )
            
            return {
                "success": True,
                "active": active_orders["retCode"] == 0,
                "conditional": conditional_orders["retCode"] == 0
            }
            
        except Exception as e:
            logging.error(f"Error canceling orders: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def close_position(self, symbol):
        """Close position with full error handling and consistent response format."""
        try:
            # 1. Cancel all orders first
            cancel_result = self.cancel_all_orders(symbol)
            if not cancel_result.get("success", False):
                return {
                    "retCode": 1,
                    "retMsg": f"Failed to cancel orders: {cancel_result.get('error', 'Unknown error')}"
                }

            # 2. Get position info
            position_data = self.session.get_positions(category="linear", symbol=symbol)
            if position_data["retCode"] != 0:
                return {
                    "retCode": position_data["retCode"],
                    "retMsg": position_data["retMsg"]
                }

            # 3. Check existing position
            for position in position_data["result"]["list"]:
                if position["symbol"] == symbol:
                    position_amt = float(position["size"])
                    if position_amt == 0:
                        return {
                            "retCode": 0,
                            "retMsg": "No position to close"
                        }
                    
                    # 4. Close position
                    side = "Sell" if position_amt > 0 else "Buy"
                    quantity = abs(position_amt)
                    
                    response = self.session.place_order(
                        category="linear",
                        symbol=symbol,
                        side=side,
                        orderType="Market",
                        qty=quantity,
                        reduceOnly=True
                    )
                    
                    # 5. Verify closure
                    if response["retCode"] == 0:
                        # Double-check position
                        verified = self.get_position_risk(symbol)
                        if verified and verified.get("quantity", 0) == 0:
                            return {
                                "retCode": 0,
                                "retMsg": "Position closed successfully"
                            }
                        return {
                            "retCode": 1,
                            "retMsg": "Position closure unverified"
                        }
                    
                    return {
                        "retCode": response["retCode"],
                        "retMsg": response["retMsg"]
                    }

            return {
                "retCode": 1,
                "retMsg": "Position not found"
            }
            
        except Exception as e:
            logging.error(f"Position close error: {str(e)}")
            return {
                "retCode": 1,
                "retMsg": str(e)
            }

async def bybit_main_script(symbol, occ_threshold, ohc_threshold, olc_threshold, position_duration, tp_percent=None, sl_percent=None):
    
    activation_message = (
    "🚀 *FT Bot v1.1 Activated!* 🚀\n"
    "🔹 Exchange: Bybit\n"
    f"🔹 Trading Pair: {symbol}\n"
    f"🔹 Metrics: {occ_threshold}% | {ohc_threshold}%\n"
    )
    send_telegram_message(activation_message)
    logging.info(activation_message)
    
    last_doge_close = None
    close_time = None
    long_signal_time = None
    entry_price = None  # Track entry price explicitly

    # WebSocket Configuration
    BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
    doge_trades = deque()

    # Initialize Bybit API
    bybit_api = BybitAPI(api_key="yA7NGrSFHzCZTYgeEH", api_secret="zhY4YcRxvAECA3tvHksx7DlSIYhH1EZBfhgX", testnet=False)

    # Add connection tracking
    active_connection = None
    connection_lock = asyncio.Lock()
    
    logging.info(f"⚡ BYBIT BOT ACTIVATED | Symbol: {symbol} | Thresholds: "
                f"OCC={occ_threshold}%, OHC={ohc_threshold}%, OLC={olc_threshold}")

    async def process_doge(data):
        """Process trade data from Bybit's WebSocket."""
        nonlocal last_doge_close
        ts = data["T"] / 1000
        price = round(float(data['p']), 5)
        if price != 0:
            doge_trades.append({'timestamp': ts, 'price': price})
            last_doge_close = price

    async def handle_message(message):
        """Handle WebSocket messages with error handling."""
        try:
            data = json.loads(message)
            if data.get('topic') == f"publicTrade.{symbol}":
                for trade in data['data']:
                    await process_doge({
                        "T": trade["T"],
                        "p": trade["p"]
                    })
        except Exception as e:
            logging.error(f"Message handling error: {str(e)}")

    async def websocket_listener():
        """Single-point WebSocket listener with connection control"""
        nonlocal active_connection
        reconnect_delay = 5
        
        while True:
            try:
                async with connection_lock:  # Prevent overlapping connections
                    async with websockets.connect(BYBIT_WS_URL) as ws:
                        active_connection = ws
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"publicTrade.{symbol}"]
                        }))
                        async for message in ws:
                            await handle_message(message)
            except Exception as e:
                logging.error(f"WS Error: {str(e)}")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay*2, 60)
            finally:
                active_connection = None

    async def evaluate_signals():
        """Core trading logic aligned with Binance version."""
        nonlocal last_doge_close, close_time, long_signal_time, entry_price
        interval = 3
        # Align to next 3-second interval
        next_run = time.time()
        next_run = next_run - (next_run % interval) + interval

        while True:
            now = time.time()
            await asyncio.sleep(max(next_run - now, 0))
            next_run += interval  # Set next run before processing

            # Process data
            cutoff_time = now - 15
            while doge_trades and doge_trades[0]['timestamp'] < cutoff_time:
                doge_trades.popleft()

            valid_trades = [t for t in doge_trades if t['price'] != 0]
            if not valid_trades:
                continue

            ohlc = {
                'open': valid_trades[0]['price'],
                'high': max(t['price'] for t in valid_trades),
                'low': min(t['price'] for t in valid_trades),
                'close': valid_trades[-1]['price']
            }
            last_doge_close = ohlc['close']

            # Calculate metrics
            occ = ((ohlc['close'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] else 0
            ohc = ((ohlc['high'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] else 0
            olc = ((ohlc['low'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] else 0

            logging.info(
                f"{symbol.upper()} - O: {ohlc['open']} H: {ohlc['high']} "
                f"L: {ohlc['low']} C: {ohlc['close']}"
            )
            logging.info(f"OCC: {occ:.3f}% | OHC: {ohc:.3f}% | OLC: {olc:.3f}%")

            if all([occ >= occ_threshold, ohc >= ohc_threshold, olc >= olc_threshold]):
                # Position management
                position_info = bybit_api.get_position_risk(symbol)
                position_size = position_info.get('quantity', 0) if position_info else 0
                has_position = position_size != 0

                if has_position:
                    # Update existing position
                    if long_signal_time and close_time:
                        new_close_time = datetime.datetime.utcnow() + datetime.timedelta(
                            seconds=position_duration * 15
                        )
                        if new_close_time > close_time:
                            logging.info("♻️ Updating position timing")
                            close_time = new_close_time
                            bybit_api.cancel_all_orders(symbol)
                            # Re-place orders
                            mark_price = bybit_api.get_mark_price(symbol)
                            if mark_price:
                                if tp_percent:
                                    tp_price = mark_price * (1 + (tp_percent/100))
                                    bybit_api.tp_limit(symbol, position_size, tp_price)
                                if sl_percent:
                                    sl_price = mark_price * (1 - (sl_percent/100))
                                    bybit_api.sl_market(symbol, "Sell", position_size, sl_price)
                else:
                    # Open new position
                    try:
                        response = bybit_api.market_buy(symbol, "Buy", 40)  # Adjust qty as needed
                        if response.get("retCode") == 0:
                            position_info = bybit_api.get_position_risk(symbol)
                            if position_info:
                                entry_price = position_info['entry_price']
                                position_size = position_info['quantity']
                                long_signal_time = datetime.datetime.utcnow()
                                close_time = long_signal_time + datetime.timedelta(
                                    seconds=position_duration * 15
                                )
                                
                                # Place orders
                                if tp_percent:
                                    tp_price = entry_price * (1 + (tp_percent/100))
                                    bybit_api.tp_limit(symbol, position_size, tp_price)
                                if sl_percent:
                                    sl_price = entry_price * (1 - (sl_percent/100))
                                    bybit_api.sl_market(symbol, "Sell", position_size, sl_price)

                                # Telegram notification
                                message = (
                                    f"🟢 *LONG OPENED* 🟢\n"
                                    f"• Pair: {symbol}\n"
                                    f"• Entry: `{entry_price:.5f}`\n"
                                    f"• Size: {position_size}\n"
                                    f"• Close: {close_time.strftime('%H:%M:%S UTC')}"
                                )
                                send_telegram_message(message)
                    except Exception as e:
                        logging.error(f"Order failed: {str(e)}")

            # Position closing logic
            if close_time and datetime.datetime.utcnow() >= close_time:
                # Close position
                close_response = bybit_api.close_position(symbol)
                bybit_api.cancel_all_orders(symbol)
                
                if close_response.get("retCode") == 0:
                    # Get PnL data
                    mark_price = bybit_api.get_mark_price(symbol)
                    if mark_price and entry_price:
                        pnl_value = (mark_price - entry_price) * 40  # Use actual position size
                        pnl_percent = ((mark_price - entry_price) / entry_price) * 100
                        duration = (datetime.datetime.utcnow() - long_signal_time).total_seconds() / 60
                        
                        message = (
                            f"🔴 *POSITION CLOSED* 🔴\n"
                            f"• Pair: {symbol}\n"
                            f"• Entry: `{entry_price:.5f}`\n"
                            f"• Exit: `{mark_price:.5f}`\n"
                            f"• PnL: {pnl_value:.2f} ({pnl_percent:.2f}%)\n"
                            f"• Duration: {duration:.1f} mins"
                        )
                        send_telegram_message(message)
                        
                close_time = None
                long_signal_time = None
                entry_price = None

    await asyncio.gather(
        websocket_listener(),
        evaluate_signals()
    )

async def process_websocket(ws, handler):
    """Process WebSocket messages."""
    async for msg in ws:
        data = json.loads(msg)
        await handler(data)

if __name__ == "__main__":
    asyncio.run(bybit_main_script(
        symbol="DOGEUSDT",
        occ_threshold=0.30,
        ohc_threshold=0.30,
        olc_threshold=-100.0,
        position_duration=4,
        tp_percent=None,
        sl_percent=0.05 #without minus sign
    ))