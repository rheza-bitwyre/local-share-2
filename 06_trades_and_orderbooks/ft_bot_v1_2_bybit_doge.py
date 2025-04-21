import asyncio
import websockets
import json
import datetime
import requests
import time
import logging
import os
from collections import deque
from pybit.unified_trading import HTTP

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8150510214:AAHpr2CpTsArAJfz2Ro6URSedm_A5WsiMQo"
TELEGRAM_CHAT_ID = "-4665308596"

# Logging Setup
LOG_DIR = "/home/ubuntu/Rheza/local-share/06_trades_and_orderbooks"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/bybit_ft_bot_v1_2_doge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def sl_market(self, symbol, side, qty, sl_market_price):
        """Proper trigger directions according to Bybit docs"""
        # Determine trigger direction based on position side
        trigger_dir = 2 if side == "Sell" else 1  # Long=2 (falling), Short=1 (rising)
        
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty,
            triggerPrice=sl_market_price,
            triggerBy="LastPrice",
            triggerDirection=trigger_dir,
            reduceOnly=True
        )

    def get_mark_price(self, symbol):
        """Retrieve the mark price for a given symbol."""
        response = self.session.get_tickers(category="linear", symbol=symbol)
        if response["retCode"] == 0 and "result" in response:
            return float(response["result"]["list"][0]["markPrice"])
        return None  # Return None if request fails

    def get_position_risk(self, symbol):
        """Retrieve position details including side"""
        response = self.session.get_positions(category="linear", symbol=symbol)
        if response["retCode"] == 0 and "result" in response:
            position_data = response["result"]["list"][0]
            return {
                "quantity": float(position_data["size"]),
                "entry_price": float(position_data["avgPrice"]),
                "side": position_data["side"].lower()  # Add this line
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
        """Close position with proper side detection"""
        try:
            position_info = self.get_position_risk(symbol)
            if not position_info or position_info.get('quantity', 0) == 0:
                return {"retCode": 0, "retMsg": "No position to close"}

            # Use actual position side from the position info
            close_side = "Buy" if position_info['side'] == "sell" else "Sell"
            
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=position_info['quantity'],
                reduceOnly=True
            )

            if response["retCode"] == 0:
                return {
                    "retCode": 0,
                    "retMsg": "Close order placed successfully",
                    "side": close_side
                }
            return response

        except Exception as e:
            logging.error(f"Position close error: {str(e)}")
            return {"retCode": 1, "retMsg": str(e)}

async def bybit_main_script(
    symbol,
    position_size_usdt,
    occ_long_threshold,
    ohc_long_threshold,
    olc_long_threshold,
    occ_short_threshold,
    ohc_short_threshold,
    olc_short_threshold,
    long_pos_dur,
    short_pos_dur,
    sl_long_pct=0.5,
    sl_short_pct=0.5,
):
    activation_message = (
        "🚀 *FT Bot v1.2 Activated!* 🚀\n"
        "🔹 Exchange: Bybit\n"
        f"🔹 Trading Pair: {symbol} - Amount: {position_size_usdt} $\n"
        f"🔹 Long Thresholds: {occ_long_threshold}% | {ohc_long_threshold}% | {olc_long_threshold}%\n"
        f"🔹 Short Thresholds: {occ_short_threshold}% | {ohc_short_threshold}% | {olc_short_threshold}%"
    )
    send_telegram_message(activation_message)
    logging.info(activation_message)

    # Position tracking variables
    position_type = None
    entry_price = None
    position_size = 0
    close_time = None
    sl_pct = None
    last_doge_close = None

    # WebSocket Configuration
    BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
    doge_trades = deque()

    # Initialize Bybit API
    bybit_api = BybitAPI(api_key="yA7NGrSFHzCZTYgeEH", api_secret="zhY4YcRxvAECA3tvHksx7DlSIYhH1EZBfhgX", testnet=False)

    # Add connection tracking
    active_connection = None
    connection_lock = asyncio.Lock()

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
        """WebSocket listener with reconnection logic"""
        nonlocal active_connection
        max_retries = 5  # Maximum number of connection attempts
        base_delay = 20  # Base delay in seconds between retries
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                async with connection_lock:
                    if retry_count != 0 :
                        logging.info(f"Attempting WebSocket connection (Attempt {retry_count}/{max_retries})")
                    async with websockets.connect(BYBIT_WS_URL) as ws:
                        active_connection = ws
                        retry_count = 0  # Reset retry counter on successful connection
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"publicTrade.{symbol}"]
                        }))
                        
                        async for message in ws:
                            await handle_message(message)
            
            except Exception as e:
                retry_count += 1
                error_msg = f"WebSocket connection failed: {str(e)}. Retrying in {base_delay} seconds ({retry_count}/{max_retries})"
                logging.error(error_msg)
                
                if retry_count >= max_retries:
                    critical_msg = "❗ Maximum WebSocket reconnection attempts reached. Stopping bot."
                    logging.error(critical_msg)
                    send_telegram_message(critical_msg)
                    raise  # Propagate the error to stop the bot
                    
                await asyncio.sleep(base_delay)
                
            finally:
                active_connection = None

    async def evaluate_signals():
        nonlocal last_doge_close, position_type, entry_price, close_time, position_size, sl_pct
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
            occ = ((ohlc['close'] - ohlc['open']) / ohlc['open']) * 100 if ohlc['open'] else 0
            ohc = ((ohlc['high'] - ohlc['open']) / ohlc['open']) * 100 if ohlc['open'] else 0
            olc = ((ohlc['low'] - ohlc['open']) / ohlc['open']) * 100 if ohlc['open'] else 0

            len_valid_trades = len(valid_trades)

            logging.info(
                f"{symbol.upper()} - O: {ohlc['open']} H: {ohlc['high']} "
                f"L: {ohlc['low']} C: {ohlc['close']}"
            )
            logging.info(f"OCC: {occ:.3f}% | OHC: {ohc:.3f}% | OLC: {olc:.3f}% | Data Points: {len_valid_trades}")    

            long_condition = False
            short_condition = False

            long_condition = all([
                occ >= occ_long_threshold,
                ohc >= ohc_long_threshold,
                olc >= olc_long_threshold
            ])

            short_condition = all([
                occ <= occ_short_threshold,
                ohc <= ohc_short_threshold,
                olc <= olc_short_threshold
            ])

            position_info = bybit_api.get_position_risk(symbol)
            if position_info:
                raw_side = position_info.get('side', '').lower()
                current_side = 'short' if raw_side == 'sell' else 'long' if raw_side == 'buy' else None
                has_position = position_info.get('quantity', 0) > 0
            else:
                current_side = None
                has_position = False

            logging.info(f"Position state: {has_position} | Side: {current_side} | Qty: {position_info.get('quantity') if position_info else 0}")
            logging.info(f"Long Signal: {long_condition} | Short Signal: {short_condition}")

            if long_condition:
                await handle_position(
                    condition_type='long',
                    current_side=current_side,
                    has_position=has_position,
                    position_info=position_info,
                    pos_duration=long_pos_dur,
                    sl_pct=sl_long_pct
                )
            elif short_condition:
                await handle_position(
                    condition_type='short',
                    current_side=current_side,
                    has_position=has_position,
                    position_info=position_info,
                    pos_duration=short_pos_dur,
                    sl_pct=sl_short_pct
                )
            # Position closing logic
            if close_time and datetime.datetime.utcnow() >= close_time:
                await close_current_position()

    async def handle_position(condition_type, current_side, has_position, position_info, pos_duration, sl_pct):
        nonlocal position_type, entry_price, close_time, position_size
        try:
            if has_position:
                if current_side and current_side.lower() == condition_type:
                    # Same position active
                    bybit_api.cancel_all_orders(symbol)
                    await update_orders(condition_type, position_info, 2, sl_pct)
                    new_close_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=pos_duration * 15) # Recalculate New Close Time
                    close_time = new_close_time # Update Close Time
                    
                else:
                    # Opposite position active
                    await close_current_position()
                    await open_new_position(condition_type, pos_duration)
                    await update_orders(condition_type, position_info, 1, sl_pct)
            else:
                # New position
                await open_new_position(condition_type, pos_duration)
                await update_orders(condition_type, position_info, 1, sl_pct)

        except Exception as e:
            logging.error(f"Position handling error: {str(e)}")

    async def open_new_position(pos_type, duration):
        nonlocal position_type, entry_price, close_time, position_size
        try:
            # Get current mark price
            mark_price = bybit_api.get_mark_price(symbol)
            if not mark_price:
                logging.error("Failed to get mark price")
                return

            # Calculate quantity based on USDT amount
            quantity = round(position_size_usdt / mark_price)  # Adjust rounding based on symbol requirements
            if quantity <= 0:
                logging.error(f"Invalid quantity: {quantity}")
                return

            if pos_type == 'long':
                response = bybit_api.market_buy(symbol, "Buy", quantity)
            elif pos_type == 'short':
                response = bybit_api.market_buy(symbol, "Sell", quantity)

            if response.get("retCode") == 0:
                await asyncio.sleep(1)  # Add delay for exchange update
                position_info = bybit_api.get_position_risk(symbol)
                if position_info:
                    position_type = pos_type
                    entry_price = position_info['entry_price']
                    position_size = position_info['quantity']
                    close_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=duration * 15)
                    
                    # Telegram notification
                    message = (
                        f"🟢 *{pos_type.upper()} OPENED* 🟢\n"
                        f"• Pair: {symbol}\n"
                        f"• Size: ${position_size_usdt} ({position_size} coins)\n"
                        f"• Entry: `{entry_price:.5f}`\n"
                        f"• Close: {close_time.strftime('%H:%M:%S UTC')}"
                    )
                    send_telegram_message(message)
                    logging.info(message)

        except Exception as e:
            logging.error(f"{pos_type} order failed: {str(e)}")

    async def update_orders(pos_type, position_info, ref, sl_pct):
        mark_price = bybit_api.get_mark_price(symbol)
        if not mark_price:
            return ("Mark Price Invalid!")

        if ref == 1 : #for new position
            ref_price = entry_price
        else : # for updating position
            ref_price = mark_price

        position_size = position_info['quantity']
        # Error Tolerance
        try:
            if pos_type == 'long':
                if sl_pct:
                    try:
                        sl_price = ref_price * (1 - (sl_pct/100))
                        bybit_api.sl_market(symbol, "Sell", position_size, sl_price)
                    except Exception as e:
                        logging.error(f"Long SL failed: {str(e)} - Continuing without SL")
                        
            elif pos_type == 'short':
                if sl_pct:
                    try:
                        sl_price = ref_price * (1 + (sl_pct/100))
                        bybit_api.sl_market(symbol, "Buy", position_size, sl_price)
                    except Exception as e:
                        logging.error(f"Short SL failed: {str(e)} - Continuing without SL")

        except Exception as e:
            logging.error(f"Order update failed: {str(e)}")

    async def close_current_position():
        nonlocal position_type, entry_price, close_time, position_size
        
        try:
            # Store closing values before any operations
            close_type = position_type
            close_entry = entry_price
            close_size = position_size
            position_start_time = close_time - datetime.timedelta(seconds=15) if close_time else None
            
            # 1. Verify position exists
            position_info = bybit_api.get_position_risk(symbol)
            if not position_info or float(position_info.get('quantity', 0)) <= 0:
                close_time = None
                logging.warning("No active position to close, Might hit SL already!")
                return {"retCode": 0, "retMsg": "No position found"}
            else :
                bybit_api.close_position(symbol)

            # 2. Get current mark price for PnL calculation
            current_mark = bybit_api.get_mark_price(symbol)
            if not current_mark:
                logging.error("Failed to get mark price")
                return {"retCode": 1, "retMsg": "Mark price unavailable"}
                
            # 5. Cancel all remaining orders
            cancel_result = bybit_api.cancel_all_orders(symbol)
            if not cancel_result.get("success"):
                logging.warning(f"Order cancellation issues: {cancel_result}")
                
            # 6. Calculate PnL and duration
            pnl_value = (current_mark - close_entry) * close_size
            if close_type == 'short':
                pnl_value = -pnl_value
                
            duration_min = ((datetime.datetime.utcnow() - position_start_time).total_seconds() / 60 
                        if position_start_time else 0)
            pnl_percent = (pnl_value / (close_entry * close_size)) * 100 if close_entry else 0
                
            # 7. Send closure notification
            message = (
                f"🔴 *POSITION CLOSED* 🔴\n"
                f"• Type: {close_type.upper()}\n"
                f"• Pair: {symbol}\n"
                f"• Entry: `{close_entry:.5f}`\n"
                f"• Exit: `{current_mark:.5f}`\n"
                f"• Size: {close_size} coins\n"
                f"• PnL: ${pnl_value:.2f} ({pnl_percent:.2f}%)\n"
                f"• Duration: {duration_min:.1f} mins"
            )
            send_telegram_message(message)
            logging.info(message)
                
            # 8. Reset tracking variables AFTER everything is done
            position_type = None
            entry_price = None
            close_time = None
            position_size = 0
                
            return {"retCode": 0, "retMsg": "Success", "pnl": pnl_value}
            
        except Exception as e:
            logging.error(f"Critical close error: {str(e)}", exc_info=True)
            return {"retCode": 1, "retMsg": f"System error: {str(e)}"}

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
        position_size_usdt=5.5,
        occ_long_threshold=0.3,
        ohc_long_threshold=0.3,
        olc_long_threshold=-100.0,
        occ_short_threshold=-0.3,
        ohc_short_threshold=100,
        olc_short_threshold=-0.3,
        long_pos_dur=20,
        short_pos_dur=4,
        sl_long_pct=0.1,
        sl_short_pct=0.1
    ))