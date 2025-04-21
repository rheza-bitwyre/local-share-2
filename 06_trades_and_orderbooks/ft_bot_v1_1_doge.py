import asyncio
import websockets
import json
import datetime
import requests
import hashlib
import urllib.parse
import hmac
import time
import logging
import os
from collections import deque
import numpy as np

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8150510214:AAHpr2CpTsArAJfz2Ro6URSedm_A5WsiMQo"
TELEGRAM_CHAT_ID = "-4665308596"

# Logging Setup
LOG_DIR = "/home/ubuntu/Rheza/local-share/06_trades_and_orderbooks"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/ft_bot_v1_1_doge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"

    def _generate_signature(self, params: dict) -> str:
        """Generate HMAC SHA256 signature for request."""
        query = urllib.parse.urlencode(params)
        return hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _send_request(self, method: str, endpoint: str, params=None, signed=False) -> dict:
        """Send request to Binance API."""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        if params is None:
            params = {}
        if signed:
            # Add timestamp in milliseconds
            params["timestamp"] = int(time.time() * 1000)
            logging.info(f"Generated timestamp: {params['timestamp']}")  # Log the timestamp
            # Generate signature
            params["signature"] = self._generate_signature(params)
        logging.info(f"Sending {method} request to {url} with params: {params}")  # Log the full request
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                # Send params as form data for POST requests
                response = requests.post(url, headers=headers, data=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Invalid method: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Capture the complete error response
            error_message = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_message = e.response.json()  # Try to parse the JSON error response
                except ValueError:
                    error_message = e.response.text  # Fallback to raw text if JSON parsing fails
            logging.error(f"API request failed: {error_message}")
            return {"error": error_message}  # Return the error message in a structured way

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price for a symbol."""
        return self._send_request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})

    def get_position_risk(self, symbol: str = None) -> dict:
        """Get position risk information."""
        params = {"symbol": symbol} if symbol else {}
        return self._send_request("GET", "/fapi/v3/positionRisk", params, signed=True)
    
    def get_position_size(self, symbol: str) -> float:
        """Fetch the current position size for a symbol."""
        positions = self.get_position_risk(symbol=symbol)
        if isinstance(positions, list) and len(positions) > 0:
            position = positions[0]
            return float(position.get('positionAmt', 0))
        return 0

    def create_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                    price: float = None, time_in_force: str = None, 
                    reduce_only: bool = False, stopPrice: float = None) -> dict:
        """Create a new order on Binance Futures."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": reduce_only,
        }

        # Handle STOP_MARKET orders
        if order_type == "STOP_MARKET":
            if stopPrice is None:
                raise ValueError("stopPrice is required for STOP_MARKET orders.")
            params["stopPrice"] = stopPrice  # Ensure correct spelling

        # Handle LIMIT orders
        if order_type == "LIMIT":
            if price is None or time_in_force is None:
                raise ValueError("Price and timeInForce are required for LIMIT orders.")
            params["price"] = price
            params["timeInForce"] = time_in_force

        return self._send_request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders for a symbol."""
        endpoint = "/fapi/v1/allOpenOrders"
        params = {"symbol": symbol}
        return self._send_request("DELETE", endpoint, params, signed=True)
    
    def close_position(self, symbol: str):
        """Close the active position for a specific symbol."""
        positions = self.get_position_risk()
        if positions is None:
            logging.error("Failed to retrieve position information.")
            return

        for position in positions:
            if position['symbol'] == symbol:
                position_amt = float(position['positionAmt'])

                if position_amt != 0:
                    side = 'SELL' if position_amt > 0 else 'BUY'
                    quantity = abs(position_amt)

                    response = self.create_order(
                        symbol=symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=quantity,
                        reduce_only=True
                    )

                    if response:
                        logging.info(f"Closed position: {symbol} with quantity {quantity}")
                    else:
                        logging.error(f"Failed to close position: {symbol}")
                else:
                    logging.info(f"No active position found for {symbol}.")
                return

        logging.error(f"No position information found for {symbol}.")

async def binance_ws(symbol, occ_threshold, ohc_threshold, olc_threshold, position_duration, tp_percent=None, sl_percent=None):
    
    activation_message = (
    "🚀 *FT Bot v1.1 Activated!* 🚀\n"
    "🔹 Exchange: Binance\n"
    f"🔹 Trading Pair: {symbol}\n"
    f"🔹 Metrics: {occ_threshold}% | {ohc_threshold}%\n"
    )
    send_telegram_message(activation_message)

    # Define last_doge_close in the parent scope
    last_doge_close = None
    close_time = None  # Initialize close_time in the parent scope
    long_signal_time = None  # Initialize long_signal_time in the parent scope

    # WebSocket URLs
    doge_ws = f"wss://fstream.binance.com/ws/{symbol.lower()}@trade"

    # Data stores
    doge_trades = deque()  # Stores raw trade data

    # Initialize Binance API
    binance_api = BinanceAPI(api_key="StZt5sP9jn4GIaBRohoAWJJ9y1QJIILOIFgjD5TuctFprI2DS2MRMX8zuBrpdWkm", api_secret="0hlhVwVcDvhlPwF8SbXXkAyWe8b7yFIwI23VdIiNJGDLpBJWugCgHgZ61FSk7Set")  # Replace with your API keys

    async with websockets.connect(doge_ws) as ws_doge:
        logging.info(f"⚡ BOT ACTIVATED | Symbol: {symbol.upper()} | Thresholds: "
                    f"OCC={occ_threshold}%, OHC={ohc_threshold}%, OLC={olc_threshold}")

        async def process_doge(data):
            """Process DOGE trades and store in deque."""
            nonlocal last_doge_close
            ts = data["T"] / 1000
            price = round(float(data['p']), 5)
            if price != 0:  # Only add trades with non-zero price
                doge_trades.append({'timestamp': ts, 'price': price})
                last_doge_close = price  # Always update the last close price

        async def evaluate_signals():
            """Evaluate trade signals every 3 seconds with precise position management."""
            nonlocal last_doge_close, close_time, long_signal_time
            interval = 3  # Strict 3-second interval
            next_run = time.time()

            while True:
                # Maintain precise timing
                now = time.time()
                if now < next_run:
                    await asyncio.sleep(next_run - now)
                    now = time.time()
                next_run = now + interval

                # Process DOGE data
                cutoff_time = now - 15
                while doge_trades and doge_trades[0]['timestamp'] < cutoff_time:
                    doge_trades.popleft()

                valid_doge_trades = [t for t in doge_trades if t['price'] != 0]
                ohlc = {
                    'open': valid_doge_trades[0]['price'] if valid_doge_trades else last_doge_close,
                    'high': max(t['price'] for t in valid_doge_trades) if valid_doge_trades else last_doge_close,
                    'low': min(t['price'] for t in valid_doge_trades) if valid_doge_trades else last_doge_close,
                    'close': valid_doge_trades[-1]['price'] if valid_doge_trades else last_doge_close
                } if valid_doge_trades or last_doge_close else None

                if not ohlc:
                    continue

                # Calculate metrics
                last_doge_close = ohlc['close']
                occ = ((ohlc['close'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] != 0 else 0
                ohc = ((ohlc['high'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] != 0 else 0
                olc = ((ohlc['low'] - ohlc['open']) / ohlc['open'] * 100) if ohlc['open'] != 0 else 0

                # Check conditions
                conditions_met = all([
                    occ >= occ_threshold,
                    ohc >= ohc_threshold,
                    olc >= olc_threshold
                ])

                logging.info(
                        f"{symbol.upper()} - Open: {ohlc['open']}, Close: {ohlc['close']}, "
                        f"High: {ohlc['high']}, Low: {ohlc['low']}"
                    )
                logging.info(f"OCC: {occ:.3f}% |OHC: {ohc:.3f}% | OLC: {olc:.3f}%")

                if conditions_met:

                    # Check Active Position
                    position = None
                    position_info = binance_api.get_position_risk(symbol=symbol)
                    if isinstance(position_info, list):
                        position = next((p for p in position_info if p['symbol'] == symbol), None)

                    position_size = float(position['positionAmt']) if position else 0
                    has_position = position_size != 0

                    if has_position:
                        # Update existing position
                        if long_signal_time and close_time:
                            new_close_time = datetime.datetime.utcnow() + datetime.timedelta(
                                seconds=position_duration * 15
                            )
                            if new_close_time > close_time:
                                logging.info("♻️ Updating existing position timing")
                                close_time = new_close_time
                                logging.info(f"⏱ Position will close at {new_close_time.strftime('%H:%M:%S UTC')}")
                                # Cancel and replace orders
                                binance_api.cancel_all_orders(symbol)
                                mark_price = float(binance_api.get_mark_price(symbol)['markPrice'])
                                await place_tp("BUY", symbol, mark_price, tp_percent)
                                await place_sl_market("BUY", symbol, mark_price, 40, sl_percent)
                    else:
                        # Open new position
                        logging.info("🚀 Attempting to open new long position")
                        order_response = await place_order("BUY", symbol)
                        
                        if order_response and 'orderId' in order_response:
                            long_signal_time = datetime.datetime.utcnow()
                            close_time = long_signal_time + datetime.timedelta(
                                seconds=position_duration * 15
                            )
                            logging.info(f"⏱ Position will close at {close_time.strftime('%H:%M:%S UTC')}")
                            
                            # Get confirmed entry price
                            position_info = binance_api.get_position_risk(symbol=symbol)
                            
                            if position_info:
                                position = binance_api.get_position_risk(symbol=symbol)[0]
                                entry_price = float(position['entryPrice'])
                                
                                # Place orders
                                await place_tp("BUY", symbol, entry_price, tp_percent)
                                await place_sl_market("BUY", symbol, entry_price, 40, sl_percent)

                                # In the position opening block:
                                quantity = abs(float(position['positionAmt']))

                                message = (
                                    f"🟢 *LONG POSITION OPENED* 🟢\n"
                                    f"• Symbol: {symbol}\n"
                                    f"• Entry Price: `{entry_price:.5f}`\n"
                                    f"• Quantity of Coin: {quantity:.0f}\n"
                                    f"• Scheduled Close: {close_time.strftime('%H:%M:%S UTC')}\n"
                                    f"• Timestamp: {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')}"
                                )
                                send_telegram_message(message)

                # Close position logic
                if close_time and datetime.datetime.utcnow() >= close_time:

                    # Execute closing
                    binance_api.close_position(symbol)
                    binance_api.cancel_all_orders(symbol)

                    # Check Active Position
                    position_info = binance_api.get_position_risk(symbol=symbol)
                    
                    if position_info :

                        logging.info("🕒 Closing position based on timing")
                        
                        # Get closing price (mark price)
                        mark_price_response = binance_api.get_mark_price(symbol=symbol)
                        close_price = float(mark_price_response['markPrice'])
                        
                        # Get entry price from position data
                        entry_price = float(position_info[0]['entryPrice'])
                        quantity = abs(float(position_info[0]['positionAmt']))
                        
                        # Calculate PnL
                        pnl_value = (close_price - entry_price) * quantity
                        pnl_percent = ((close_price - entry_price) / entry_price) * 100
                        
                        # Format message
                        duration = (datetime.datetime.utcnow() - long_signal_time).total_seconds() / 60  # in minutes
                        message = (
                            f"🔴 *POSITION CLOSED* 🔴\n"
                            f"• Symbol: {symbol}\n"
                            f"• Entry Price: `{entry_price:.5f}`\n"
                            f"• Exit Price: `{close_price:.5f}`\n"
                            f"• Quantity: {quantity:.0f}\n"
                            f"• Duration: {duration:.1f} mins\n"
                            f"• PnL: ${pnl_value:.2f} ({pnl_percent:.2f}%)\n"
                            f"• Timestamp: {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')}"
                        )
                        
                        # Send notification
                        send_telegram_message(message)
                    
                    # Reset tracking variables
                    close_time = None
                    long_signal_time = None

        async def place_order(side, symbol, quantity=40):
            """Place a new market order."""
            try:
                response = binance_api.create_order(
                    symbol=symbol,
                    side=side,
                    order_type="MARKET",
                    quantity=quantity
                )

                # Check if response contains an error
                if response is None or "error" in response:
                    error_msg = response.get("error", "No response from Binance API")
                    logging.error(f"Order execution failed: {error_msg}")
                    return None

                # Log the order execution
                if "orderId" in response:
                    logging.info(f"{side} order executed: {response}")
                    return response
                else:
                    logging.error(f"Order failed: {response}")
                    return None
            except Exception as e:
                logging.error(f"Order execution error: {str(e)}")
                return None


        async def place_tp(side, symbol, price, tp_percent):
            """Place or update take-profit order."""
            try:
                if tp_percent is None:
                    return

                # Calculate TP price
                tp_price = round(price * (1 + tp_percent / 100), 5)
                
                # Determine order side based on position direction
                order_side = "SELL" if side == "BUY" else "BUY"
                
                tp_response = binance_api.create_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="LIMIT",
                    quantity=40,
                    price=tp_price,
                    time_in_force="GTC",
                    reduce_only=True
                )
                
                if "orderId" in tp_response:
                    logging.info(f"📈 Take-profit order placed at {tp_price:.5f}: {tp_response}")
                else:
                    logging.error(f"❌ Take-profit order failed: {tp_response}")
                    
            except Exception as e:
                logging.error(f"🔥 TP placement error: {str(e)}")

        async def place_sl_market(side, symbol, price, quantity, sl_percent):
            """Place a stop-loss market order."""
            try:
                if sl_percent is None:
                    return

                # Calculate Stop-Loss price
                sl_price = price * (1 - (sl_percent + 0.02) / 100)

                # Determine order side
                order_side = "SELL" if side == "BUY" else "BUY"

                # Adjust decimal precision based on symbol
                decimal_precision = 5  # Default for most altcoins
                if symbol.endswith("USDT"):
                    decimal_precision = 2 if symbol.startswith("BTC") else 5  # BTC: 2 decimals, others: 5

                # Place stop-loss market order
                sl_response = binance_api.create_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="STOP_MARKET",
                    quantity=quantity,
                    reduce_only=True,
                    stopPrice=round(sl_price, decimal_precision)  # Dynamically adjust precision
                )

                if sl_response and "orderId" in sl_response:
                    print(f"📉 Stop-loss market order placed at {round(sl_price, decimal_precision)}: {sl_response}")
                else:
                    print(f"❌ Stop-loss market order failed: {sl_response}")

            except Exception as e:
                print(f"🔥 SL Market placement error: {str(e)}")

        # Run processing tasks
        await asyncio.gather(
            process_websocket(ws_doge, process_doge),
            evaluate_signals()
        )

async def process_websocket(ws, handler):
    """Process WebSocket messages."""
    async for msg in ws:
        data = json.loads(msg)
        await handler(data)

if __name__ == "__main__":
    asyncio.run(binance_ws(
        symbol="DOGEUSDT",
        occ_threshold=0.25,
        ohc_threshold=0.25,
        olc_threshold=-100.0,
        position_duration=4,
        tp_percent=1.0,
        sl_percent=0.05 #without minus sign
    ))