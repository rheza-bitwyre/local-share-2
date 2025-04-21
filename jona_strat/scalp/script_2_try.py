import requests
from datetime import datetime, timedelta
import time
import logging
import os

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7238226122:AAFf2OAEarJUpM6Bmd20RojCgPG4TdNYrIA"  # Replace with your bot token
TELEGRAM_CHAT_ID = "-4790253567"  # Replace with your group chat ID

# Logging Configuration
LOG_DIR = "/home/ubuntu/Rheza/local-share/jona_strat/scalp"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_filename = f"{LOG_DIR}/jona_strat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def send_telegram_message(message):
    """Send a message to a Telegram group using the bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"  # Use Markdown formatting
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to send Telegram message: {response.text}")
    else:
        logging.info("Telegram message sent successfully!")

def get_binance_server_time():
    """Fetch the current server time from Binance."""
    url = 'https://fapi.binance.com/fapi/v1/time'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch server time: {response.status_code} - {response.text}")
    return response.json()['serverTime']

def fetch_ohlc(symbol, interval, lookback_minutes):
    """Fetch OHLC data for the specified symbol and interval."""
    server_time = get_binance_server_time()
    server_time_dt = datetime.fromtimestamp(server_time / 1000)
    end_time = server_time_dt - timedelta(minutes=1)
    start_time = end_time - timedelta(minutes=lookback_minutes)
    
    url = (
        f'https://fapi.binance.com/fapi/v1/klines'
        f'?symbol={symbol}USDT&interval={interval}'
        f'&startTime={int(start_time.timestamp() * 1000)}'
        f'&endTime={int(end_time.timestamp() * 1000)}'
        f'&limit={lookback_minutes + 1}'
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
    
    ohlc_data = []
    for kline in response.json():
        ohlc_data.append({
            'time': datetime.fromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
        })
    return ohlc_data

def get_minute_info(
    symbol, 
    interval, 
    lookback_minutes, 
    reversal_candles=10,
    tp_percent=3,         # Adjustable TP percentage (default: 3%)
    sl_percent=3,         # Adjustable SL percentage (default: 3%)
    time_limit_candles=1500  # Adjustable time limit (default: 1500 candles)
):
    """Fetch OHLC data and implement sequence, reversal, confirmation, TP/SL, and time limit logic."""
    sequence = None
    breakout_close = None
    confirmation_candles = []
    reversal_level = None
    reversal_candle_count = 0
    second_confirmation_candles = []
    long_zone = None
    short_zone = None
    entry_triggered = False
    tp_sl_levels = {}
    confirmed_time = None
    historical_events = {}
    candle_counter = 0
    second_breakout_close = None

    while True:
        now = datetime.utcnow()
        time_to_wait = 60 - now.second + 1
        logging.info(f"Waiting {time_to_wait} seconds until next minute...")
        time.sleep(time_to_wait)
        time.sleep(5)  # 5-second delay

        ohlc_data = fetch_ohlc(symbol, interval, lookback_minutes)
        if len(ohlc_data) > lookback_minutes + 1:
            ohlc_data = ohlc_data[:lookback_minutes + 1]

        close_prices = [candle['close'] for candle in ohlc_data]
        max_close = max(close_prices[:-1])
        min_close = min(close_prices[:-1])
        newest_close = close_prices[-1]
        newest_candle = ohlc_data[-1]

        # --- Sequence Logic ---
        if sequence is None:
            if newest_close > max_close:
                sequence = "First Break Up Signal"
                historical_events['first_break'] = {'time': newest_candle['time'], 'level': max_close}
                breakout_close = newest_close
                confirmation_candles = []
                # Send Telegram Message for First Break Up Signal
                message = (
                    f"🚨 *{sequence}* 🚨\n"
                    f"Time: {newest_candle['time']}\n"
                    f"Level: {max_close}"
                )
                # send_telegram_message(message)
            elif newest_close < min_close:
                sequence = "First Break Down Signal"
                historical_events['first_break'] = {'time': newest_candle['time'], 'level': min_close}
                breakout_close = newest_close
                confirmation_candles = []
                # Send Telegram Message for First Break Down Signal
                message = (
                    f"🚨 *{sequence}* 🚨\n"
                    f"Time: {newest_candle['time']}\n"
                    f"Level: {min_close}"
                )
                # send_telegram_message(message)

        # --- First Signal Confirmation ---
        elif sequence in ["First Break Up Signal", "First Break Down Signal"]:
            confirmation_candles.append(newest_close)
            if len(confirmation_candles) == 3:
                condition_met = (
                    all(c >= breakout_close for c in confirmation_candles) if "Up" in sequence
                    else all(c <= breakout_close for c in confirmation_candles)
                )
                if condition_met:
                    max_min_3_candles = max(confirmation_candles) if "Up" in sequence else min(confirmation_candles)
                    sequence = "First Break Up Signal Confirmed" if "Up" in sequence else "First Break Down Signal Confirmed"
                    historical_events['first_break_confirmed'] = {'time': newest_candle['time'], 'level': max_min_3_candles}
                    # Send Telegram Notification for Renewed First Signal
                    message = (
                        f"✅ {sequence} ✅\n"
                        f"Time: {newest_candle['time']}\n"
                        f"Level: {breakout_close}\n"
                        f"Reversal limit reset."
                    )
                    # send_telegram_message(message)

                else:
                    sequence = None
                    historical_events = {}
                    confirmation_candles = []
                    logging.warning("First Signal failed: Next 3 candles invalid.")
                    
                    # Send Telegram Notification for Sequence Reset
                    message = (
                        f"❌ *Sequence Reset* ❌\n"
                        f"Reason: First Signal Confirmation failed.\n"
                        f"Time: {newest_candle['time']}"
                    )
                    # send_telegram_message(message)

        # --- Reversal Break ---
        elif sequence in ["First Break Up Signal Confirmed", "First Break Down Signal Confirmed"]:
            reversal_candle_count += 1
            reversal_condition = (
                (newest_close < min_close) if "Up" in sequence
                else (newest_close > max_close)
            )
            second_breakout_close = newest_close
            if reversal_condition:
                sequence = "Reversal Break Down" if "Up" in sequence else "Reversal Break Up"
                reversal_level = min_close if "Up" in sequence else max_close
                historical_events['reversal'] = {'time': newest_candle['time'], 'level': reversal_level}
                confirmation_candles = []
                # Send Telegram Message for Reversal Break
                message = (
                    f"🚨 *{sequence}* 🚨\n"
                    f"Time: {newest_candle['time']}\n"
                    f"Level: {reversal_level}"
                )
                # send_telegram_message(message)
            elif reversal_candle_count >= reversal_candles:
                sequence = None
                historical_events = {}
                reversal_candle_count = 0
                logging.warning("Reversal failed: Time limit exceeded.")
                
                # Send Telegram Notification for Sequence Reset
                message = (
                    f"❌ *Sequence Reset* ❌\n"
                    f"Reason: Reversal failed (time limit exceeded).\n"
                    f"Time: {newest_candle['time']}"
                )
                # send_telegram_message(message)

        # --- Second Break Confirmation ---
        elif sequence in ["Reversal Break Up", "Reversal Break Down"]:
            second_confirmation_candles.append(newest_close)
            if len(second_confirmation_candles) == 1:
                first_candle = second_confirmation_candles[0]
                condition_met = (
                    first_candle < second_breakout_close and 
                    all(c <= first_candle for c in second_confirmation_candles[1:]) and
                    first_candle < historical_events['first_break']['level'] 
                    if "Down" in sequence
                    else first_candle > second_breakout_close and 
                    all(c >= first_candle for c in second_confirmation_candles[1:]) and
                    first_candle > historical_events['first_break']['level'] 
                )
                if condition_met:
                    sequence = "Confirmed Break Down" if "Down" in sequence else "Confirmed Break Up"
                    confirmed_time = datetime.utcnow()
                    candle_counter = 0

                    # Define zones and TP/SL
                    if sequence == "Confirmed Break Down":
                        short_zone = {
                            'lower': historical_events['first_break']['level'], 
                            'upper': historical_events['first_break_confirmed']['level']
                        }
                        tp_sl_levels = {
                            'SL': short_zone['lower'] * (1 + sl_percent/100),  # Adjustable SL
                            'TP': short_zone['lower'] * (1 - tp_percent/100)   # Adjustable TP
                        }
                    else:
                        long_zone = {
                            'upper': historical_events['first_break']['level'], 
                            'lower': historical_events['first_break_confirmed']['level']
                        }
                        tp_sl_levels = {
                            'SL': long_zone['upper'] * (1 - sl_percent/100),  # Adjustable SL
                            'TP': long_zone['upper'] * (1 + tp_percent/100)   # Adjustable TP
                        }

                    # Calculate time limit end time
                    time_limit_end = confirmed_time + timedelta(minutes=time_limit_candles)

                    # Print Confirmation Details
                    logging.info(f"\n=== {sequence} CONFIRMED ===")
                    logging.info(f"First Break: {historical_events['first_break']['time']} (Level: {historical_events['first_break']['level']})")
                    logging.info(f"Reversal Break: {historical_events['reversal']['time']} (Level: {historical_events['reversal']['level']})")
                    logging.info(f"Zone: {long_zone if sequence == 'Confirmed Break Up' else short_zone}")
                    logging.info(f"TP: {tp_sl_levels['TP']:.2f}, SL: {tp_sl_levels['SL']:.2f}")
                    logging.info(f"Time Limit: {time_limit_candles} candles until {time_limit_end.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    # Send Telegram Message for Confirmation
                    message = (
                        f"🚨 *{sequence} CONFIRMED* 🚨\n"
                        f"First Break: {historical_events['first_break']['time']} (Level: {historical_events['first_break']['level']})\n"
                        f"Reversal Break: {historical_events['reversal']['time']} (Level: {historical_events['reversal']['level']})\n"
                        f"Zone: {long_zone if sequence == 'Confirmed Break Up' else short_zone}\n"
                        f"TP: {tp_sl_levels['TP']:.2f}, SL: {tp_sl_levels['SL']:.2f}\n"
                        f"Time Limit: {time_limit_candles} candles until {time_limit_end.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_message(message)

                    # Reset all
                    sequence = None
                    breakout_close = None
                    confirmation_candles = []
                    reversal_level = None
                    reversal_candle_count = 0
                    second_confirmation_candles = []
                    long_zone = None
                    short_zone = None
                    tp_sl_levels = {}
                    confirmed_time = None
                    historical_events = {}
                    second_breakout_close = None

                    
                else:
                    second_confirmation_candles.pop(0)  # Remove the first appended candle

        # # --- TP/SL and Time Limit Check ---
        # elif sequence in ["Confirmed Break Down", "Confirmed Break Up"]:
        #     candle_counter += 1

        #     # Check if price entered the zone
        #     if not entry_triggered:
        #         if sequence == "Confirmed Break Up":
        #             entry_triggered = newest_candle['low'] <= long_zone['upper']
        #         else:
        #             entry_triggered = newest_candle['high'] >= short_zone['lower']
        #         if entry_triggered:
        #             logging.info(f"Entry into zone triggered at {newest_candle['time']}")

        #     # Check TP/SL only if entry triggered
        #     if entry_triggered:
        #         if (sequence == "Confirmed Break Up" and (newest_close >= tp_sl_levels['TP'] or newest_close <= tp_sl_levels['SL'])) or \
        #            (sequence == "Confirmed Break Down" and (newest_close <= tp_sl_levels['TP'] or newest_close >= tp_sl_levels['SL'])):
        #             logging.info(f"{'TP' if newest_close >= tp_sl_levels['TP'] else 'SL'} Hit at {newest_candle['time']}!")
        #             sequence = None
        #             entry_triggered = False

        #     # Check time limit
        #     if datetime.utcnow() >= time_limit_end:
        #         logging.warning(f"Time Limit Reached: {time_limit_candles} candles expired.")
        #         sequence = None
        #         entry_triggered = False

        # --- Log Results ---
        logging.info(f"Time: {newest_candle['time']}")
        logging.info(f"Last Close: {newest_close}")
        logging.info(f"Prev Max {lookback_minutes} Close: {max_close}")
        logging.info(f"Prev Min {lookback_minutes} Close: {min_close}")
        logging.info(f"Sequence: {sequence}")
        # logging.info(f"Confirmed Signal: {sequence}")
        # logging.info(f"Entry Triggered: {entry_triggered}")
        # logging.info(f"Candle Counter: {candle_counter}/{time_limit_candles}")
        logging.info("-" * 40)

# Example Usage
if __name__ == "__main__":
    get_minute_info(
        symbol='BTC',
        interval='1m',
        lookback_minutes=100,
        reversal_candles=80,  # Wait 80 candles for reversal
        tp_percent=3,         # 3% TP
        sl_percent=3,         # 3% SL
        time_limit_candles=1500  # 1500-candle time limit
    )