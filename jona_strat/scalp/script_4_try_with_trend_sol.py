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
log_filename = f"{LOG_DIR}/jona_strat_sol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def check_trend(trend_check_ohlc, lookback_minutes, trend_lookback):
    """
    Check the trend based on the trend_lookback candles, excluding the last 5 candles.
    Returns: "uptrend", "downtrend", or "no trend"
    """
    # Ensure there is enough data to determine the trend
    if len(trend_check_ohlc) < (lookback_minutes + trend_lookback + 10):
        logging.info("Not enough data to determine trend")
        return None  # Not enough data to determine trend

    # Extract close prices for the entire period (lookback_minutes + trend_lookback + 5)
    close_prices = [candle['close'] for candle in trend_check_ohlc]

    # Initialize flags for trend breaks
    has_high_break = False
    has_low_break = False

    # Iterate through the trend_lookback period, excluding the last 5 candles
    for i in range(lookback_minutes, lookback_minutes + trend_lookback):
        # Get the previous lookback_minutes high and low up to the current candle
        prev_high = max(close_prices[i - lookback_minutes:i])
        prev_low = min(close_prices[i - lookback_minutes:i])

        # Check if the current candle breaks the previous high or low
        current_close = close_prices[i]
        if current_close > prev_high:
            has_high_break = True
        if current_close < prev_low:
            has_low_break = True

    # Determine the trend based on breaks
    if not has_low_break:
        return "Up Trend"
    elif not has_high_break:
        return "Down Trend"
    else:
        return "No Trend"
    
def get_minute_info(
    symbol='SOL',
    interval='1m',
    lookback_minutes=100,
    reversal_candles=100,
    tp_percent=3,         # Adjustable TP percentage (default: 3%)
    sl_percent=3,         # Adjustable SL percentage (default: 3%)
    time_limit_candles=1500,  # Adjustable time limit (default: 1500 candles)
    first_confirmation_candles=3,
    second_confirmation_candles=2,
    trend_lookback = 200
):
    """Fetch OHLC data and implement sequence, reversal, confirmation, TP/SL, and time limit logic."""
    sequence = None
    confirmation_candles = []
    confirmation_max = []
    confirmation_min = []
    reversal_level = None
    reversal_candle_count = 0
    long_zone = None
    short_zone = None
    tp_sl_levels = {}
    confirmed_time = None
    historical_events = {}

    opening_message = "IM AWAKE, WAIT FOR MY SIGNAL NIGGA !!!"
    # send_telegram_message(opening_message)
    logging.info(opening_message)

    while True:
        now = datetime.utcnow()
        time_to_wait = 60 - now.second + 1
        logging.info(f"Waiting {time_to_wait} seconds until next minute...")
        time.sleep(time_to_wait)
        time.sleep(5)  # 5-second delay

        ohlc_data = fetch_ohlc(symbol, interval, lookback_minutes)
        if len(ohlc_data) > lookback_minutes + 1:
            ohlc_data = ohlc_data[:lookback_minutes + 1]

        # Check Trend
        total_lookback = lookback_minutes + trend_lookback + 10  # Fetch 10 additional candles
        trend_check_ohlc = fetch_ohlc(symbol, interval, total_lookback)
        trend = check_trend(trend_check_ohlc, lookback_minutes, trend_lookback)
        logging.info(f"Trend: {trend}")

        close_prices = [candle['close'] for candle in ohlc_data]
        max_close = max(close_prices[:-1])
        min_close = min(close_prices[:-1])
        newest_close = close_prices[-1]
        newest_candle = ohlc_data[-1]

        # --- SEQUENCE LOGIC ---

        # --- First Break Confirmation ---
        if sequence is None:
            # Append the newest candle data
            confirmation_candles.append(newest_close)
            confirmation_max.append(max_close)
            confirmation_min.append(min_close)

            logging.info(f"confirmation_candles: {confirmation_candles}")
            logging.info(f"confirmation_max: {confirmation_max}")
            logging.info(f"confirmation_min: {confirmation_min}")

            if newest_close > max_close :
                logging.info(f"Prev {lookback_minutes} Highest Close Passed !")
            elif newest_close < min_close :
                logging.info(f"Prev {lookback_minutes} Lowest Close Passed !")

            # Check if we have enough candles to confirm the break
            if len(confirmation_candles) == (first_confirmation_candles + 1):
                first_candle = confirmation_candles[0]
                first_max = confirmation_max[0]
                first_min = confirmation_min[0]

                logging.info(f"first_candle: {first_candle}")
                logging.info(f"first_max: {first_max}")
                logging.info(f"first_min: {first_min}")

                if first_candle > first_max and all(c >= first_candle for c in confirmation_candles) and trend == "Down Trend":
                    sequence = "First Break Up Confirmed"
                    logging.info(f"Sequence Turn to {sequence} !")
                elif first_candle < first_min and all(c <= first_candle for c in confirmation_candles) and trend == "Up Trend":
                    sequence = "First Break Down Confirmed"
                    logging.info(f"Sequence Turn to {sequence} !")
                else:
                    logging.info(f"Sequence doesn't change !")  
                
                if sequence:
                    broken_level = first_max if "Up" in sequence else first_min
                    max_min_candles = max(confirmation_candles) if "Up" in sequence else min(confirmation_candles)
                    historical_events['first_break'] = {'time': newest_candle['time'], 'level': broken_level}
                    historical_events['first_break_confirmed'] = {'time': newest_candle['time'], 'level': max_min_candles}

                    # Convert newest_candle['time'] to a datetime object
                    newest_candle_time = datetime.strptime(newest_candle['time'], '%Y-%m-%d %H:%M:%S')
                    reversal_limit_end = newest_candle_time + timedelta(minutes=reversal_candles)

                    message = (
                        f"{symbol}USDT - Futures\n"
                        f"✅ {sequence} ✅\n"
                        f"Time: {newest_candle['time']}\n"
                        f"Level: {max_min_candles}\n"
                        f"Reversal Time Limit: {reversal_limit_end.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    # send_telegram_message(message)
                    logging.info(message)

                    # Reset Lists
                    confirmation_candles = []
                    confirmation_max = []
                    confirmation_min = []
                    reversal_candle_count = 0

                else:
                    # Remove the first appended
                    confirmation_candles.pop(0)
                    confirmation_max.pop(0)
                    confirmation_min.pop(0)

        # --- Reversal Break ---
        elif sequence in ["First Break Up Confirmed", "First Break Down Confirmed"]:
            reversal_candle_count += 1 # Start the counter for the reversal time limit

            reversal_condition = None
            
            reversal_condition = (
                (newest_close < min_close) if "Up" in sequence
                else (newest_close > max_close)
            )

            if reversal_condition:
                sequence = "Reversal Break Down" if "Up" in sequence else "Reversal Break Up"
                reversal_level = newest_close
                historical_events['reversal'] = {'time': newest_candle['time'], 'level': reversal_level}

                message = (
                    f"{symbol}USDT - Futures\n"
                    f"🚨 *{sequence}* 🚨\n"
                    f"Time: {newest_candle['time']}\n"
                    f"Level: {reversal_level}"
                )
                # send_telegram_message(message)
                logging.info(message)

            elif reversal_candle_count >= reversal_candles:
                sequence = None
                historical_events = {}
                reversal_candle_count = 0
                logging.warning("Reversal failed: Time limit exceeded.")
                
                message = (
                    f"{symbol}USDT - Futures\n"
                    f"❌ *Sequence Reset* ❌\n"
                    f"Reversal failed (time limit exceeded).\n"
                    f"Time: {newest_candle['time']}"
                )
                # send_telegram_message(message)
                logging.info(message)

        # --- Second Break Confirmation ---
        elif sequence in ["Reversal Break Up", "Reversal Break Down"]:
            confirmation_candles.append(newest_close)
            
            if len(confirmation_candles) == (second_confirmation_candles + 1):
                first_candle = confirmation_candles[0]

                second_condition_met = None

                second_condition_met = (
                    first_candle > reversal_level and 
                    all(c >= first_candle for c in confirmation_candles) and
                    first_candle > historical_events['first_break']['level'] 
                    if "Up" in sequence
                    else first_candle < reversal_level and 
                    all(c <= first_candle for c in confirmation_candles) and
                    first_candle < historical_events['first_break']['level'] 
                )

                if second_condition_met:
                    sequence = "Confirmed Break Down" if "Down" in sequence else "Confirmed Break Up"
                    confirmed_time = datetime.utcnow()

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

                    # Send Telegram Message for Confirmation
                    message = (
                        f"{symbol}USDT - Futures\n"
                        f"✅ *{sequence}* ✅\n"
                        f"First Break: {historical_events['first_break']['time']} (Level: {historical_events['first_break']['level']})\n"
                        f"Reversal Break: {historical_events['reversal']['time']} (Level: {historical_events['reversal']['level']})\n"
                        f"Zone: {long_zone if sequence == 'Confirmed Break Up' else short_zone}\n"
                        f"TP: {tp_sl_levels['TP']:.5f}, SL: {tp_sl_levels['SL']:.5f}\n"
                        f"Time Limit: {time_limit_candles} candles until {time_limit_end.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_message(message)
                    logging.info(message)

                    # Reset all
                    sequence = None
                    confirmation_candles = []
                    confirmation_max = []
                    confirmation_min = []
                    reversal_level = None
                    reversal_candle_count = 0
                    long_zone = None
                    short_zone = None
                    tp_sl_levels = {}
                    confirmed_time = None
                    historical_events = {}
                    
                else:
                    if len(confirmation_candles) > 0 : # the length
                        # Remove the first appended
                        confirmation_candles.pop(0)

        # --- Log Results ---
        logging.info(f"Time: {newest_candle['time']}")
        logging.info(f"Last Close: {newest_close}")
        logging.info(f"Prev Max {lookback_minutes} Close: {max_close}")
        logging.info(f"Prev Min {lookback_minutes} Close: {min_close}")
        logging.info(f"Sequence: {sequence}")
        logging.info("-" * 40)

# Example Usage
if __name__ == "__main__":
    get_minute_info(
        symbol='SOL',
        interval='1m',
        lookback_minutes=100,
        reversal_candles=100,  # Wait 80 candles for reversal
        tp_percent=3,         # 3% TP
        sl_percent=3,         # 3% SL
        time_limit_candles=1500, # 1500-candle time limit
        first_confirmation_candles = 1,
        second_confirmation_candles = 0,
        trend_lookback = 150
    )