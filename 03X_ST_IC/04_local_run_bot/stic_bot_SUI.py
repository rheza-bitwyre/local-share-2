import numpy as np
import sys

# Temporarily monkey-patch numpy to make NaN import work
sys.modules['numpy.NaN'] = np.nan

# Now proceed with your imports
import requests
import time
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pandas_ta.overlap import hl2
from pandas_ta.volatility import atr
from pandas_ta.utils import get_offset, verify_series
import hashlib
import urllib.parse
import hmac
import logging
import math
import json

# Load the configuration from the JSON file
with open("C:/Bitwyre/local-share-2/03X_ST_IC/04_local_run_bot/stic_binance_SUI_config.json", "r") as file:
    config = json.load(file)

# Access configuration values
API_KEY = config["API_KEY"]
API_SECRET = config["API_SECRET"]
trade_amount_usdt = config["trade_amount_usdt"]
symbol = config["symbol"]
position = config["position"]
path = config["path"]
log_filename = config["log_filename"]
csv_filename = config["csv_filename"]
long_tp = config["long_tp"]
long_sl = config["long_sl"]
short_tp = config["short_tp"]
short_sl = config["short_sl"]
f_symbol = symbol.replace('USDT', '')

# Get today's date and time in 'yyyymmddhhmmss' format
today_datetime = datetime.today().strftime('%Y%m%d%H%M%S')

# Logging Setup
logging.basicConfig(
    filename=f'{path}/{log_filename}_{today_datetime}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info(f"Trading symbol: {symbol}, Log file: {log_filename}_{today_datetime}")

def binance_recursive_fetch_2(coins, interval, starttime, endtime=None, data_type='futures'):

    # Define the column structure
    BINANCE_CANDLE_COLUMNS = ['opentime', 'openprice', 'highprice', 'lowprice', 'closeprice', 'volume', 'closetime',
                              'quotevolume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'unused']

    if endtime is None:
        endtime = int(time.time() * 1000)  # Current time in milliseconds

    all_coins_result = {}
    data_list = []
    call_dict = {}

    for coin in tqdm(coins):
        result_list = []
        current_time = starttime
        call = 0
        timestamps = []

        while current_time < endtime:
            limit = min(1000, int((endtime - current_time) / (1000 * 60)) + 1)
            
            if data_type == 'spot':
                url = (f'https://api.binance.com/api/v3/klines'
                       f'?symbol={coin}USDT'
                       f'&startTime={current_time}'
                       f'&interval={interval}'
                       f'&limit={limit}')
            elif data_type == 'futures':
                url = (f'https://fapi.binance.com/fapi/v1/klines'
                       f'?symbol={coin}USDT'
                       f'&startTime={current_time}'
                       f'&interval={interval}'
                       f'&limit={limit}')
            else:
                raise ValueError("Invalid data_type. Choose either 'spot' or 'futures'.")

            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            result_list += response.json()

            if result_list:
                current_time = result_list[-1][0] + 60000  # Update to the next timestamp
                timestamps.append(current_time)
                call += 1

                if current_time >= endtime:
                    logging.info(f"Reached endtime at {datetime.fromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}. Stopping fetch.")
                    break

                logging.info(f"{datetime.fromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')} "
                      f"status: {current_time < endtime}, time: {current_time}, calls: {call}")

            if len(timestamps) > 1 and timestamps[-1] == timestamps[-2]:
                logging.info("Duplicate timestamp detected. Stopping fetch.")
                break

        current_df = pd.DataFrame(result_list, columns=BINANCE_CANDLE_COLUMNS)
        current_df['coin'] = coin
        current_df = current_df[['coin'] + BINANCE_CANDLE_COLUMNS]
        current_df = current_df.values.tolist()

        data_list += current_df
        call_dict.update({coin: call})

    return {'data': data_list, 'call': call_dict}

def fetch_and_append_data(f_symbol):
    # Get the latest opentime from the CSV file
    try:
        current_df = pd.read_csv(f'{path}/{csv_filename}')
        last_opentime = current_df['opentime'].iloc[-1]
    except FileNotFoundError:
        logging.error("CSV file not found. Ensure the path is correct.")
        return 0
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return 0

    last_opentime_converted = datetime.utcfromtimestamp(last_opentime / 1000).strftime('%Y-%m-%d %H:%M:%S')

    # Log the last opentime and the length of the CSV before appending
    logging.info(f"Last opentime in CSV: {last_opentime_converted}")
    logging.info(f"CSV length before appending: {len(current_df)}")

    # Get the current Unix timestamp in seconds
    current_timestamp = int(time.time())

    # Round down to the nearest 30 minutes (1800 seconds)
    rounded_timestamp = current_timestamp - (current_timestamp % 1800)

    # Convert the timestamp back to milliseconds
    rounded_timestamp_ms = rounded_timestamp * 1000

    rounded_timestamp_ms_converted = datetime.utcfromtimestamp(rounded_timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

    # Log the rounded timestamp for comparison
    logging.info(f"Rounded timestamp: {rounded_timestamp_ms_converted}, Last opentime in CSV: {last_opentime_converted}")

    # Initialize new_row_count to 0 by default
    new_row_count = 0

    # Check if there is new data to fetch (if rounded_timestamp_ms > last_opentime)
    if rounded_timestamp_ms > last_opentime:
        # Fetch the next data (increment by 1800000 ms, or 30 minutes)
        try:
            data = binance_recursive_fetch_2(
                [f_symbol],
                '30m',
                starttime=int(last_opentime + 1800000),
                endtime=None,
                data_type='futures'  # Fetch futures/sport
            )

            print(f"Fetched data: {data}")
            # logging.info(f"Fetched data: {data}")

            # Check if the data exists
            if 'data' in data and data['data']:
                # Define the column names for the DataFrame based on the Binance API response structure
                columns = ['coin', 'opentime', 'openprice', 'highprice', 'lowprice', 'closeprice', 'volume', 'closetime',
                           'quotevolume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'unused']

                # Convert the list of data into a DataFrame
                new_data = pd.DataFrame(data['data'], columns=columns)

                # Drop unnecessary columns
                new_data.drop(columns=['coin', 'volume', 'closetime', 'quotevolume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'unused'], inplace=True)

                # Convert columns 1-4 (openprice, highprice, lowprice, closeprice) to float
                new_data[['openprice', 'highprice', 'lowprice', 'closeprice']] = new_data[['openprice', 'highprice', 'lowprice', 'closeprice']].apply(pd.to_numeric, errors='coerce')
                logging.info(f'before dropping :{len(new_data)}')

                # Check if there are new rows
                new_data = new_data.iloc[:-1]

                # Check if there are new rows
                new_row_count = len(new_data)
                logging.info(new_data)

                if new_row_count > 0:
                    # Append the new data to the existing CSV
                    new_data.to_csv(f'{path}/{csv_filename}', mode='a', header=False, index=False)
                    
                    # Log the number of new rows appended
                    logging.info(f"{new_row_count} new rows fetched and appended successfully.")

                    # Log the new CSV length after appending
                    current_df = pd.read_csv(f'{path}/{csv_filename}')
                    logging.info(f"New CSV length after appending: {len(current_df)}")
                    logging.info(f"New data: {current_df.tail(1)}")
                else:
                    logging.info("No new data to append.")
            else:
                logging.info("No new data fetched from Binance.")
        except Exception as e:
            logging.error(f"Error fetching data from Binance: {e}")
    else:
        logging.info("No new data available. The current timestamp is not greater than the last opentime.")

    return new_row_count

def calculate_supertrend(df, high_col='highprice', low_col='lowprice', close_col='closeprice', length=10, multiplier=3.0, offset=0, drop_columns=True, **kwargs):

    # Validate Arguments
    high = verify_series(df[high_col], length)
    low = verify_series(df[low_col], length)
    close = verify_series(df[close_col], length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return df

    m = close.size
    dir_, trend = [1] * m, [0] * m
    long, short = [np.nan] * m, [np.nan] * m

    hl2_ = hl2(high, low)
    matr = multiplier * atr(high, low, close, length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    for i in range(1, m):
        if close.iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband.iloc[i]
        else:
            trend[i] = short[i] = upperband.iloc[i]

    _props = f"_{length}_{multiplier}"
    supertrend_df = pd.DataFrame({
        f"SUPERT{_props}": trend,
        f"SUPERTd{_props}": dir_,
        f"SUPERTl{_props}": long,
        f"SUPERTs{_props}": short,
    }, index=close.index)

    # Apply offset if needed
    if offset != 0:
        supertrend_df = supertrend_df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        supertrend_df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        supertrend_df.fillna(method=kwargs["fill_method"], inplace=True)

    # Merge with original DataFrame
    result_df = df.join(supertrend_df)

    # Drop unnecessary columns
    if drop_columns:
        result_df.drop(columns=[f"SUPERT{_props}", f"SUPERTd{_props}"], inplace=True)

    logging.info(f"Last row: {result_df.tail(1)}")

    return result_df

def compute_ichimoku_with_supertrend(supertrend_df, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26):

    # Helper to calculate the average of the highest high and lowest low
    def donchian(data, period):
        return (data['highprice'].rolling(window=period).max() + 
                data['lowprice'].rolling(window=period).min()) / 2

    # Compute Ichimoku Cloud components
    supertrend_df['conversion_line'] = donchian(supertrend_df, conversion_periods)
    supertrend_df['base_line'] = donchian(supertrend_df, base_periods)
    supertrend_df['leading_span_a'] = ((supertrend_df['conversion_line'] + supertrend_df['base_line']) / 2).shift(displacement)
    supertrend_df['leading_span_b'] = donchian(supertrend_df, span_b_periods).shift(displacement)
    supertrend_df['lagging_span'] = supertrend_df['closeprice'].shift(-displacement)
    
    # Drop unnecessary columns
    supertrend_df.drop(columns=['conversion_line', 'base_line', 'lagging_span'], inplace=True)

    logging.info(f"Last row: {supertrend_df.tail(1)}")

    return supertrend_df

def determine_suggested_action(df,postion_option = 2, active_pos=0):
    # Get the last 2 rows of the DataFrame
    last_two_rows = df.tail(2).copy()

    # Rename the last 4 columns for convenience
    new_column_names = {
        'SUPERTl_10_3.0': 'Up Trend',
        'SUPERTs_10_3.0': 'Down Trend',
        'leading_span_a': 'Leading Span A',
        'leading_span_b': 'Leading Span B'
    }
    last_two_rows = last_two_rows.rename(columns=new_column_names)

    # Extract scalar values from the last row (latest row)
    up_trend_last = last_two_rows['Up Trend'].iloc[1]
    down_trend_last = last_two_rows['Down Trend'].iloc[1]
    closeprice_last = last_two_rows['closeprice'].iloc[1]
    leading_span_a_last = last_two_rows['Leading Span A'].iloc[1]
    leading_span_b_last = last_two_rows['Leading Span B'].iloc[1]

    # Extract scalar values from the second last row
    up_trend_second_last = last_two_rows['Up Trend'].iloc[0]

    logging.info(f"Previous up trend: {up_trend_second_last}")
    logging.info(f"Current up trend: {up_trend_last}")

    # Check if the trend has changed
    if pd.isna(up_trend_last) and pd.isna(up_trend_second_last):
        trend = 'unchange'
    elif not pd.isna(up_trend_last) and not pd.isna(up_trend_second_last):
        trend = 'unchange'
    else:
        trend = 'change'
        active_pos = 0
        logging.info(f"Active Position: {active_pos} - Reset!")

    logging.info(f"Trend: {trend}")
    logging.info(f"Active Position: {active_pos}")

    # Determine the suggested action
    suggested_action = None

    if postion_option == 2: # will open both open and short
        if trend == 'change':
            suggested_action = 'Close'
        elif trend == 'unchange' and pd.notna(up_trend_last) and closeprice_last > (leading_span_a_last if pd.notna(leading_span_a_last) else leading_span_b_last):
            suggested_action = 'Long'
        elif trend == 'unchange' and pd.notna(down_trend_last) and closeprice_last < (leading_span_a_last if pd.notna(leading_span_a_last) else leading_span_b_last):
            suggested_action = 'Short'

    elif postion_option == 1: # will only open long
        if trend == 'change':
            suggested_action = 'Close'
        elif trend == 'unchange' and pd.notna(up_trend_last) and closeprice_last > (leading_span_a_last if pd.notna(leading_span_a_last) else leading_span_b_last):
            suggested_action = 'Long'
            
    elif postion_option == -1: # will only open short
        if trend == 'change':
            suggested_action = 'Close'
        elif trend == 'unchange' and pd.notna(down_trend_last) and closeprice_last < (leading_span_a_last if pd.notna(leading_span_a_last) else leading_span_b_last):
            suggested_action = 'Short'

    logging.info(f"Suggested Action: {suggested_action}")

    return suggested_action, active_pos

import requests
import hashlib
import urllib.parse
import hmac
import time
import logging

import requests
import hashlib
import urllib.parse
import hmac
import time
import logging

class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key  
        self.api_secret = api_secret
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"

    def _generate_signature(self, params: dict) -> str:
        """Generate HMAC SHA256 signature for request"""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _send_request(
    self, method: str, endpoint: str, params: dict = None, signed: bool = False
) -> dict:
        """Send request to Binance FAPI"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}

        if params is None:
            params = {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Invalid method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            logging.error(f"Response: {e.response.text}")  # Log the response for debugging
            return None

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price for symbol"""
        params = {"symbol": symbol}
        return self._send_request("GET", "/fapi/v1/premiumIndex", params)

    def get_position_risk(self, symbol: str = None) -> dict:
        """Get position information"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._send_request("GET", "/fapi/v3/positionRisk", params, signed=True)

    def get_symbol_info(self, symbol: str) -> dict:
        """Fetch symbol information (e.g., tick size, lot size)."""
        endpoint = "/fapi/v1/exchangeInfo"
        response = self._send_request("GET", endpoint)
        if response and "symbols" in response:
            for s in response["symbols"]:
                if s["symbol"] == symbol:
                    return s
        return None

    def create_take_profit_order(self, symbol: str, side: str, take_profit_price: float) -> dict:
        """Create a take-profit order."""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Failed to fetch symbol info for {symbol}.")
            return None

        # Get tick size for stopPrice
        tick_size = float(next(filter(lambda f: f["filterType"] == "PRICE_FILTER", symbol_info["filters"]))["tickSize"])

        # Format stopPrice to adhere to tick size and round to 2 decimal places
        formatted_take_profit_price = round(take_profit_price / tick_size) * tick_size
        formatted_take_profit_price = round(formatted_take_profit_price, 2)  # Round to 2 decimal places

        order_type = "TAKE_PROFIT_MARKET"
        params = {
            "symbol": symbol,
            "side": "SELL" if side == "BUY" else "BUY",  # Opposite side for TP
            "type": order_type,
            "stopPrice": formatted_take_profit_price,  # Formatted stop price
            "closePosition": True,  # Close the entire position
            "workingType": "MARK_PRICE",  # Use mark price for stopPrice
            "priceProtect": True  # Enable price protection
        }

        logging.info(f"Creating take-profit order with params: {params}")
        return self._send_request("POST", "/fapi/v1/order", params, signed=True)

    def create_stop_loss_order(self, symbol: str, side: str, stop_loss_price: float) -> dict:
        """Create a stop-loss order."""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Failed to fetch symbol info for {symbol}.")
            return None

        # Get tick size for stopPrice
        tick_size = float(next(filter(lambda f: f["filterType"] == "PRICE_FILTER", symbol_info["filters"]))["tickSize"])

        # Format stopPrice to adhere to tick size and round to 2 decimal places
        formatted_stop_loss_price = round(stop_loss_price / tick_size) * tick_size
        formatted_stop_loss_price = round(formatted_stop_loss_price, 2)  # Round to 2 decimal places

        order_type = "STOP_MARKET"
        params = {
            "symbol": symbol,
            "side": "SELL" if side == "BUY" else "BUY",  # Opposite side for SL
            "type": order_type,
            "stopPrice": formatted_stop_loss_price,  # Formatted stop price
            "closePosition": True,  # Close the entire position
            "workingType": "MARK_PRICE",  # Use mark price for stopPrice
            "priceProtect": True  # Enable price protection
        }

        logging.info(f"Creating stop-loss order with params: {params}")
        return self._send_request("POST", "/fapi/v1/order", params, signed=True)

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        reduce_only: bool = False,
        time_in_force: str = None,
        stop_price: float = None,
        take_profit_percentage: float = None,  # New parameter for take-profit percentage
        stop_loss_percentage: float = None    # New parameter for stop-loss percentage
    ) -> dict:
        """Create a new order with optional take-profit and stop-loss orders."""
        # Create the initial order
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": reduce_only,
        }

        if price:
            params["price"] = price

        if stop_price:
            params["stopPrice"] = stop_price

        if order_type == "LIMIT" and time_in_force:
            params["timeInForce"] = time_in_force

        if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
            params["stopPrice"] = stop_price

        # Send the initial order
        response = self._send_request("POST", "/fapi/v1/order", params, signed=True)

        if response is None:
            logging.error("Failed to create the initial order.")
            return None

        # If the initial order is successful, create TP and SL orders
        if take_profit_percentage or stop_loss_percentage:
            # Get the current mark price to calculate TP and SL prices
            mark_price_response = self.get_mark_price(symbol)
            if mark_price_response is None:
                logging.error("Failed to fetch mark price for TP/SL calculation.")
                return response

            mark_price = float(mark_price_response['markPrice'])

            # Calculate TP and SL prices based on the percentage
            if take_profit_percentage:
                take_profit_price = mark_price * (1 + take_profit_percentage / 100) if side == "BUY" else mark_price * (1 - take_profit_percentage / 100)
                tp_response = self.create_take_profit_order(symbol, side, take_profit_price)
                if tp_response:
                    logging.info(f"Take-profit order created: {tp_response}")
                else:
                    logging.error("Failed to create take-profit order.")

            if stop_loss_percentage:
                stop_loss_price = mark_price * (1 + stop_loss_percentage / 100) if side == "BUY" else mark_price * (1 - stop_loss_percentage / 100)
                sl_response = self.create_stop_loss_order(symbol, side, stop_loss_price)
                if sl_response:
                    logging.info(f"Stop-loss order created: {sl_response}")
                else:
                    logging.error("Failed to create stop-loss order.")

        return response

    def close_all_position(self):
        """Close all active positions on the account"""
        positions = self.get_position_risk()
        if positions is None:
            logging.error("Failed to retrieve position information.")
            return

        for position in positions:
            symbol = position['symbol']
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

    def ping(self) -> dict:
        """Ping the Binance API to check connectivity."""
        return self._send_request("GET", "/fapi/v1/ping")

    def get_server_time(self) -> dict:
        """Get Binance server time to confirm connectivity."""
        return self._send_request("GET", "/fapi/v1/time")

    def check_balance(self):
        endpoint = "/fapi/v2/balance"
        params = {
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = self._generate_signature(params)
        return self._send_request("GET", endpoint, params=params)

def handle_trading_action(suggested_action, prev_action=None, trade_amount_usdt=100, symbol='SUIUSDT', API_KEY=None, API_SECRET=None, position = 2, active_pos = 1):
    # Initiate connection to Binance
    binance_api = BinanceAPI(api_key=API_KEY, api_secret=API_SECRET, testnet=False)

    logging.info(f"Previous Action: {prev_action}")
    logging.info(f"Suggested Action: {suggested_action}")
    logging.info(f"Active Position: {active_pos}")

    # Initiate variable
    curr_action = None

    # If previous action is the same as suggested action, no need to take any new action
    if prev_action == suggested_action:
        logging.info("No action needed as previous and suggested actions are the same.")
        return prev_action, active_pos
    else:
        # Get balance information
        balance = binance_api.check_balance()
        usdt_balance = next((float(item['availableBalance']) for item in balance if item['asset'] == 'USDT'), None)
        logging.info(f'USDT Available Balance: {usdt_balance}')
        logging.info(f'Determined Trade Amount: {trade_amount_usdt} USDT')

    # Get the current mark price
    mark_price = float(binance_api.get_mark_price(symbol).get('markPrice', 0))
    logging.info(f'Current {symbol} Mark Price: {mark_price}')

    # Calculate the position quantity
    coin_quantity = trade_amount_usdt / mark_price
    position_coin_amount = math.floor(coin_quantity) # round as per the coin rules
    logging.info(f'Expected {symbol} Amount: {position_coin_amount}')

    # Get current coin amount (position)
    get_postition_risk = binance_api.get_position_risk(symbol)

    if get_postition_risk :
        close_position_coin_amount = float(get_postition_risk[0]['positionAmt'])
        logging.info(f'Current Opened {symbol} Amount: {close_position_coin_amount}')

    if position == 2: # long and short
        if prev_action is None:
            if suggested_action == 'Long' and active_pos == 0:
                curr_action = 'Open Long'
                binance_api.create_order(symbol, "BUY", "MARKET", position_coin_amount, take_profit_percentage= long_tp, stop_loss_percentage= long_sl)
                prev_action = 'Long'
                active_pos = 1
            elif suggested_action == 'Short' and active_pos == 0:
                curr_action = 'Open Short'
                binance_api.create_order(symbol, "SELL", "MARKET", position_coin_amount, take_profit_percentage=short_tp, stop_loss_percentage=short_sl)
                prev_action = 'Short'
                active_pos = 1
        elif prev_action == 'Long' and suggested_action == 'Close':
            curr_action = 'Close Long'
            binance_api.create_order(symbol, "SELL", "MARKET", close_position_coin_amount)
            prev_action = None
        elif prev_action == 'Short' and suggested_action == 'Close':
            curr_action = 'Close Short'
            binance_api.create_order(symbol, "BUY", "MARKET", -close_position_coin_amount)
            prev_action = None
        elif prev_action == 'Long' and suggested_action == 'Short':
            curr_action = 'Close Long & Open Short'
            binance_api.create_order(symbol, "SELL", "MARKET", close_position_coin_amount)
            binance_api.create_order(symbol, "SELL", "MARKET", position_coin_amount, take_profit_percentage=short_tp, stop_loss_percentage=short_sl)
            prev_action = 'Short'
            active_pos = 1
        elif prev_action == 'Short' and suggested_action == 'Long':
            curr_action = 'Close Short & Open Long'
            binance_api.create_order(symbol, "BUY", "MARKET", -close_position_coin_amount)
            binance_api.create_order(symbol, "BUY", "MARKET", position_coin_amount,take_profit_percentage= long_tp, stop_loss_percentage= long_sl)
            prev_action = 'Long'
            active_pos = 1

    if position == 1: # only long
        if prev_action is None:
            if suggested_action == 'Long' and active_pos == 0:
                curr_action = 'Open Long'
                binance_api.create_order(symbol, "BUY", "MARKET", position_coin_amount,take_profit_percentage= long_tp, stop_loss_percentage= long_sl)
                prev_action = 'Long'
                active_pos = 1
        elif prev_action == 'Long' and suggested_action == 'Close':
            curr_action = 'Close Long'
            binance_api.create_order(symbol, "SELL", "MARKET", close_position_coin_amount)
            prev_action = None
        elif prev_action == 'Long' and suggested_action == 'Short':
            curr_action = 'Close Long'
            binance_api.create_order(symbol, "SELL", "MARKET", close_position_coin_amount)
            prev_action = None
        elif prev_action == 'Short' and suggested_action == 'Long':
            curr_action = 'Open Long'
            binance_api.create_order(symbol, "BUY", "MARKET", position_coin_amount,take_profit_percentage= long_tp, stop_loss_percentage= long_sl)
            prev_action = 'Long'
            active_pos = 1

    if position == -1: # only short
        if prev_action is None:
            if suggested_action == 'Short' and active_pos == 0:
                curr_action = 'Open Short'
                binance_api.create_order(symbol, "SELL", "MARKET", position_coin_amount, take_profit_percentage=short_tp, stop_loss_percentage=short_sl)
                prev_action = 'Short'
                active_pos = 1
        elif prev_action == 'Short' and suggested_action == 'Close':
            curr_action = 'Close Short'
            binance_api.create_order(symbol, "BUY", "MARKET", -close_position_coin_amount)
            prev_action = None
        elif prev_action == 'Long' and suggested_action == 'Short':
            curr_action = 'Open Short'
            binance_api.create_order(symbol, "SELL", "MARKET", position_coin_amount, take_profit_percentage=short_tp, stop_loss_percentage=short_sl)
            prev_action = 'Short'
            active_pos = 1
        elif prev_action == 'Short' and suggested_action == 'Long':
            curr_action = 'Close Short'
            binance_api.create_order(symbol, "BUY", "MARKET", -close_position_coin_amount)
            prev_action = None

    # Handle CSV updates and timestamp conversion outside of main logic to optimize
    new_df = pd.read_csv(f'{path}/{csv_filename}')
    latest_opentime = new_df['opentime'].iloc[-1]
    converted_opentime = datetime.utcfromtimestamp(latest_opentime / 1000).strftime('%Y-%m-%d %H:%M:%S')

    logging.info(f"at: {converted_opentime} - {curr_action if curr_action else 'No action taken'}")

    return prev_action, active_pos

########################################################################### 
# Main Loop
def main():
    # Initialize connection to Binancee
    binance_api = BinanceAPI(api_key=API_KEY, api_secret=API_SECRET, testnet=False)

    # Check connectivity with ping
    ping_response = binance_api.ping()
    logging.info("Ping Response:", ping_response)  # Should return an empty dictionary if successful

    # Check server time
    server_time = binance_api.get_server_time()
    logging.info("Server Time:", server_time)  # Should return the server time in milliseconds

    # Get balance information
    balance = binance_api.check_balance()
    usdt_balance = next((float(item['availableBalance']) for item in balance if item['asset'] == 'USDT'), None)
    logging.info(f'USDT Available Balance: {usdt_balance}')

    # Determine Trade Amount
    logging.info(f'Determined Trade Amount: {trade_amount_usdt} USDT')

    # Get the initial position (if any) for the first action
    prev_action = None
    active_pos = 1
    logging.info(f'Active Position Initiation: {active_pos}')

    gpr = binance_api.get_position_risk(symbol=symbol)

    if gpr:
        active_pos = 1
        if float(gpr[0]['entryPrice']) < float(gpr[0]['breakEvenPrice']):
            prev_action = 'Long'
        elif float(gpr[0]['entryPrice']) > float(gpr[0]['breakEvenPrice']):
            prev_action = 'Short'
    else :
        # Get balance information
        balance = binance_api.check_balance()
        usdt_balance = next((float(item['availableBalance']) for item in balance if item['asset'] == 'USDT'), None)
        logging.info(f'USDT Available Balance: {usdt_balance}')

        # Determine Trade Amount
        logging.info(f'Determined Trade Amount: {trade_amount_usdt} USDT')

    while True:
        new_row_count = fetch_and_append_data(f_symbol)

        if new_row_count >= 1:
            # Log the new data fetch
            logging.info('New data fetched, processing now.')

            # Get the latest 51 data as the Ichimoku Cloud needs 50 data
            df_sliced = pd.read_csv(f'{path}/{csv_filename}').tail(52)

            # Apply super trend indicator
            df_st = calculate_supertrend(df_sliced, length=10, multiplier=3.0)
            logging.info('Supertrend indicator applied.')

            # Apply Ichimoku cloud indicator
            df_st_ic = compute_ichimoku_with_supertrend(df_st)
            logging.info('Ichimoku cloud indicator applied.')

            # Define action suggestion
            suggested_action, active_pos = determine_suggested_action(df_st_ic)
            logging.info(f'Suggestion : {suggested_action}, Previous ACtion {prev_action}, Active Position: {active_pos}')

            # Define real action and log it
            prev_action, active_pos = handle_trading_action(suggested_action, prev_action, trade_amount_usdt, symbol, API_KEY, API_SECRET, position, active_pos)                                                       

        # Sleep for 1 minute before starting the next iteration of the inner loop
        time.sleep(60)

# Run the main function
if __name__ == "__main__":
    main()      