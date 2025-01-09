import requests
import time
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import numpy as np
from pandas_ta.overlap import hl2
from pandas_ta.volatility import atr
from pandas_ta.utils import get_offset, verify_series
import hashlib
import urllib.parse
import hmac
import logging

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
                    print(f"Reached endtime at {datetime.fromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}. Stopping fetch.")
                    break

                print(f"{datetime.fromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')} "
                      f"status: {current_time < endtime}, time: {current_time}, calls: {call}")

            if len(timestamps) > 1 and timestamps[-1] == timestamps[-2]:
                print("Duplicate timestamp detected. Stopping fetch.")
                break

        current_df = pd.DataFrame(result_list, columns=BINANCE_CANDLE_COLUMNS)
        current_df['coin'] = coin
        current_df = current_df[['coin'] + BINANCE_CANDLE_COLUMNS]
        current_df = current_df.values.tolist()

        data_list += current_df
        call_dict.update({coin: call})

    return {'data': data_list, 'call': call_dict}

def fetch_and_append_data():
    # Get the latest opentime from the CSV file
    current_df = pd.read_csv('sol_usdt_data.csv')
    last_opentime = current_df['opentime'].iloc[-1]

    # Print the last opentime and the length of the CSV before appending
    print(f"Last opentime in CSV: {last_opentime}")
    print(f"CSV length before appending: {len(current_df)}")

    # Get the current Unix timestamp in seconds
    current_timestamp = int(time.time())

    # Round down to the nearest 30 minutes (1800 seconds)
    rounded_timestamp = current_timestamp - (current_timestamp % 1800)

    # Convert the timestamp back to milliseconds
    rounded_timestamp_ms = rounded_timestamp * 1000

    # Print the current nearest previous rounded timestamp in ms
    print(f"Rounded timestamp (previous 30 minutes) in ms: {rounded_timestamp_ms}")

    # Initialize new_row_count to 0 by default
    new_row_count = 0

    # Check if there is new data to fetch (if rounded_timestamp_ms > last_opentime)
    if rounded_timestamp_ms > last_opentime:
        # Fetch the next data (increment by 1800000 ms, or 30 minutes)
        data = binance_recursive_fetch_2(
            ['SOL'],
            '30m',
            starttime=int(last_opentime + 1800000),
            endtime=int(last_opentime + 3600000),
            data_type='futures'  # Fetch futures/sport
        )

        # Define the column names for the DataFrame based on the Binance API response structure
        columns = ['coin', 'opentime', 'openprice', 'highprice', 'lowprice', 'closeprice', 'volume', 'closetime', 
                   'quotevolume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'unused']

        # Convert the list of data into a DataFrame
        new_data = pd.DataFrame(data['data'], columns=columns)

        # Drop unnecessary columns
        new_data.drop(columns=['coin', 'volume', 'closetime', 'quotevolume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'unused'], inplace=True)

        # Convert columns 1-4 (openprice, highprice, lowprice, closeprice) to float
        new_data[['openprice', 'highprice', 'lowprice', 'closeprice']] = new_data[['openprice', 'highprice', 'lowprice', 'closeprice']].apply(pd.to_numeric, errors='coerce')

        # Check if there are new rows
        new_row_count = len(new_data)

        if new_row_count > 0:
            # Append the new data to the existing CSV
            new_data.to_csv('sol_usdt_data.csv', mode='a', header=False, index=False)
            
            # Print the number of new rows appended
            print(f"{new_row_count} new rows fetched and appended successfully.")
            # Print the new CSV length after appending
            current_df = pd.read_csv('sol_usdt_data.csv')
            print(f"New CSV length after appending: {len(current_df)}")
        else:
            print("No new data to append.")
    else:
        print("No new data available. The current timestamp is not greater than the last opentime.")

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
    
    return supertrend_df

def determine_suggested_action(df):

    # Get the last row of the DataFrame
    last_row = df.tail(1).copy()

    # Rename the last 4 columns for convenience
    new_column_names = {
        'SUPERTl_10_3.0': 'Up Trend',
        'SUPERTs_10_3.0': 'Down Trend',
        'leading_span_a': 'Leading Span A',
        'leading_span_b': 'Leading Span B'
    }
    last_row = last_row.rename(columns=new_column_names)

    # Extract scalar values
    up_trend = last_row['Up Trend'].iloc[0]
    down_trend = last_row['Down Trend'].iloc[0]
    closeprice = last_row['closeprice'].iloc[0]
    leading_span_a = last_row['Leading Span A'].iloc[0]
    leading_span_b = last_row['Leading Span B'].iloc[0]

    # Determine the suggested action
    if pd.notna(up_trend) and closeprice > leading_span_a and closeprice > leading_span_b:
        return 'Long'
    elif pd.notna(down_trend) and closeprice < leading_span_a and closeprice < leading_span_b:
        return 'Short'
    else:
        return None

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

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        reduce_only: bool = False,
    ) -> dict:
        """Create a new order"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": reduce_only,
        }

        if price:
            params["price"] = price

        return self._send_request("POST", "/fapi/v1/order", params, signed=True)

def handle_trading_action(suggested_action, prev_action=prev_action):

    print(f"Previous Action: {prev_action}")
    print(f"Suggested Action: {suggested_action}")
    
    # Initialize current action
    curr_action = None
    
    # Action handling logic
    if prev_action == suggested_action:
        pass  # Do nothing if the action is the same
    else:
        if prev_action is None and suggested_action == 'Long':
            curr_action = 'Open Long'
            # print(f'Open a Long Position: {curr_action}')
            print(curr_action)
            prev_action = 'Long'
        elif prev_action is None and suggested_action == 'Short':
            curr_action = 'Open Short'
            # print(f'Open a Short Position: {curr_action}')
            print(curr_action)
            prev_action = 'Short'
        elif prev_action == 'Long' and suggested_action == 'Short':
            curr_action = 'Close Long & Open Short'
            # print(f'Close all positions: {curr_action}')
            print(curr_action)
            prev_action = 'Short'
        elif prev_action == 'Long' and suggested_action is None:
            pass  # Do nothing
        elif prev_action == 'Short' and suggested_action == 'Long':
            curr_action = 'Close Short & Open Long'
            # print(f'Close all positions: {curr_action}')
            print(curr_action)
            prev_action = 'Long'
        elif prev_action == 'Short' and suggested_action is None:
            pass  # Do nothing

    # Print the result
    print(f"Current Action: {curr_action if curr_action else 'No action taken'}")
    print(f"New Previous Action: {prev_action}")
    
    return curr_action, prev_action

#Main
while True:
    prev_action = None
    # Fetch new data continuously
    while True:
        # Call the function to fetch and append the data
        fetch_and_append_data()
        if new_row_count >= 1:
            # Get the latest 51 data as the Ichimoku Cloud need 50 data
            df_sliced = pd.read_csv('sol_usdt_data.csv').tail(52)
            # Apply super trend indicator
            df_st = calculate_supertrend(df_sliced, length=10, multiplier=3.0)
            # Apply ichomoku cloud indicator
            df_st_ic = compute_ichimoku_with_supertrend(df_st)
            # Define action suggestion
            suggested_action = determine_suggested_action(df_st_ic)
            print(f"Suggested Action: {suggested_action}")
            # Define real action
            current_action, new_prev_action = handle_trading_action(suggested_action, prev_action)
         