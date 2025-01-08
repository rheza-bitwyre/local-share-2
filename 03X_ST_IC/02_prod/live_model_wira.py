import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import torch
from chronos import ChronosPipeline
import hmac
import hashlib
import requests
import urllib.parse
import logging
from typing import Dict, List, Tuple
import json
import os
from decimal import Decimal
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("stat_arb.log"), logging.StreamHandler()],
)


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

    def get_exchange_info(self) -> dict:
        """Get exchange information"""
        return self._send_request("GET", "/fapi/v1/exchangeInfo")

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 60) -> List:
        """Get kline/candlestick data"""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._send_request("GET", "/fapi/v1/klines", params)

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price for symbol"""
        params = {"symbol": symbol}
        return self._send_request("GET", "/fapi/v1/premiumIndex", params)

    def get_position_risk(self, symbol: str = None) -> List[dict]:
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

class BinanceStatArb:
    def __init__(self, config: dict):
        """Initialize the bot with configuration"""
        self.config = config
        self.api = BinanceAPI(
            config["api_key"], config["api_secret"], config.get("testnet", False)
        )

        self.window = config.get("window", 60)
        self.zscore_threshold = config.get("zscore_threshold", 1.5)
        self.prediction_length = config.get("prediction_length", 25)
        self.trade_amount_usdt = config.get("trade_amount_usdt", 1000)
        self.leverage = config.get("leverage", 1)

        self.pairs = [tuple(pair) for pair in config["pairs"]]

        self.positions = {pair: 0 for pair in self.pairs}
        self.entry_prices = {pair: {"price1": 0, "price2": 0} for pair in self.pairs}
        self.position_values = {pair: 0 for pair in self.pairs}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny", device_map=self.device, torch_dtype=torch.bfloat16
        )

        self.symbol_info = {}
        self._load_symbol_info()

    def _load_symbol_info(self):
        """Load symbol information and precision requirements"""
        exchange_info = self.api.get_exchange_info()
        if exchange_info:
            for symbol_info in exchange_info["symbols"]:
                self.symbol_info[symbol_info["symbol"]] = {
                    "quantityPrecision": symbol_info["quantityPrecision"],
                    "pricePrecision": symbol_info["pricePrecision"],
                    "filters": {f["filterType"]: f for f in symbol_info["filters"]},
                }

    def round_step_size(self, quantity: float, step_size: float) -> float:
        """Round quantity to step size"""
        precision = int(round(-math.log10(step_size)))
        return float(Decimal(str(quantity)).quantize(Decimal(str(step_size))))

    def get_historical_klines(self, symbol: str) -> np.ndarray:
        """Get historical kline data for a symbol"""
        klines = self.api.get_klines(symbol, limit=self.window)
        if klines:
            closes = np.array([float(k[4]) for k in klines])
            return closes
        return np.array([])

    def calculate_zscore(self, spread_values: np.ndarray) -> float:
        """Calculate z-score for the spread"""
        if len(spread_values) < self.window:
            return 0
        recent_spread = spread_values[-self.window :]
        rolling_mean = np.mean(recent_spread)
        rolling_std = np.std(recent_spread)
        if rolling_std == 0:
            return 0
        current_spread = spread_values[-1]
        return (current_spread - rolling_mean) / rolling_std

    def predict_spread(self, spread_data: np.ndarray) -> int:
        """Predict spread direction using the ML model"""
        try:
            if len(spread_data) < self.window:
                logging.warning(
                    f"Insufficient data for prediction: {len(spread_data)} points"
                )
                return 0

            spread_values = spread_data[-self.window :]
            context = torch.tensor(spread_values, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                forecast = self.pipeline.predict(
                    context, prediction_length=self.prediction_length, num_samples=10
                )

                last_value = spread_values[-1]
                pred_mean = forecast.mean(axis=1)[0][0].item()

                logging.info(
                    f"Prediction - Last: {last_value:.4f}, Predicted: {pred_mean:.4f}"
                )
                return 1 if pred_mean > last_value else -1

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return 0

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity according to symbol's precision requirements"""
        try:
            info = self.symbol_info.get(symbol)
            if not info:
                logging.error(f"No symbol info found for {symbol}")
                return 0

            lot_size = info["filters"].get("LOT_SIZE")
            if not lot_size:
                logging.error(f"No LOT_SIZE filter found for {symbol}")
                return 0

            min_qty = float(lot_size["minQty"])
            max_qty = float(lot_size["maxQty"])
            step_size = float(lot_size["stepSize"])

            precision = int(round(-math.log10(step_size)))

            if step_size < 1:
                rounded_quantity = round(quantity * (1 / step_size)) / (1 / step_size)
            else:

                rounded_quantity = math.floor(quantity)

            rounded_quantity = max(min_qty, min(rounded_quantity, max_qty))

            logging.info(
                f"Original quantity: {quantity}, Rounded quantity: {rounded_quantity} for {symbol}, "
                f"Step size: {step_size}, Precision: {precision}"
            )
            return rounded_quantity

        except Exception as e:
            logging.error(f"Error rounding quantity for {symbol}: {e}")
            return 0

    def open_positions(
        self, pair: Tuple[str, str], direction: int, price1: float, price2: float
    ):
        """Open positions for a pair"""
        try:
            symbol1, symbol2 = pair

            if price1 <= 0 or price2 <= 0:
                logging.warning(f"Invalid prices: {price1}, {price2} for pair {pair}")
                return

            info1 = self.symbol_info.get(symbol1)
            info2 = self.symbol_info.get(symbol2)

            if not info1 or not info2:
                logging.error(f"Missing symbol info for {symbol1} or {symbol2}")
                return

            min_notional1 = float(info1["filters"]["MIN_NOTIONAL"]["notional"])
            min_notional2 = float(info2["filters"]["MIN_NOTIONAL"]["notional"])

            raw_quantity1 = self.trade_amount_usdt / price1
            raw_quantity2 = self.trade_amount_usdt / price2

            quantity1 = self.round_quantity(symbol1, raw_quantity1)
            quantity2 = self.round_quantity(symbol2, raw_quantity2)

            value1 = quantity1 * price1
            value2 = quantity2 * price2

            if value1 < min_notional1 or value2 < min_notional2:
                logging.warning(
                    f"Trade values below minimum notional: {symbol1}={value1}, {symbol2}={value2}"
                )
                return

            value_ratio = max(value1, value2) / min(value1, value2)
            if value_ratio > 1.2:
                logging.warning(
                    f"Trade values too imbalanced: {symbol1}={value1}, {symbol2}={value2}, ratio={value_ratio}"
                )
                return

            logging.info(f"Opening positions for {pair}")
            logging.info(
                f"Symbol1: {symbol1}, Direction: {'BUY' if direction == 1 else 'SELL'}, "
                f"Quantity: {quantity1}, Value: {value1}"
            )
            logging.info(
                f"Symbol2: {symbol2}, Direction: {'SELL' if direction == 1 else 'BUY'}, "
                f"Quantity: {quantity2}, Value: {value2}"
            )

            if direction == 1:
                order1 = self.api.create_order(symbol1, "BUY", "MARKET", quantity1)
                if not order1:
                    logging.error(f"Failed to create order for {symbol1}")
                    return
                order2 = self.api.create_order(symbol2, "SELL", "MARKET", quantity2)
                if not order2:

                    self.api.create_order(
                        symbol1, "SELL", "MARKET", quantity1, reduce_only=True
                    )
                    logging.error(
                        f"Failed to create order for {symbol2}, reversed {symbol1} position"
                    )
                    return
            else:
                order1 = self.api.create_order(symbol1, "SELL", "MARKET", quantity1)
                if not order1:
                    logging.error(f"Failed to create order for {symbol1}")
                    return
                order2 = self.api.create_order(symbol2, "BUY", "MARKET", quantity2)
                if not order2:
                    self.api.create_order(
                        symbol1, "BUY", "MARKET", quantity1, reduce_only=True
                    )
                    logging.error(
                        f"Failed to create order for {symbol2}, reversed {symbol1} position"
                    )
                    return

            self.positions[pair] = direction
            self.entry_prices[pair] = {"price1": price1, "price2": price2}
            self.position_values[pair] = (value1 + value2) / 2
            logging.info(
                f"Successfully opened {direction} position for {pair[0]}-{pair[1]}"
            )

        except Exception as e:
            logging.error(f"Error opening positions for {pair}: {e}")

    def close_positions(self, pair: Tuple[str, str], price1: float, price2: float):
        """Close positions for a pair"""
        try:
            symbol1, symbol2 = pair
            entry_price1 = self.entry_prices[pair]["price1"]
            entry_price2 = self.entry_prices[pair]["price2"]
            position = self.positions[pair]

            if position == 1:
                pnl = (
                    (price1 - entry_price1) / entry_price1
                    - (price2 - entry_price2) / entry_price2
                ) * self.trade_amount_usdt
            else:
                pnl = (
                    (entry_price1 - price1) / entry_price1
                    - (entry_price2 - price2) / entry_price2
                ) * self.trade_amount_usdt

            logging.info(
                f"Closing position for {pair[0]}-{pair[1]}, PnL: {pnl:.2f} USDT"
            )

            pos_info1 = self.api.get_position_risk(symbol1)
            pos_info2 = self.api.get_position_risk(symbol2)

            pos1 = int(float(pos_info1[0]["positionAmt"])) if pos_info1 else 0
            pos2 = int(float(pos_info2[0]["positionAmt"])) if pos_info2 else 0

            if pos1 != 0:
                self.api.create_order(
                    symbol1,
                    "SELL" if pos1 > 0 else "BUY",
                    "MARKET",
                    abs(pos1),
                    reduce_only=False,
                )

            if pos2 != 0:
                self.api.create_order(
                    symbol2,
                    "SELL" if pos2 > 0 else "BUY",
                    "MARKET",
                    abs(pos2),
                    reduce_only=False,
                )

            self.positions[pair] = 0
            self.entry_prices[pair] = {"price1": 0, "price2": 0}
            self.position_values[pair] = 0

        except Exception as e:
            logging.error(f"Error closing positions for {pair}: {e}")

    def trade_pairs(self):
        """Main trading logic for all pairs"""
        for pair in self.pairs:
            try:
                symbol1, symbol2 = pair

                price1_info = self.api.get_mark_price(symbol1)
                price2_info = self.api.get_mark_price(symbol2)

                if not price1_info or not price2_info:
                    continue

                price1 = float(price1_info["markPrice"])
                price2 = float(price2_info["markPrice"])

                if price1 == 0 or price2 == 0:
                    continue

                close1 = self.get_historical_klines(symbol1)
                close2 = self.get_historical_klines(symbol2)

                if len(close1) == 0 or len(close2) == 0 or len(close1) != len(close2):
                    continue

                spread_values = np.log(close1) - np.log(close2)
                zscore = self.calculate_zscore(spread_values)

                logging.info(f"Pair: {symbol1}-{symbol2}, Z-score: {zscore:.4f}")

                current_position = self.positions[pair]

                if current_position == 0:
                    if zscore < -self.zscore_threshold:
                        pred_direction = self.predict_spread(spread_values)
                        if pred_direction == 1:
                            self.open_positions(pair, 1, price1, price2)
                    elif zscore > self.zscore_threshold:
                        pred_direction = self.predict_spread(spread_values)
                        if pred_direction == -1:
                            self.open_positions(pair, -1, price1, price2)
                elif current_position == 1 and zscore >= 0:
                    self.close_positions(pair, price1, price2)
                elif current_position == -1 and zscore <= 0:
                    self.close_positions(pair, price1, price2)

            except Exception as e:
                logging.error(f"Error processing pair {pair}: {str(e)}")
                continue

    def run(self):
        """Run the trading bot on exact hour marks"""
        logging.info("Starting statistical arbitrage bot...")

        while True:
            try:

                now = datetime.now()

                next_hour = (now + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )

                sleep_seconds = (next_hour - now).total_seconds()

                logging.info(f"Waiting for next hour. Next execution at: {next_hour}")

                time.sleep(sleep_seconds)

                self.trade_pairs()

            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")

                time.sleep(60)


def load_config(file_path: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Binance Futures Statistical Arbitrage Bot"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    parser.add_argument(
        "--testnet", action="store_true", help="Use testnet instead of mainnet"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    bot = BinanceStatArb(config)

    try:
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot stopped due to error: {e}")
