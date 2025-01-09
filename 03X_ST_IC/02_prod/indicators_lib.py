# indicators_lib.py

import pandas as pd
import numpy as np
from pandas_ta.overlap import hl2
from pandas_ta.volatility import atr
from pandas_ta.utils import get_offset, verify_series

class Indicators:
    def __init__(self, df=None):
        self.df = df if df is not None else pd.DataFrame()

    def calculate_supertrend(self, high_col='highprice', low_col='lowprice', close_col='closeprice', length=10, multiplier=3.0, offset=0, drop_columns=True, **kwargs):
        """
        Calculate the Supertrend indicator and merge it with the original DataFrame.
        """
        # Validate Arguments
        high = verify_series(self.df[high_col], length)
        low = verify_series(self.df[low_col], length)
        close = verify_series(self.df[close_col], length)
        offset = get_offset(offset)

        if high is None or low is None or close is None:
            return self.df

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
        result_df = self.df.join(supertrend_df)

        # Drop unnecessary columns
        if drop_columns:
            result_df.drop(columns=[f"SUPERT{_props}", f"SUPERTd{_props}"], inplace=True)

        return result_df

    def compute_ichimoku_with_supertrend(self, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26):
        """
        Compute Ichimoku Cloud components and join them with the original supertrend DataFrame.
        """
        # Helper to calculate the average of the highest high and lowest low
        def donchian(data, period):
            return (data['highprice'].rolling(window=period).max() + 
                    data['lowprice'].rolling(window=period).min()) / 2

        # Compute Ichimoku Cloud components
        self.df['conversion_line'] = donchian(self.df, conversion_periods)
        self.df['base_line'] = donchian(self.df, base_periods)
        self.df['leading_span_a'] = ((self.df['conversion_line'] + self.df['base_line']) / 2).shift(displacement)
        self.df['leading_span_b'] = donchian(self.df, span_b_periods).shift(displacement)
        self.df['lagging_span'] = self.df['closeprice'].shift(-displacement)
        
        # Drop unnecessary columns
        self.df.drop(columns=['conversion_line', 'base_line', 'lagging_span'], inplace=True)
        
        return self.df