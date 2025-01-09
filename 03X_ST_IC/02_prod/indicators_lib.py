import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def super_trend(df, period=7, multiplier=3):
        """
        Calculates the Super Trend indicator.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.
            period (int): ATR calculation period.
            multiplier (int or float): ATR multiplier for calculating Super Trend bands.
        
        Returns:
            pd.DataFrame: DataFrame with additional columns 'Supertrend' and 'Supertrend Direction'.
        """
        # Calculate ATR (Average True Range)
        df['TR'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        df['UpperBand'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['LowerBand'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
        
        # Initialize Supertrend and its direction
        df['Supertrend'] = np.nan
        df['Supertrend Direction'] = np.nan
        
        # Calculate Supertrend
        for i in range(1, len(df)):
            prev_supertrend = df['Supertrend'].iloc[i - 1] if i > 1 else df['LowerBand'].iloc[0]
            if df['close'].iloc[i] > prev_supertrend:
                df.at[i, 'Supertrend'] = max(df['LowerBand'].iloc[i], prev_supertrend)
                df.at[i, 'Supertrend Direction'] = 'Bullish'
            else:
                df.at[i, 'Supertrend'] = min(df['UpperBand'].iloc[i], prev_supertrend)
                df.at[i, 'Supertrend Direction'] = 'Bearish'
        
        return df

    @staticmethod
    def ichimoku_cloud(df):
        """
        Calculates the Ichimoku Cloud indicator.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.
        
        Returns:
            pd.DataFrame: DataFrame with additional columns for Ichimoku Cloud components.
        """
        # Conversion Line (Tenkan-sen)
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['Tenkan-sen'] = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['Kijun-sen'] = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['Senkou Span B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Lagging Span (Chikou Span)
        df['Chikou Span'] = df['close'].shift(-26)

        # Drop unnecessary columns
        df.drop(columns=['conversion_line', 'base_line', 'lagging_span'], inplace=True)
        
        return df