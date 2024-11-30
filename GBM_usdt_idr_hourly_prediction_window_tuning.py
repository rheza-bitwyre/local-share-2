# %%
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging
import pytz

# Define the UTC+8 timezone (Bali time)
bali_timezone = pytz.timezone('Asia/Makassar')  # Bali is in the same timezone as Makassar (UTC+8)

# Get the current time in UTC+8 and format it as a string
current_time = datetime.now(bali_timezone).strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=f'/home/ubuntu/Rheza/local-share/Logs/GBM_window_tuning_{current_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# # Set a custom script name for interactive use
# script_name = 'GBM_window_tuning'  # Replace with a name relevant to your session

# # Get the current timestamp to append to the filename
# current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Construct the log file name
# log_filename = f"{script_name}_{current_time}.log"

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log the DataFrame
logging.info("Analysis Start!")

# Initialize the Indodax exchange
exchange = ccxt.indodax()

# Define the trading pair and timeframe
symbol = 'USDT/IDR'  # Example: USDT to Indonesian Rupiah
timeframe = '1h'     # Supported timeframes: '1m', '5m', '15m', '1h', '1d', etc.
limit = 60000        # Number of candles to fetch (max depends on the exchange)

# Fetch OHLCV data
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# Convert to DataFrame
columns = ['date', 'open', 'high', 'low', 'close', 'volume']
data = [
    [datetime.utcfromtimestamp(c[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'), *c[1:]]
    for c in ohlcv
]
indodax_df = pd.DataFrame(data, columns=columns)

# Ensure 'date' column is in datetime format
indodax_df['date'] = pd.to_datetime(indodax_df['date'])

# Convert from UTC to Bali time (UTC+8)
indodax_df['date'] = indodax_df['date'] + pd.Timedelta(hours=8)

# Log the DataFrame
logging.info("Indodax DataFrame loaded successfully.")

# %%
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging

# Initialize the Tokocrypto exchange
exchange = ccxt.tokocrypto()

# Define the trading pair and timeframe
symbol = 'USDT/IDR'  # Example: USDT to Indonesian Rupiah
timeframe = '1h'     # Supported timeframes: '1m', '5m', '15m', '1h', '1d', etc.
limit = 1000         # Number of candles to fetch per request

# Define the start and end times
start_time = int(datetime(2018, 8, 24, 15, 0).timestamp() * 1000)  # Start time in milliseconds
end_time = int(datetime.now().timestamp() * 1000)                  # Current time in milliseconds

# Initialize an empty list to collect OHLCV data
all_ohlcv = []
current_time = start_time

# Loop to fetch data in chunks of `limit`
while current_time < end_time:
    logging.info(f"Fetching data starting from: {datetime.utcfromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_time, limit=limit)

    if not ohlcv:
        logging.info("No more data available.")
        break

    all_ohlcv.extend(ohlcv)
    current_time = ohlcv[-1][0] + 1  # Increment start time to avoid overlapping

# Convert the collected data into a DataFrame
columns = ['date', 'open', 'high', 'low', 'close', 'volume']
data = [
    [datetime.utcfromtimestamp(c[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'), *c[1:]]
    for c in all_ohlcv
]
tokocrypto_df = pd.DataFrame(data, columns=columns)

# Ensure 'date' column is in datetime format
tokocrypto_df['date'] = pd.to_datetime(tokocrypto_df['date'])

# Convert from UTC to Bali time (UTC+8)
tokocrypto_df['date'] = tokocrypto_df['date'] + pd.Timedelta(hours=8)

# Log the DataFrame
logging.info("Tokocrypto DataFrame loaded successfully.")

# %%
# Merge the two DataFrames on the 'date' column
df = pd.merge(indodax_df, tokocrypto_df, on='date', suffixes=('_indodax', '_tokocrypto'))

# Calculate the average of 'open' and 'close' for the two sources
df['open'] = (df['open_indodax'] + df['open_tokocrypto']) / 2
df['close'] = (df['close_indodax'] + df['close_tokocrypto']) / 2

# Keep only the desired columns
df = df[['date', 'open', 'close']]

# Log the resulting DataFrame
logging.info("Merged DataFrame created.")

# %%
# Lag features
for lag in range(1, 6):
    df[f'open_lag_{lag}'] = df['open'].shift(lag)
    df[f'close_lag_{lag}'] = df['close'].shift(lag)

# Moving averages
for window in [5, 10, 20]:
    df[f'ma_open_{window}'] = df['open'].rolling(window=window).mean()
    df[f'ma_close_{window}'] = df['close'].rolling(window=window).mean()

# Percentage changes
df['open_pct_change'] = df['open'].pct_change() * 100
df['close_pct_change'] = df['close'].pct_change() * 100

# Time-based features
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Rolling statistics
for window in [5, 10, 20]:
    df[f'rolling_min_close_{window}'] = df['close'].rolling(window=window).min()
    df[f'rolling_max_close_{window}'] = df['close'].rolling(window=window).max()

# Exponential moving averages
for span in [5, 10, 20]:
    df[f'ema_close_{span}'] = df['close'].ewm(span=span).mean()

# Target variable
df['next_close'] = df['close'].shift(-1)

# Drop rows with NaN values
df = df.dropna()

# Log the DataFrame after feature engineering
logging.info("Feature engineering completed.")

# %%
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Separate df into features and target
features_df = df.drop(['date', 'next_close'], axis=1)
target_df = df['next_close']

# Convert to NumPy arrays
X = features_df.values
y = target_df.values

# Specify parameters for the sliding window approach
num_predictions = 1   # Number of rows to predict
gap = 1               # Gap (number of rows to skip after each window)
max_windows = 50      # Maximum number of windows to process
set_limit = False     # Set this to False to process all windows

# Define list of window sizes
window_sizes = list(range(6000, 11001, 1000))

# Loop through each window size
for window_size in window_sizes:

    # Initiate lists to store RMSEs and percentages
    all_val_rmse = []
    all_val_rmse_perc = []
    all_train_rmse = []
    all_train_rmse_perc = []
    total_window_times = 0  # Variable to store total time for all windows

    # Counters and variables for prediction comparison to actual values
    lower_count = 0
    higher_count = 0
    max_rmse_perc_lower = 0  # To store max RMSE% when prediction is lower than actual
    max_rmse_perc_higher = 0  # To store max RMSE% when prediction is higher than actual

    # Separate lists to store RMSE percentage for lower and higher predictions
    lower_rmse_percs = []
    higher_rmse_percs = []

    # Separate lists to store val_rmse for lower and higher predictions
    rmse_lower_perc = []  # To store RMSE when prediction is lower than actual
    rmse_higher_perc = []  # To store RMSE when prediction is higher than actual

    # Calculate the number of windows based on dataset size
    num_windows = len(X) - window_size - num_predictions

    # Apply maximum window limit if set
    if set_limit:
        num_windows = min(num_windows, max_windows)

    # Loop through each sliding window with the gap applied
    window_number = 0
    while window_number < num_windows:
        if window_number % 100 == 0:
            logging.info(f"Processing Window {window_number}/{num_windows}")

        start = window_number
        end = start + window_size
        X_train = X[start:end]
        y_train = y[start:end]

        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_normalized = (X_train - X_train_mean) / X_train_std

        # Get the column index for 'next_close' from features_df
        close_index = features_df.columns.get_loc('close')

        # Normalize y_train using the mean and std of close
        close_mean = X_train[:, close_index].mean()
        close_std = X_train[:, close_index].std()
        y_train_normalized = (y_train - close_mean) / close_std

        # Prepare validation data for prediction
        X_val = X[end:end + num_predictions]
        y_val = y[end:end + num_predictions]

        # Normalize validation data using the statistics from the training set
        X_val_normalized = (X_val - X_train_mean) / X_train_std

        # Track the start time of the window processing
        start_time = time.time()

        # Initialize and fit the model
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train_normalized, y_train_normalized)

        # Predict on validation data
        y_pred_val = model.predict(X_val_normalized)
        # Predict on training data
        y_pred_train = model.predict(X_train_normalized)

        # Denormalize y_val, y_pred_train, and y_pred_val using the mean and std of weekly_volatility
        y_pred_train_denorm = y_pred_train * close_std + close_mean
        y_pred_val_denorm = y_pred_val * close_std + close_mean

        # Calculate RMSE and RMSE percentage for validation
        mse_val = np.mean((y_val - y_pred_val_denorm) ** 2)
        rmse_val = np.sqrt(mse_val)
        rmse_val_perc = (rmse_val / y_val).mean() * 100  # Convert to percentage

        # Log validation RMSE
        logging.info(f"Window {window_number}: Validation RMSE = {rmse_val:.4f}")
        logging.info(f"Window {window_number}: Validation RMSE% = {rmse_val_perc:.4f}")

        # Store validation RMSE
        all_val_rmse.append(rmse_val)
        all_val_rmse_perc.append(rmse_val_perc)

        # Calculate RMSE for training
        mse_train = np.mean((y_train - y_pred_train_denorm) ** 2)
        rmse_train = np.sqrt(mse_train)
        rmse_train_perc = np.mean((y_train - y_pred_train_denorm) ** 2 / y_train) * 100 # Convert to percentage

        # Log training RMSE
        logging.info(f"Window {window_number}: Training RMSE = {rmse_train:.4f}")
        logging.info(f"Window {window_number}: Training RMSE% = {rmse_train_perc:.4f}")

        # Store training RMSE
        all_train_rmse.append(rmse_train)
        all_train_rmse_perc.append(rmse_train_perc)

        # Count predictions relative to actual values and update max RMSE percentage
        if y_pred_val_denorm < y_val:
            lower_count += 1
            lower_rmse_percs.append(rmse_val_perc)  # Store RMSE percentage for lower predictions
            max_rmse_perc_lower = max(max_rmse_perc_lower, rmse_val_perc)
        elif y_pred_val_denorm > y_val:
            higher_count += 1
            higher_rmse_percs.append(rmse_val_perc)  # Store RMSE percentage for higher predictions
            max_rmse_perc_higher = max(max_rmse_perc_higher, rmse_val_perc)

        # Increment window number for the next window
        window_number += gap

        # Calculate total time for window processing
        total_window_time = time.time() - start_time
        total_window_times += total_window_time

# Calculate percentage for lower and higher counts
    lower_count_perc = (lower_count / num_windows) * 100
    higher_count_perc = (higher_count / num_windows) * 100

    # Calculate average RMSE percentage errors for lower and higher predictions
    avg_rmse_perc_lower = np.mean(lower_rmse_percs) if lower_rmse_percs else 0
    avg_rmse_perc_higher = np.mean(higher_rmse_percs) if higher_rmse_percs else 0

    # Calculate average, max, min, and variance for validation and training RMSEs, percentages
    avg_val_rmse = np.mean(all_val_rmse)
    var_val_rmse = np.var(all_val_rmse)

    avg_val_rmse_perc = np.mean(all_val_rmse_perc)
    var_val_rmse_perc = np.var(all_val_rmse_perc)
    max_avg_val_rmse_perc = np.max(all_val_rmse_perc)

    avg_train_rmse = np.mean(all_train_rmse)
    var_train_rmse = np.var(all_train_rmse)

    avg_train_rmse_perc = np.mean(all_train_rmse_perc)
    var_train_rmse_perc = np.var(all_train_rmse_perc)

    # Calculate percentage of times prediction is lower or equal to average RMSE
    percentage_lower_equal_avg_rmse = (np.sum(np.array(all_val_rmse) <= avg_val_rmse) / len(all_val_rmse)) * 100

    # Log the final average results
    logging.info(f"Window size [{window_size}] | Time Elapsed: {total_window_times:.3f} seconds")
    logging.info(f"Average Prediction Error: {avg_val_rmse:.3f} IDR | {avg_val_rmse_perc:.3f} % | Confidence Level: {percentage_lower_equal_avg_rmse:.3f} % ")
    logging.info(f"Under Predict - Count: {lower_count_perc:.3f} % | Average: {avg_rmse_perc_lower:.3f} % | Max: {max_rmse_perc_lower:.3f} %")
    logging.info(f"Over Predict - Count: {higher_count_perc:.3f} % | Average: {avg_rmse_perc_higher:.3f} % | Max: {max_rmse_perc_higher:.3f} %")
    logging.info(f"Absolute Training Error Variance: {var_train_rmse:.3f} | Absolute Training Error Percentage Variance: {var_train_rmse_perc:.3f} %") 
    logging.info(f"Absolute Validation Error Variance: {var_val_rmse:.3f} | Absolute Validation Error Percentage Variance: {var_val_rmse_perc:.3f} %")

logging.info(f"Analysis Done!")