# %% [markdown]
# # USD JPY Hour Data

# %%
import polars as pl
import logging
import pytz
from datetime import datetime

# Define the UTC+8 timezone (Bali time)
bali_timezone = pytz.timezone('Asia/Makassar')  # Bali is in the same timezone as Makassar (UTC+8)

# Get the current time in UTC+8 and format it as a string
current_time = datetime.now(bali_timezone).strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=f'/home/ubuntu/Rheza/local-share/Logs/usd_jpy_linear_{current_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Read the CSV file into a Polars DataFrame
file_path = "USDJPY60.csv"  # Replace with your CSV file path
df = pl.read_csv(file_path, separator="\t", has_header=False)

# Rename columns to match the expected structure
df.columns = ["date", "open", "high", "low", "close", "volume"]

# Display the first few rows
df

# %% [markdown]
# # Clean + Features Engineering

# %%
import polars as pl

# Load the DataFrame (assuming it is already loaded as df)

df_features = df.clone()

# Ensure 'date' is in datetime format
df_features = df_features.with_columns(
    pl.col("date").str.strptime(pl.Datetime).alias("date")
)

# Create lagged features
df_features = df_features.with_columns(
    pl.col('open').shift(1).alias('prev_open'),
    pl.col('high').shift(1).alias('prev_high'),
    pl.col('low').shift(1).alias('prev_low'),
    pl.col('close').shift(1).alias('prev_close'),
    pl.col('volume').shift(1).alias('prev_volume')
)

# Calculate changes
df_features = df_features.with_columns(
    (pl.col('open') - pl.col('prev_open')).alias('open_change'),
    (pl.col('high') - pl.col('prev_high')).alias('high_change'),
    (pl.col('low') - pl.col('prev_low')).alias('low_change'),
    (pl.col('close') - pl.col('prev_close')).alias('close_change'),
    ((pl.col('open') - pl.col('prev_open')) / pl.col('prev_open') * 100).alias('open_pct_change'),
    ((pl.col('close') - pl.col('prev_close')) / pl.col('prev_close') * 100).alias('close_pct_change')
)

# Calculate rolling statistics
df_features = df_features.with_columns(
    pl.col('open').rolling_mean(3).alias('rolling_avg_open_3h'),
    pl.col('close').rolling_mean(5).alias('rolling_avg_close_5h'),
    pl.col('high').rolling_max(3).alias('rolling_max_high_3h'),
    pl.col('low').rolling_min(3).alias('rolling_min_low_3h')
)

# Calculate volume features
df_features = df_features.with_columns(
    (pl.col('volume') - pl.col('prev_volume')).alias('volume_change'),
    pl.col('volume').rolling_mean(3).alias('volume_ma_3h')
)

# Extract time-related features
df_features = df_features.with_columns(
    pl.col('date').dt.hour().alias('hour_of_day'),
    pl.col('date').dt.weekday().alias('day_of_week')  # Corrected method
)

# Calculate ratios
df_features = df_features.with_columns(
    (pl.col('high') / pl.col('low')).alias('high_low_ratio'),
    (pl.col('close') / pl.col('open')).alias('close_open_ratio')
)

# Calculate the highest close in the last 4, 8, 12, and 24 hours using rolling windows
df_features = df_features.with_columns(
    pl.col('close').rolling_max(window_size=4).alias('max_close_4h'),
    pl.col('close').rolling_max(window_size=8).alias('max_close_8h'),
    pl.col('close').rolling_max(window_size=12).alias('max_close_12h'),
    pl.col('close').rolling_max(window_size=24).alias('max_close_24h')
)

# Calculate the min and max of the entire 'close' column
max_close_all = df_features['close'].max()
min_close_all = df_features['close'].min()

# Calculate the difference from the current close to the max and min close
df_features = df_features.with_columns(
    (pl.col('close') - max_close_all).alias('diff_to_max_close'),
    (pl.col('close') - min_close_all).alias('diff_to_min_close')
)

# Drop rows with any null values
df_features = df_features.drop_nulls()

df_features = df_features.with_columns(
    pl.col('close').shift(-1).alias('next_close'),
)

# Print the updated DataFrame schema
df_features

# %% [markdown]
# # Linear Regression

# %%
import time
import numpy as np
import polars as pl
import pandas as pd
from sklearn.linear_model import LinearRegression

# Separate df into features and target using Polars' select method
features_df = df_features.select([col for col in df_features.columns if col not in ['datetime', 'next_close']])
target_df = df_features['next_close']

# Convert Polars DataFrame to NumPy arrays for processing
X = features_df.to_numpy()
y = target_df.to_numpy()

# Specify parameters for the sliding window approach
num_predictions = 1   # Number of rows to predict
gap = 1               # Gap (number of rows to skip after each window)
max_windows = 50      # Maximum number of windows to process
set_limit = False     # Set this to False to process all windows

# Define list of window sizes
window_sizes = list(range(10000, 95001, 10000))
# window_sizes = [90000]

# List to store results
results = []

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

        start = window_number
        end = start + window_size
        X_train = X[start:end]
        y_train = y[start:end]

        # Normalize the training features
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_normalized = (X_train - X_train_mean) / X_train_std

        # Get the column index for 'close' from features_df
        close_index = features_df.columns.index('close')

        # Normalize y_train using the mean and std of the 'close' column
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
        model = LinearRegression()
        model.fit(X_train_normalized, y_train_normalized)

        # Predict on validation data
        y_pred_val = model.predict(X_val_normalized)
        # Predict on training data
        y_pred_train = model.predict(X_train_normalized)

        # Denormalize y_val, y_pred_train, and y_pred_val using the mean and std of 'close'
        y_train_denorm = y_train * close_std + close_mean
        y_pred_train_denorm = y_pred_train * close_std + close_mean
        y_pred_val_denorm = y_pred_val * close_std + close_mean

        # Calculate RMSE and RMSE percentage for validation
        mse_val = np.mean((y_val - y_pred_val_denorm) ** 2)
        rmse_val = np.sqrt(mse_val)
        rmse_val_perc = (rmse_val / y_val)[0] * 100  # Convert to percentage

        # Calculate RMSE for training
        mse_train = np.mean((y_train - y_pred_train_denorm) ** 2)
        rmse_train = np.sqrt(mse_train)
        rmse_train_perc = ((y_train - y_pred_train_denorm) ** 2 / y_train).mean() * 100  # Convert to percentage

        # Track the end time of the window processing
        end_time = time.time()

        # Calculate the time taken for this window
        window_time = end_time - start_time
        total_window_times += window_time  # Add the window time to the total time

        # Append RMSEs and percentage errors
        all_val_rmse.append(rmse_val)
        all_val_rmse_perc.append(rmse_val_perc)
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

        # Move to the next window based on the gap
        window_number += gap

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

    # Calculate percentage of times RMSE is less than or equal to 0.0776
    rmse_less_equal_avg_change = (np.sum(np.array(all_val_rmse) <= 0.0776) / len(all_val_rmse)) * 100

    # Calculate percentage of times prediction is lower or equal to average RMSE
    percentage_lower_equal_avg_rmse = (np.sum(np.array(all_val_rmse) <= avg_val_rmse) / len(all_val_rmse)) * 100

    # Append results to the list with updated metric name
    results.append({
        'window_size': window_size,
        'avg_val_rmse': avg_val_rmse,
        'windowed_confidence_level': percentage_lower_equal_avg_rmse,  # Renamed metric
        'var_val_rmse': var_val_rmse,
        'avg_val_rmse_perc': avg_val_rmse_perc,
        'var_val_rmse_perc': var_val_rmse_perc,
        'avg_train_rmse': avg_train_rmse,
        'var_train_rmse': var_train_rmse,
        'avg_train_rmse_perc': avg_train_rmse_perc,
        'var_train_rmse_perc': var_train_rmse_perc,
        'window_time': total_window_times,
        'max_avg_val_rmse_perc': max_avg_val_rmse_perc,
        'lower_count_perc': lower_count_perc,
        'higher_count_perc': higher_count_perc,
        'max_rmse_perc_lower': max_rmse_perc_lower,
        'max_rmse_perc_higher': max_rmse_perc_higher,
        'avg_rmse_perc_lower': avg_rmse_perc_lower,
        'avg_rmse_perc_higher': avg_rmse_perc_higher,
        'rmse_less_equal_60_perc': rmse_less_equal_avg_change
    })

    # Print results for the current window size with the new name
    logging.info(f'Window size [{window_size}] | Time Elapsed: {total_window_times:.3f} seconds')
    logging.info(f'Average Prediction Error: {avg_val_rmse:.3f} JPY | {avg_val_rmse_perc:.3f} % | Confidence Level: {percentage_lower_equal_avg_rmse:.3f} % ')
    logging.info(f'Average Prediction Error that less than avg change: {rmse_less_equal_avg_change:.3f} %')

# Optionally, you could convert the results to a DataFrame or CSV for further analysis
results_summary = pd.DataFrame(results)
results_summary


