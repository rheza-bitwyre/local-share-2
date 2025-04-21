import polars as pl
import numpy as np
import logging
from itertools import product
from pathlib import Path
import os
import datetime

# Logging Setup
LOG_DIR = "/home/ubuntu/Rheza/local-share/06_trades_and_orderbooks"
os.makedirs(LOG_DIR, exist_ok=True)
time_start = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{LOG_DIR}/ft_bot_backtest_{time_start}.log"
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

def read_aggregated_files(base_path, symbol, interval, years):
    """
    Reads and concatenates aggregated trade data files for a given symbol, interval, and multiple years.
    """
    dfs = []
    data_dir = Path(base_path) / f"{symbol}_perps" / f"agg_{interval}"

    for year in years:
        files = [
            data_dir / f"{symbol}-aggTrades-{year}-{month:02d}_aggregated_{interval}.parquet"
            for month in range(1, 13)
        ]

        for file in files:
            if file.exists():
                try:
                    df = pl.read_parquet(file).with_columns(
                        [pl.col(col).cast(pl.Float64) for col in pl.read_parquet(file).columns]
                    )
                    dfs.append(df)
                    logging.info(f"Loaded {file}")
                except Exception as e:
                    logging.error(f"Error reading {file}: {e}")

    if dfs:
        return pl.concat(dfs)
    else:
        logging.warning(f"No files found for {symbol} at interval {interval}.")
        return pl.DataFrame()

# Example usage
base_path = "/home/ubuntu/Rheza/data/binance_aggtrades"
symbol_a = "DOGEUSDT"
symbol_b = "BTCUSDT"
interval = "15s"
years = [2022, 2023]

dfa = read_aggregated_files(base_path, symbol_a, interval, years)
logging.info(f"Data loaded for {symbol_a}, total rows: {len(dfa)}")

dfb = read_aggregated_files(base_path, symbol_b, interval, years)
logging.info(f"Data loaded for {symbol_b}, total rows: {len(dfb)}")

# Feature Engineering
dfa_featured = dfa.with_columns([
    ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).cast(pl.Float64).alias("occ"),
    ((pl.col("high") - pl.col("open")) / pl.col("open") * 100).cast(pl.Float64).alias("ohc"),
    ((pl.col("low") - pl.col("open")) / pl.col("open") * 100).cast(pl.Float64).alias("olc"),
    pl.col("close").rolling_mean(window_size=120).alias("rolling_mean"),
    pl.col("close").rolling_std(window_size=120).alias("rolling_std"),
])

dfa_featured = dfa_featured.with_columns(
    ((pl.col("close") - pl.col("rolling_mean")) / pl.col("rolling_std")).alias("rolling_zscore")
).drop_nulls()

dfb_featured = dfb.with_columns([
    pl.col("close").rolling_mean(window_size=120).alias("rolling_mean"),
    pl.col("close").rolling_std(window_size=120).alias("rolling_std"),
])

dfb_featured = dfb_featured.with_columns([
    ((pl.col("close") - pl.col("rolling_mean")) / pl.col("rolling_std")).alias("rolling_zscore_btc"),
    pl.when(pl.col("close") > pl.col("open"))
    .then(1)
    .when(pl.col("close") == pl.col("open"))
    .then(0)
    .otherwise(-1)
    .alias("bull_btc"),
]).drop_nulls()

dfb_featured = dfb_featured.drop(["open", "high", "low", "close", "rolling_mean", "rolling_std"])

# Combine Data
dfs_featured = dfa_featured.join(dfb_featured, on=["year", "month", "day", "hour", "minute", "interval"], how="inner")
logging.info(f"Combined data shape: {dfs_featured.shape}")

import polars as pl
import numpy as np
from itertools import product

# Define threshold lists
thresholds = {
    "occ": [-0.2, -0.25, -0.3],
    "ohc": [100],
    "olc": [-0.2, -0.25, -0.3],
    "rolling_zscore_btc": [1, -1, -100],
    "bull_btc": [-1, 0, 1],
    "hold_periods": [1, 2, 4, 6, 8,],
    "tp": [100],
    "sl": [-0.05],
}

# Trading fees
taker_fee, maker_fee = 0.05, 0.02  

# Generate all possible threshold combinations
threshold_combinations = list(product(*thresholds.values()))
len_combinations = len(threshold_combinations)

# Store results
results = []
combination_no = 1

# Iterate over all combinations
for thresholds in threshold_combinations:
    (
        occ_threshold, ohc_threshold, olc_threshold,
        rolling_zscore_btc_threshold, bull_btc_threshold,
        hold_periods, tp_threshold, sl_threshold
    ) = thresholds

    logging.info(f"Processing Combination - {combination_no}/{len_combinations}")
    combination_no += 1

    logging.info(f"Processing - OCC:{occ_threshold}%, OHC:{ohc_threshold}%, Zscore:{rolling_zscore_btc_threshold}, Bull:{bull_btc_threshold}")
    logging.info(f"Processing - Hold Period:{hold_periods}, TP:{tp_threshold}%, SL:{sl_threshold}%")


    if occ_threshold < olc_threshold:
        logging.info("Combination Skipped as OCC < OLC!")
        continue
    if abs(sl_threshold) > tp_threshold:
        logging.info("Combination Skipped as abs(SL) > TP!")
        continue

    # Tracking variables
    short_open, short_positions, wins = False, 0, 0
    total_pnl_pct, tp_hit_count, sl_hit_count = 0, 0, 0
    most_negative_pnl = 0  
    max_lose_streak, current_lose_streak = 0, 0  
    lose_streaks = []
    entry_price, initial_entry_price, holding_counter, trade_pnl = None, None, 0, None
    trade_pnl_list = []

    # Convert to list for fast indexing
    dfs_rows = dfs_featured.to_dicts()

    # Process trades
    rows = []
    for i, row in enumerate(dfs_rows[:-1]):
        next_row = dfs_rows[i + 1]
        action, tp_hit, sl_hit, trade_pnl = "", 0, 0, None

        if short_open:
            # Calculate PnL using the VERY FIRST entry price
            pnl_pct = ((initial_entry_price - row["close"]) / initial_entry_price) * 100
            adjusted_pnl = pnl_pct - taker_fee

            # Calculate TP/SL thresholds using the LATEST entry price
            tp_pct = (entry_price - entry_price * (1 - tp_threshold / 100)) / entry_price * 100
            sl_pct = (entry_price - entry_price * (1 - sl_threshold / 100)) / entry_price * 100

            # 1. Check SL/TP first
            if adjusted_pnl <= sl_pct:
                sl_hit, sl_hit_count = 1, sl_hit_count + 1
                trade_pnl = sl_threshold - taker_fee
                action = "Close"
                short_open = False
            elif adjusted_pnl >= tp_pct:
                tp_hit, tp_hit_count = 1, tp_hit_count + 1
                trade_pnl = tp_threshold - maker_fee
                action = "Close"
                short_open = False
            else:
                # 2. Check for new signals EVERY BAR
                if (
                    row["occ"] <= occ_threshold
                    and row["ohc"] <= ohc_threshold
                    and row["olc"] <= olc_threshold
                    and row["rolling_zscore_btc"] >= rolling_zscore_btc_threshold
                    and row["bull_btc"] <= bull_btc_threshold
                ):
                    # Update entry price (for TP/SL) AND reset counter
                    entry_price = next_row["open"]  # 🟡 Update for TP/SL
                    holding_counter = 0

                # 3. Check hold period expiration
                if holding_counter >= hold_periods:
                    trade_pnl = adjusted_pnl - taker_fee
                    action = "Close"
                    short_open = False

            # Update metrics if closing
            if action == "Close":
                total_pnl_pct += trade_pnl
                trade_pnl_list.append(trade_pnl)
                
                most_negative_pnl = min(most_negative_pnl, trade_pnl)
                
                if trade_pnl < 0:
                    current_lose_streak += 1
                    max_lose_streak = max(max_lose_streak, current_lose_streak)
                else:
                    if current_lose_streak > 0:
                        lose_streaks.append(current_lose_streak)
                        current_lose_streak = 0
                    wins += 1

        # Check for new short entry
        if not short_open and (
            row["occ"] <= occ_threshold
            and row["ohc"] <= ohc_threshold
            and row["olc"] <= olc_threshold
            and row["rolling_zscore_btc"] >= rolling_zscore_btc_threshold
            and row["bull_btc"] <= bull_btc_threshold
        ):
            action, short_open = "short", True
            entry_price = next_row["open"]  # Initial entry
            initial_entry_price = entry_price  # 🟡 Track first entry price
            holding_counter, short_positions = 0, short_positions + 1
            trade_pnl = None

        # Append results
        rows.append({
            **row,
            "action": action,
            "tp_hit": tp_hit,
            "sl_hit": sl_hit,
            "pnl": trade_pnl if trade_pnl is not None else 0.0
        })

        if short_open:
            holding_counter += 1

    # Convert to Polars DataFrame
    dfs_featured_result = pl.DataFrame(rows)

    # Win/Loss Trade Analysis
    trade_pnl_array = np.array(trade_pnl_list) if trade_pnl_list else np.array([0])

    # Separate winning and losing trades
    win_trades = trade_pnl_array[trade_pnl_array >= 0]
    lose_trades = trade_pnl_array[trade_pnl_array < 0]

    # Compute mean and median
    win_mean = np.mean(win_trades) if len(win_trades) > 0 else 0
    win_median = np.median(win_trades) if len(win_trades) > 0 else 0
    lose_mean = np.mean(lose_trades) if len(lose_trades) > 0 else 0
    lose_median = np.median(lose_trades) if len(lose_trades) > 0 else 0

    # Compute median losing streak
    lose_streaks = np.array(lose_streaks) if lose_streaks else np.array([0])
    median_lose_streak = np.median(lose_streaks)

    # Calculate final stats
    win_rate = (wins / short_positions) * 100 if short_positions > 0 else 0
    total_pnl_pct = round(total_pnl_pct, 2)

    logging.info(f"Results - PnL:{total_pnl_pct}%, Long Opened:{short_positions}, Median Losing Streak:{median_lose_streak}")

    # Store results
    results.append({
        "OCC": occ_threshold, "OHC": ohc_threshold, "OLC": olc_threshold,
        "BTC_Zscore": rolling_zscore_btc_threshold, "BTC_Bull": bull_btc_threshold,
        "Hold_Period": hold_periods, "TP": tp_threshold, "SL": sl_threshold,
        "Total short Positions": short_positions, "Total PnL%": total_pnl_pct,
        "Win Rate%": win_rate, "Total TP Hits": tp_hit_count, "Total SL Hits": sl_hit_count,
        "Most Negative PnL": most_negative_pnl, "Max Losing Streak": max_lose_streak,
        "Median Losing Streak": median_lose_streak,  # New field
        "Win Mean": win_mean, "Win Median": win_median,
        "Lose Mean": lose_mean, "Lose Median": lose_median
    })

logging.info("Backtest Looping Done!")

# Convert results to Polars DataFrame
df_results = pl.DataFrame(results)

# Display the final DataFrame containing all results
df_results.write_csv(f"/home/ubuntu/Rheza/local-share/06_trades_and_orderbooks/ft_bot_short_backtest_result_{time_start}.csv")