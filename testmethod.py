import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from datetime import datetime
import ccxt

# Configuration
MIN_CHANGE_THRESHOLD = 0.01  # 0.5% price change threshold
FIB_TOLERANCE = 0.005  # Tolerance for Fibonacci ratio matching
LOOKBACK_DEFAULT = 5  # Default lookback for swing detection
LOOKBACK_LONG_TERM = 100  # Lookback for long-term swing detection
ATR_PERIOD = 14  # Period for ATR-based adaptive lookback
SHORT_TERM_WINDOW = 20  # Window size for short-term patterns
LONG_TERM_WINDOW = 500  # Window size for long-term patterns


# Data Fetching and Preprocessing
def load_data(csv_file=None, exchange='binance', symbol='BTC/USDT', timeframe='1h', limit=1000):
    if csv_file:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df.dropna(subset=['price'], inplace=True)
        prices = df['price'].to_numpy()
        dates = df['timestamp'].to_list()
    else:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        prices = df['close'].to_numpy()
        dates = df['timestamp'].to_list()
    return prices, dates, df


# Trend Detection (Adapted from Provided Code)
def detect_trend_with_ma(prices, short_window=20, long_window=50):
    prices_array = np.array(prices)
    if len(prices_array) < long_window:
        return "up" if prices_array[-1] > prices_array[0] else "down"
    short_ma = np.convolve(prices_array, np.ones(short_window) / short_window, mode='valid')
    long_ma = np.convolve(prices_array, np.ones(long_window) / long_window, mode='valid')
    offset = long_window - short_window
    short_ma = short_ma[offset:]
    recent_periods = min(10, len(short_ma) - 1)
    short_ma_slope = short_ma[-1] - short_ma[-recent_periods - 1]
    if short_ma[-1] > long_ma[-1] and short_ma_slope > 0:
        return "up"
    elif short_ma[-1] < long_ma[-1] and short_ma_slope < 0:
        return "down"
    elif short_ma_slope > 0:
        return "up"
    else:
        return "down"


# Swing Point Detection (Proposed Algorithm with Long-Term Support)
def detect_swings(prices, long_term=False, lookback=LOOKBACK_DEFAULT, atr_period=ATR_PERIOD,
                  min_change_threshold=MIN_CHANGE_THRESHOLD):
    # Use larger lookback for long-term patterns
    lookback = LOOKBACK_LONG_TERM if long_term else lookback

    # Adaptive lookback based on ATR
    atr = np.std(np.diff(prices[-atr_period:])) if len(prices) >= atr_period else 0.01
    lookback = max(3, min(100 if long_term else 10, int(atr / np.mean(prices[-atr_period:]) * 100)))

    # Detect peaks and troughs
    peaks, _ = find_peaks(prices, distance=lookback)
    troughs, _ = find_peaks(-prices, distance=lookback)

    # Filter swings based on 0.5% price change threshold
    significant_peaks = []
    significant_troughs = []

    # Check price changes for peaks
    for i in peaks:
        if i > 0 and i < len(prices) - 1:
            prev_price = prices[i - 1]
            next_price = prices[i + 1]
            price_change_prev = abs(prices[i] - prev_price) / prev_price
            price_change_next = abs(prices[i] - next_price) / prices[i]
            if price_change_prev >= min_change_threshold or price_change_next >= min_change_threshold:
                significant_peaks.append(i)

    # Check price changes for troughs
    for i in troughs:
        if i > 0 and i < len(prices) - 1:
            prev_price = prices[i - 1]
            next_price = prices[i + 1]
            price_change_prev = abs(prices[i] - prev_price) / prev_price
            price_change_next = abs(prices[i] - next_price) / prices[i]
            if price_change_prev >= min_change_threshold or price_change_next >= min_change_threshold:
                significant_troughs.append(i)

    return np.array(significant_peaks), np.array(significant_troughs)


# Fibonacci Pattern Detection (Hybrid with Long-Term and Failure Criteria)
def detect_patterns(prices, dates, swings, trend, long_term=False, fib_tolerance=FIB_TOLERANCE,
                    min_change_threshold=MIN_CHANGE_THRESHOLD):
    patterns = []
    peaks, troughs = swings
    swing_points = sorted([(i, prices[i], 'high' if i in peaks else 'low') for i in set(peaks).union(troughs)])

    for i in range(len(swing_points) - 3):
        A_idx, A_price, A_type = swing_points[i]
        B_idx, B_price, B_type = swing_points[i + 1]
        C_idx, C_price, C_type = swing_points[i + 2]
        D_idx, D_price, D_type = swing_points[i + 3]

        # Ensure correct swing sequence and trend alignment
        if A_type == B_type or (trend == 'up' and A_price > B_price) or (trend == 'down' and A_price < B_price):
            continue

        # Check price changes between points (>= 0.5%)
        ab_change = abs(B_price - A_price) / A_price
        bc_change = abs(C_price - B_price) / B_price
        cd_change = abs(D_price - C_price) / C_price
        if ab_change < min_change_threshold or bc_change < min_change_threshold or cd_change < min_change_threshold:
            continue

        # Calculate AB move and BC retracement
        ab_move = abs(A_price - B_price)
        bc_retracement = abs(B_price - C_price) / ab_move if ab_move > 0 else 0
        fib_ratios = [0.382, 0.5, 0.618, 0.786]

        # Check if BC is a valid retracement
        if any(abs(bc_retracement - ratio) < fib_tolerance for ratio in fib_ratios):
            # Check CD extension or AB=CD
            cd_move = abs(C_price - D_price)
            ab_cd_ratio = cd_move / ab_move if ab_move > 0 else 0
            extension_ratios = [1.0, 1.272, 1.618]

            is_valid_pattern = (
                    any(abs(ab_cd_ratio - ratio) < fib_tolerance for ratio in extension_ratios) or
                    (trend == 'up' and D_price > B_price) or (trend == 'down' and D_price < B_price)
            )
            is_failed = (
                    bc_retracement >= 0.764 or
                    (trend == 'up' and C_price < A_price) or
                    (trend == 'down' and C_price > A_price)
            )

            if is_valid_pattern or is_failed:
                pattern = {
                    'A': (A_idx, A_price),
                    'B': (B_idx, B_price),
                    'C': (C_idx, C_price),
                    'D': (D_idx, D_price),
                    'direction': trend,
                    'failed': is_failed,
                    'long_term': long_term,
                    'retracement': bc_retracement,
                    'extension': ab_cd_ratio
                }
                patterns.append(pattern)

    return patterns


# Multi-Window Analysis for Short-Term and Long-Term Patterns
def analyze_multiple_windows(prices, dates, trend, window_sizes=[SHORT_TERM_WINDOW, LONG_TERM_WINDOW],
                             min_change_threshold=MIN_CHANGE_THRESHOLD):
    all_patterns = []

    for window_size in window_sizes:
        long_term = (window_size >= LONG_TERM_WINDOW)
        step_size = window_size // 2  # 50% overlap

        for start_idx in range(0, max(1, len(prices) - window_size + 1), step_size):
            end_idx = min(start_idx + window_size, len(prices))
            window_prices = prices[start_idx:end_idx]
            window_dates = dates[start_idx:end_idx]

            if len(window_prices) < 50:  # Skip small windows
                continue

            # Detect swings
            swings = detect_swings(window_prices, long_term=long_term, min_change_threshold=min_change_threshold)

            # Detect patterns
            patterns = detect_patterns(window_prices, window_dates, swings, trend, long_term=long_term,
                                       min_change_threshold=min_change_threshold)

            # Adjust indices to global dataset
            for pattern in patterns:
                adjusted_pattern = {
                    'A': (pattern['A'][0] + start_idx, pattern['A'][1]),
                    'B': (pattern['B'][0] + start_idx, pattern['B'][1]),
                    'C': (pattern['C'][0] + start_idx, pattern['C'][1]),
                    'D': (pattern['D'][0] + start_idx, pattern['D'][1]),
                    'direction': pattern['direction'],
                    'failed': pattern['failed'],
                    'long_term': pattern['long_term'],
                    'retracement': pattern['retracement'],
                    'extension': pattern['extension'],
                    'window_info': {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'window_size': window_size
                    }
                }
                all_patterns.append(adjusted_pattern)

    # Sort by significance (long-term and failed patterns first)
    all_patterns.sort(key=lambda p: (p['long_term'], p['failed'], -p['retracement']), reverse=True)
    return all_patterns


# Visualization (Adapted from Provided Code with Long-Term Styling)
def plot_patterns(prices, dates, patterns, trend):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(dates, prices, label='Price', color='black', alpha=0.7)

    colors = ['red', 'blue', 'green', 'purple']
    for i, pattern in enumerate(patterns):
        is_failed = pattern['failed']
        is_long_term = pattern['long_term']
        color = colors[i % len(colors)] if not is_long_term else ('darkred' if is_failed else 'darkblue')
        marker = 'x' if is_failed else 'o'
        linewidth = 2 if is_long_term else 1
        markersize = 150 if is_long_term else 100

        # Plot points
        for label in ['A', 'B', 'C', 'D']:
            idx, price = pattern[label]
            if idx < len(dates):  # Ensure index is valid
                ax.scatter(dates[idx], price, color=color, marker=marker, s=markersize)
                ax.text(dates[idx], price,
                        f'{label}{" (F)" if is_failed and label == "C" else ""}{" (LT)" if is_long_term else ""}',
                        color=color, fontsize=12, ha='right')

        # Draw connecting lines
        points = [(pattern['A'][0], pattern['A'][1]), (pattern['B'][0], pattern['B'][1]),
                  (pattern['C'][0], pattern['C'][1])]
        if not is_failed:
            points.append((pattern['D'][0], pattern['D'][1]))
        x_points, y_points = zip(*[(dates[idx], price) for idx, price in points if idx < len(dates)])
        ax.plot(x_points, y_points, color=color, linestyle='--' if is_failed else '-', linewidth=linewidth, alpha=0.7)

        # Draw Fibonacci levels
        A, B = pattern['A'][1], pattern['B'][1]
        main_move = B - A if pattern['direction'] == 'up' else A - B
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, 1.618]
        for level in fib_levels:
            fib_price = B - (main_move * level) if pattern['direction'] == 'up' else B + (main_move * level)
            linestyle = '-' if is_long_term and level == 0.764 and is_failed else '--'
            ax.axhline(y=fib_price, color=color, linestyle=linestyle, alpha=0.6 if not is_failed else 0.8,
                       label=f'Fib {level * 100:.1f}%{" (Failed)" if level == 0.764 and is_failed else ""}')
            if is_long_term and level == 0.764 and is_failed:
                ax.text(dates[0], fib_price, '76.4% FAILED', color='red', fontsize=12, fontweight='bold')

    ax.set_title(f'BTC Pattern Detection - Trend: {trend} (0.5% Price Change Threshold)', fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('btc_patterns.png')
    plt.show()


# Main Function
def analyze_btc_patterns(csv_file=None, timeframe='1h', limit=1000):
    # Load data
    prices, dates, df = load_data(csv_file, timeframe=timeframe, limit=limit)

    # Detect trend
    trend = detect_trend_with_ma(prices)
    print(f"Detected trend: {trend}")

    # Analyze multiple windows for short-term and long-term patterns
    patterns = analyze_multiple_windows(prices, dates, trend, window_sizes=[SHORT_TERM_WINDOW, LONG_TERM_WINDOW])

    # Analyze and visualize
    if patterns:
        for pattern in patterns:
            print(f"\nPattern: {pattern['direction']}, Long-Term: {pattern['long_term']}, Failed: {pattern['failed']}")
            print(
                f"Window: {pattern['window_info']['start_idx']} to {pattern['window_info']['end_idx']} (Size: {pattern['window_info']['window_size']})")
            print(
                f"Points: A={pattern['A'][1]:.2f}, B={pattern['B'][1]:.2f}, C={pattern['C'][1]:.2f}, D={pattern['D'][1]:.2f}")
            print(f"Retracement: {pattern['retracement'] * 100:.1f}%, Extension: {pattern['extension'] * 100:.1f}%")

        plot_patterns(prices, dates, patterns, trend)
    else:
        print("No valid patterns detected.")


# Example Usage
if __name__ == "__main__":
    analyze_btc_patterns(csv_file='C:/Users/admin/Desktop/btc_minute_data/btc_minute_data_2025-04-15.csv')