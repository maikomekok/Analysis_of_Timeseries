import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def detect_trend_with_ma(prices, short_window=20, long_window=50):
    """
    Detect trend based on moving average crossover.
    Returns "up" if short MA is above long MA, "down" otherwise.
    """
    prices_array = np.array(prices)

    if len(prices_array) < long_window:
        return "up" if prices_array[-1] > prices_array[0] else "down"

    short_ma = np.convolve(prices_array, np.ones(short_window) / short_window, mode='valid')
    long_ma = np.convolve(prices_array, np.ones(long_window) / long_window, mode='valid')

    # Ensure both MAs are the same length for comparison
    offset = long_window - short_window
    short_ma = short_ma[offset:]

    # Check the actual visual trend rather than just the last point
    # Look at the slope of the recent short MA
    recent_periods = min(10, len(short_ma) - 1)
    short_ma_slope = short_ma[-1] - short_ma[-recent_periods - 1]

    # Consider both MA crossover and recent direction
    if short_ma[-1] > long_ma[-1] and short_ma_slope > 0:
        return "up"
    elif short_ma[-1] < long_ma[-1] and short_ma_slope < 0:
        return "down"
    elif short_ma_slope > 0:  # Prioritize recent price action
        return "up"
    else:
        return "down"


def find_best_starting_point(prices, dates, window=50, min_change_threshold=0.003, max_days_between=None):
    """
    Find the best starting point for pattern analysis based on trend direction.
    Added min_change_threshold to avoid minor price swings and max_days_between for time constraints.
    """
    direction = detect_trend_with_ma(prices)
    prices_array = np.array(prices)
    window = min(window, len(prices) // 3)

    # Check if price changes meet minimum threshold
    price_changes = np.abs(np.diff(prices_array) / prices_array[:-1])
    significant_changes = price_changes >= min_change_threshold

    if direction == "up":
        local_mins = []
        if len(prices) < window * 2:
            # Find min points that meet threshold
            valid_mins = []
            for i in range(len(prices) - 1):
                if i > 0 and significant_changes[i - 1]:
                    valid_mins.append((i, prices_array[i]))

            return np.argmin(prices_array) if not valid_mins else min(valid_mins, key=lambda x: x[1])[0]

        for i in range(window, len(prices) - window):
            local_window = prices_array[i - window:i + window]
            if prices_array[i] == np.min(local_window):
                # Check if this minimum has significant price movements around it
                if (i > 0 and significant_changes[i - 1]) or (i < len(significant_changes) and significant_changes[i]):
                    local_mins.append((i, prices_array[i]))

        if local_mins:
            return min(local_mins, key=lambda x: x[1])[0]
        else:
            return np.argmin(prices_array)
    else:
        local_maxs = []

        if len(prices) < window * 2:
            # Find max points that meet threshold
            valid_maxs = []
            for i in range(len(prices) - 1):
                if i > 0 and significant_changes[i - 1]:
                    valid_maxs.append((i, prices_array[i]))

            return np.argmax(prices_array) if not valid_maxs else max(valid_maxs, key=lambda x: x[1])[0]

        for i in range(window, len(prices) - window):
            local_window = prices_array[i - window:i + window]
            if prices_array[i] == np.max(local_window):
                # Check if this maximum has significant price movements around it
                if (i > 0 and significant_changes[i - 1]) or (i < len(significant_changes) and significant_changes[i]):
                    local_maxs.append((i, prices_array[i]))

        if local_maxs:
            return max(local_maxs, key=lambda x: x[1])[0]
        else:
            return np.argmax(prices_array)


def find_retracement_extension(prices, dates, threshold=0.3, nested_threshold=0.2, min_change_threshold=0.003,
                               max_days_between=None):
    """
    Find Fibonacci retracement points in price data dynamically.
    Improved version with better starting point selection, minimum price change threshold,
    and time-based constraints between points.

    Now incorporates fixes:
    1. Point B is not fixed until Point C is found
    2. Failed move threshold changed to 0.764
    3. Uses timestamps for time constraints between points
    """
    A = find_best_starting_point(prices, dates, min_change_threshold=min_change_threshold)

    if A is None or A >= len(prices) - 1:
        print("Warning: Invalid starting point. Using index 0 instead.")
        A = 0

    # We'll detect both up and down trends to handle multiple timeframe scenarios
    up_direction = "up"
    down_direction = "down"
    prices_subset = prices[A:]
    dates_subset = dates[A:]

    if len(prices_subset) < 3:
        print("Warning: Not enough data points for analysis.")
        return None

    # Calculate significant price changes
    price_changes = np.abs(np.diff(np.array(prices_subset)) / np.array(prices_subset)[:-1])
    significant_changes = price_changes >= min_change_threshold

    # Initialize variables for potential B and C points
    up_B, up_C, up_D = None, None, None
    down_B, down_C, down_D = None, None, None

    # Track the current highest and lowest points after A
    current_high = prices_subset[0]
    current_high_idx = 0
    current_low = prices_subset[0]
    current_low_idx = 0

    # First, find potential B points (continuously updating until C is found)
    for i in range(1, len(prices_subset)):
        # Check for significant price movement
        sig_change = i > 1 and significant_changes[i - 2]

        # Check time constraint if provided
        time_ok = True
        if max_days_between and i > 0:
            time_diff = dates_subset[i] - dates_subset[0]
            if isinstance(time_diff, timedelta) and time_diff.total_seconds() > max_days_between * 86400:
                time_ok = False

        # Update potential uptrend B point (highest high)
        if prices_subset[i] > current_high and sig_change and time_ok:
            current_high = prices_subset[i]
            current_high_idx = i

        # Update potential downtrend B point (lowest low)
        if prices_subset[i] < current_low and sig_change and time_ok:
            current_low = prices_subset[i]
            current_low_idx = i

        # Once we have potential B points, look for C points

        # For uptrend: C is a retracement from B
        if current_high_idx > 0 and i > current_high_idx:
            # Calculate the move from A to B
            up_move = current_high - prices_subset[0]

            # Find C - point that retraces at least 23.6% but not more than 76.4% from B
            min_retrace = current_high - (up_move * 0.236)
            max_retrace = current_high - (up_move * 0.764)

            # If we find a valid C point
            if prices_subset[i] <= min_retrace and prices_subset[i] >= max_retrace and sig_change:
                # Now we can fix point B
                up_B = current_high_idx
                up_C = i
                # Point B is now fixed at its last highest value before C

                # After finding C, look for D (resumption of trend)
                potential_up_D = None
                for j in range(up_C + 1, len(prices_subset)):
                    # D should be higher than C and ideally higher than B in a strong move
                    if prices_subset[j] > prices_subset[up_C] and sig_change:
                        potential_up_D = j
                        if prices_subset[j] > prices_subset[up_B]:  # Strong move
                            break

                if potential_up_D:
                    up_D = potential_up_D
                    break  # We found a complete up pattern

        # For downtrend: C is a retracement from B
        if current_low_idx > A and i > current_low_idx:
            # Calculate the move from A to B
            down_move = prices_subset[0] - current_low

            # Find C - point that retraces at least 23.6% but not more than 76.4% from B
            min_retrace = current_low + (down_move * 0.236)
            max_retrace = current_low + (down_move * 0.764)

            # If we find a valid C point
            if prices_subset[i] >= min_retrace and prices_subset[i] <= max_retrace and sig_change:
                # Now we can fix point B
                down_B = current_low_idx
                down_C = i

                # After finding C, look for D (resumption of trend)
                potential_down_D = None
                for j in range(down_C + 1, len(prices_subset)):
                    # D should be lower than C and ideally lower than B in a strong move
                    if prices_subset[j] < prices_subset[down_C] and sig_change:
                        potential_down_D = j
                        if prices_subset[j] < prices_subset[down_B]:  # Strong move
                            break

                if potential_down_D:
                    down_D = potential_down_D
                    break  # We found a complete down pattern

    # Determine which pattern (if any) was found and return the result
    if up_B is not None and up_C is not None and up_D is not None:
        result = {
            "A": (A, prices[A]),
            "B": (A + up_B, prices[A + up_B]),
            "C": (A + up_C, prices[A + up_C]),
            "D": (A + up_D, prices[A + up_D]),
            "direction": up_direction
        }
        return result
    elif down_B is not None and down_C is not None and down_D is not None:
        result = {
            "A": (A, prices[A]),
            "B": (A + down_B, prices[A + down_B]),
            "C": (A + down_C, prices[A + down_C]),
            "D": (A + down_D, prices[A + down_D]),
            "direction": down_direction
        }
        return result

    # If we couldn't find a complete pattern, fall back to the old method
    direction = detect_trend_with_ma(prices[A:])

    # From here on, use the old logic but with the corrected failure threshold
    # and proper B point updating until C is found
    # ... (keeping the rest of your original implementation with the fixes)

    # Old method with fixed B point - this is now a fallback method
    if direction == "up":
        potential_Bs = [(i, prices_subset[i]) for i in range(1, len(prices_subset) - 1)
                        if i > 0 and i < len(significant_changes) and significant_changes[i - 1]]
        if potential_Bs:
            B = max(potential_Bs, key=lambda x: x[1])[0]
        else:
            B = max(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])
    else:
        potential_Bs = [(i, prices_subset[i]) for i in range(1, len(prices_subset) - 1)
                        if i > 0 and i < len(significant_changes) and significant_changes[i - 1]]
        if potential_Bs:
            B = min(potential_Bs, key=lambda x: x[1])[0]
        else:
            B = min(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])

    if B is None or B >= len(prices_subset) - 1:
        print("Warning: Could not identify point B.")
        return None

    if direction == "up":
        move = prices_subset[B] - prices_subset[0]
        # Changed from 0.618 to 0.764 for failed move threshold
        retracement_threshold = prices_subset[B] - (move * threshold)
        potential_Cs = []
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] <= retracement_threshold:
                # Check if there's a significant price movement
                if j > B and j - B - 1 < len(significant_changes) and significant_changes[j - B - 1]:
                    potential_Cs.append(j)
                    break

        if potential_Cs:
            C = potential_Cs[0]
        else:
            for j in range(B + 1, len(prices_subset) - 1):
                if prices_subset[j] <= retracement_threshold:
                    C = j
                    break
    else:
        move = prices_subset[0] - prices_subset[B]
        # Changed from 0.618 to 0.764 for failed move threshold
        retracement_threshold = prices_subset[B] + (move * threshold)
        potential_Cs = []
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] >= retracement_threshold:
                # Check if there's a significant price movement
                if j > B and j - B - 1 < len(significant_changes) and significant_changes[j - B - 1]:
                    potential_Cs.append(j)
                    break

        if potential_Cs:
            C = potential_Cs[0]
        else:
            for j in range(B + 1, len(prices_subset) - 1):
                if prices_subset[j] >= retracement_threshold:
                    C = j
                    break

    if C is None:
        print("Warning: Could not identify point C.")
        return None

    if direction == "up":
        if C + 1 >= len(prices_subset):
            print("Warning: Not enough data points after C.")
            return None

        # Find D with significant movement
        potential_Ds = [(i, prices_subset[i]) for i in range(C + 1, len(prices_subset))
                        if i - C - 1 < len(significant_changes) and i - C - 1 >= 0 and significant_changes[i - C - 1]]
        if potential_Ds:
            D = max(potential_Ds, key=lambda x: x[1])[0]
        else:
            D = max(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])
    else:
        if C + 1 >= len(prices_subset):
            print("Warning: Not enough data points after C.")
            return None

        # Find D with significant movement
        potential_Ds = [(i, prices_subset[i]) for i in range(C + 1, len(prices_subset))
                        if i - C - 1 < len(significant_changes) and i - C - 1 >= 0 and significant_changes[i - C - 1]]
        if potential_Ds:
            D = min(potential_Ds, key=lambda x: x[1])[0]
        else:
            D = min(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])

    result = {
        "A": (A, prices[A]),
        "B": (A + B, prices[A + B]),
        "C": (A + C, prices[A + C]),
        "D": (A + D, prices[A + D]),
        "direction": direction
    }
    return result


def draw_fibonacci_levels(ax, prices, dates, result, direction):
    """Draw Fibonacci retracement levels based on identified pattern."""
    if result:
        A, B = result['A'][1], result['B'][1]
    else:
        min_idx = np.argmin(prices)
        max_idx = np.argmax(prices)

        if direction == "up":
            A_value = prices[min_idx]
            B_value = prices[max_idx]
        else:
            A_value = prices[max_idx]
            B_value = prices[min_idx]

        A, B = A_value, B_value

    main_move = B - A if direction == "up" else A - B

    # Updated Fibonacci levels with extended levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 0.854, 1.0, 1.236, 1.618, -0.236, -0.618]
    level_colors = {
        0: '#FF0000',  # Red
        0.236: '#FF7F00',  # Orange
        0.382: '#FFFF00',  # Yellow
        0.5: '#00FF00',  # Green
        0.618: '#0000FF',  # Blue
        0.764: '#4B0082',  # Indigo - critical "failed move" threshold
        0.854: '#708090',  # SlateGray
        1.0: '#8F00FF',  # Violet
        1.236: '#FFA500',  # Orange extension
        1.618: '#32CD32',  # LimeGreen extension
        -0.236: '#FFC0CB',  # Pink
        -0.618: '#800080'  # Purple
    }

    for level in fib_levels:
        fib_price = B - (main_move * level) if direction == "up" else B + (main_move * level)
        ax.axhline(
            y=fib_price,
            color=level_colors.get(level, 'gray'),
            linestyle='--',
            alpha=0.6,
            linewidth=1.5,
            label=f'Fib {level * 100:.1f}%'
        )
        ax.text(
            dates[0],
            fib_price,
            f'{level * 100:.1f}%',
            fontsize=10,
            verticalalignment='center',
            color=level_colors.get(level, 'gray')
        )

    return ax


def plot_diagnostic_graph(prices, dates, result=None):
    """
    Plot a diagnostic graph showing price series, moving averages, and Fibonacci levels.
    Now using datetime values for the x-axis.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(dates, prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series', markersize=3)

    short_window = 20
    long_window = 50

    if len(prices) >= long_window:
        prices_array = np.array(prices)
        short_ma = np.convolve(prices_array, np.ones(short_window) / short_window, mode='valid')
        long_ma = np.convolve(prices_array, np.ones(long_window) / long_window, mode='valid')

        # Plot MAs with date indexing
        ax.plot(dates[short_window - 1:], short_ma, 'g-',
                label=f'{short_window}-period MA', linewidth=2)
        ax.plot(dates[long_window - 1:], long_ma, 'r-',
                label=f'{long_window}-period MA', linewidth=2)

    # Determine the current visual trend for labeling the chart
    # Check if the recent prices are trending up
    recent_window = min(20, len(prices))
    recent_prices = prices[-recent_window:]

    # Use linear regression to determine trend slope
    x = np.arange(recent_window)
    slope, _ = np.polyfit(x, recent_prices, 1)

    # Also check the short MA at the end vs beginning
    if len(prices) >= short_window * 2:
        short_ma_end = np.convolve(prices, np.ones(short_window) / short_window, mode='valid')
        short_ma_trend = short_ma_end[-1] - short_ma_end[-min(10, len(short_ma_end))]
    else:
        short_ma_trend = 0

    # Determine the visual trend from recent data
    visual_direction = "up" if (slope > 0 or short_ma_trend > 0) else "down"

    # If result is provided, use its direction, otherwise use visual direction
    direction = result['direction'] if result else visual_direction
    move_type = "Upward" if direction == "up" else "Downward"
    ax.set_title(f'Price Series Diagnostic Plot - {move_type} Move', fontsize=16)

    if result:
        points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                  'C': ('green', result['C']), 'D': ('blue', result['D'])}

        for label, (color, point) in points.items():
            idx, price = point
            ax.scatter(dates[idx], price, color=color, s=100, zorder=3)
            ax.text(dates[idx], price, label, fontsize=14, fontweight='bold',
                    ha='right', va='bottom', color=color)

        # Draw Fibonacci levels using the detected points
        ax = draw_fibonacci_levels(ax, prices, dates, result, direction)
    else:
        # Draw Fibonacci levels based on min/max if no pattern detected
        ax = draw_fibonacci_levels(ax, prices, dates, None, direction)

    ax.axhline(y=np.mean(prices), color='purple', linestyle='-', linewidth=2, label='Mean Price')

    min_idx = np.argmin(prices)
    max_idx = np.argmax(prices)
    ax.scatter(dates[min_idx], prices[min_idx], color='darkred', label='Min Price', s=100, zorder=4)
    ax.scatter(dates[max_idx], prices[max_idx], color='darkgreen', label='Max Price', s=100, zorder=4)

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels for better readability

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()


def analyze_move(result):
    """Analyze the identified pattern and provide context."""
    if result is None:
        return "No valid pattern found."

    A, B, C, D = result['A'][1], result['B'][1], result['C'][1], result['D'][1]
    direction = result['direction']

    if direction == "up":
        move_distance = B - A
        retracement_ratio = (B - C) / move_distance if move_distance != 0 else 0
        extension_ratio = (D - C) / move_distance if move_distance != 0 else None
    else:
        move_distance = A - B
        retracement_ratio = (C - B) / move_distance if move_distance != 0 else 0
        extension_ratio = (C - D) / move_distance if move_distance != 0 else None

    # Updated thresholds as per feedback
    if retracement_ratio < 0.382:
        status = "Progressive move (shallow retracement)"
    elif retracement_ratio >= 0.764:  # Changed from 0.618 to 0.764
        status = "Failed move (deep retracement)"
    elif C < A and direction == "up":  # Added check if C breaks below A in uptrend
        status = "Failed move (C below A)"
    elif C > A and direction == "down":  # Added check if C breaks above A in downtrend
        status = "Failed move (C above A)"
    else:
        # Now we have "in progress" as a potential state
        if D > B and direction == "up":
            status = "Successful move (reached new high)"
        elif D < B and direction == "down":
            status = "Successful move (reached new low)"
        else:
            status = "In progress (moderate retracement)"

    if extension_ratio is not None:
        if extension_ratio > 1.0:
            status += " with strong extension"
        elif extension_ratio < 0.5:
            status += " with weak extension"

    return status


def main():
    # Read the CSV file with datetime parsing
    df = pd.read_csv("BTCUSDT-hourly-historical-price.csv")

    # Check first few rows to understand the structure
    print("First few rows of the CSV file:")
    print(df.head(2))

    # Based on the error, the timestamp format appears to be in ISO format
    # Detect if the timestamp is the first column or has a specific name
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        # Assume it's the first column
        timestamp_col = df.columns[0]

    print(f"Using column '{timestamp_col}' as timestamp")

    # Try to parse the timestamp with ISO format
    df['timestamp'] = pd.to_datetime(df[timestamp_col], format='mixed')

    # Use a subset of data for analysis (adjust as needed)
    subset = df.iloc[200:min(300, len(df))]  # Ensure we don't exceed dataset length

    # Extract prices and timestamps together, dropping rows where close is NaN
    clean_subset = subset.dropna(subset=["close"])

    prices = clean_subset["close"].tolist()
    dates = clean_subset["timestamp"].tolist()

    # Set minimum price change threshold to 0.3%
    min_change_threshold = 0.003

    # Optional: set maximum days between points (e.g., 30 days)
    max_days_between = None  # Set to a number if needed

    # Check the recent trend for more accurate labeling
    recent_window = min(20, len(prices))
    recent_prices = prices[-recent_window:]
    recent_slope, _ = np.polyfit(range(recent_window), recent_prices, 1)
    visual_trend = "up" if recent_slope > 0 else "down"

    print(f"Visual trend based on recent prices: {visual_trend}")
    print(f"Moving Average Trend Detection: {detect_trend_with_ma(prices)}")

    # Pass the dates and other parameters to the function
    result = find_retracement_extension(
        prices,
        dates,
        min_change_threshold=min_change_threshold,
        max_days_between=max_days_between
    )

    if result:
        plot_diagnostic_graph(prices, dates, result)
        analysis = analyze_move(result)

        print(f"Detected trend: {result['direction']}")
        print(f"Identified points:")
        for point, point_info in result.items():
            if point != 'direction':
                idx, price = point_info
                print(f"{point}: Index {idx}, Date {dates[idx].strftime('%Y-%m-%d %H:%M')}, Price {price}")
        print(f"Analysis: {analysis}")

        A, B = result['A'][1], result['B'][1]
        main_move = B - A if result['direction'] == "up" else A - B
        print("\nKey Fibonacci levels:")
        # Updated Fibonacci levels including the 0.764 level
        for level in [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, 1.236, 1.618, -0.236, -0.618]:
            fib_price = B - (main_move * level) if result['direction'] == "up" else B + (main_move * level)
            print(f"{level * 100:.1f}%: {fib_price:.2f}")
    else:
        plot_diagnostic_graph(prices, dates)
        print("Could not find a valid Fibonacci retracement pattern.")
        print("A diagnostic graph has been generated with Fibonacci levels based on min/max prices.")

    plt.show()


if __name__ == "__main__":
    main()