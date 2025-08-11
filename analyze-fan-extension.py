import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import json
import os
from analyze import (
    load_and_prepare_data,
    analyze_multiple_windows
)
def find_index_from_timestamp(dates, target_timestamp):
    """Find the index corresponding to a timestamp"""
    try:
        return dates.index(target_timestamp)
    except ValueError:
        import bisect
        if isinstance(target_timestamp, str):
            target_timestamp = pd.to_datetime(target_timestamp)
        date_objects = [pd.to_datetime(d) if isinstance(d, str) else d for d in dates]
        target_obj = pd.to_datetime(target_timestamp) if isinstance(target_timestamp, str) else target_timestamp
        pos = bisect.bisect_left(date_objects, target_obj)
        if pos == 0:
            return 0
        elif pos == len(date_objects):
            return len(date_objects) - 1
        else:
            before = date_objects[pos - 1]
            after = date_objects[pos]
            if abs((target_obj - before).total_seconds()) < abs((after - target_obj).total_seconds()):
                return pos - 1
            else:
                return pos

def load_parameters():
    """Load parameters from parameters.json"""
    with open('parameters.json', 'r') as f:
        params = json.load(f)
    return params

def calculate_382_retracement_level(pattern):
    """
    Calculate the 38.2% retracement level of the AB move
    UPDATED: Handles both uptrend and downtrend patterns correctly

    Args:
        pattern (dict): Pattern dictionary with A, B, C, D points

    Returns:
        float: Price level at 38.2% retracement of AB
    """
    # Get A and B prices
    a_price = pattern['A'][1]
    b_price = pattern['B'][1]

    # Get direction from pattern
    direction = pattern.get('direction', 'unknown')

    if direction == 'up':
        # UPTREND: A is low, B is high
        # 38.2% retracement from B back toward A
        move_size = b_price - a_price
        retracement_382 = b_price - (move_size * 0.382)
        print(f"  UPTREND: A=${a_price:.2f}, B=${b_price:.2f}, Move=${move_size:.2f}")
        print(f"  38.2% retracement from B: ${retracement_382:.2f}")
    elif direction == 'down':
        # DOWNTREND: A is high, B is low
        # 38.2% retracement from B back toward A
        move_size = a_price - b_price
        retracement_382 = b_price + (move_size * 0.382)
        print(f"  DOWNTREND: A=${a_price:.2f}, B=${b_price:.2f}, Move=${move_size:.2f}")
        print(f"  38.2% retracement from B: ${retracement_382:.2f}")
    else:
        # Fallback: determine direction from prices
        if b_price > a_price:
            # Looks like uptrend
            move_size = b_price - a_price
            retracement_382 = b_price - (move_size * 0.382)
            print(f"  AUTO-DETECTED UPTREND: A=${a_price:.2f}, B=${b_price:.2f}")
        else:
            # Looks like downtrend
            move_size = a_price - b_price
            retracement_382 = b_price + (move_size * 0.382)
            print(f"  AUTO-DETECTED DOWNTREND: A=${a_price:.2f}, B=${b_price:.2f}")

    return retracement_382


def find_highest_after_d(prices, dates, d_timestamp, d_price, min_change):
    d_index = find_index_from_timestamp(dates, d_timestamp)
    if d_index >= len(prices) - 1:
        return None
    for i in range(d_index + 1, len(prices)):
        current_price = prices[i]
        price_change = (current_price - d_price) / d_price
        if price_change >= min_change:
            return (dates[i], current_price)
    return None

def find_lowest_after_d(prices, dates, d_timestamp, d_price, min_change):
    d_index = find_index_from_timestamp(dates, d_timestamp)
    if d_index >= len(prices) - 1:
        return None
    for i in range(d_index + 1, len(prices)):
        current_price = prices[i]
        price_change = (d_price - current_price) / d_price
        if price_change >= min_change:
            return (dates[i], current_price)
    return None


def analyze_fan_extension(pattern, prices, dates, min_change=0.01):
    """
    Analyze fan extension from 38.2% level of AB to first significant high/low after D
    UPDATED: Properly handles both uptrend and downtrend patterns

    Args:
        pattern: Pattern dictionary with TIMESTAMP-based points
        prices: Price array
        dates: Dates array
        min_change: Minimum percentage change from D to consider a new point

    Returns:
        dict: Analysis results including 38.2% level of AB and first significant high/low
    """
    # Calculate 38.2% retracement level of AB
    retracement_382 = calculate_382_retracement_level(pattern)

    # Get D point details - NOW EXPECTING TIMESTAMP
    d_timestamp = pattern['D'][0]  # This is now a timestamp, not index
    d_price = pattern['D'][1]

    # Get pattern direction
    direction = pattern.get('direction', 'unknown')
    print(f"\nAnalyzing {direction.upper()} pattern fan extension:")
    print(f"  D point: {d_timestamp}, ${d_price:.2f}")
    print(f"  38.2% level: ${retracement_382:.2f}")

    # Find first significant move after D based on direction
    e_point = None
    analysis_key = None

    if direction == 'down':
        # DOWNTREND: Look for first significant LOW after D
        print(f"  Looking for first significant LOW after D (min change: {min_change * 100:.1f}%)")
        e_point = find_lowest_after_d(prices, dates, d_timestamp, d_price, min_change)
        analysis_key = 'lowest_after_d'

        if e_point:
            e_timestamp, e_price = e_point
            print(f"  Found E (Low): {e_timestamp}, ${e_price:.2f}")

            # For downtrend, we want the move FROM 38.2% level DOWN to E
            # This should be a DOWNWARD move from the 38.2% level
            move_from_382 = retracement_382 - e_price  # Should be positive for downward move
            move_pct = (move_from_382 / retracement_382) * 100

            print(f"  Move from 38.2% level: ${move_from_382:.2f} ({move_pct:.1f}%)")
        else:
            print(f"  No significant low found after D")

    elif direction == 'up':
        # UPTREND: Look for first significant HIGH after D
        print(f"  Looking for first significant HIGH after D (min change: {min_change * 100:.1f}%)")
        e_point = find_highest_after_d(prices, dates, d_timestamp, d_price, min_change)
        analysis_key = 'highest_after_d'

        if e_point:
            e_timestamp, e_price = e_point
            print(f"  Found E (High): {e_timestamp}, ${e_price:.2f}")

            # For uptrend, we want the move FROM 38.2% level UP to E
            # This should be an UPWARD move from the 38.2% level
            move_from_382 = e_price - retracement_382  # Should be positive for upward move
            move_pct = (move_from_382 / retracement_382) * 100

            print(f"  Move from 38.2% level: ${move_from_382:.2f} ({move_pct:.1f}%)")
        else:
            print(f"  No significant high found after D")
    else:
        print(f"  Unknown direction '{direction}' - defaulting to uptrend logic")
        e_point = find_highest_after_d(prices, dates, d_timestamp, d_price, min_change)
        analysis_key = 'highest_after_d'

    # Build analysis results
    analysis = {
        'pattern': pattern,
        'retracement_382_level': retracement_382,
        analysis_key: e_point,
        'd_timestamp': d_timestamp,
        'd_price': d_price,
        'pattern_direction': direction
    }

    if e_point:
        e_timestamp, e_price = e_point

        # Calculate the move from 38.2% level (direction-aware)
        if direction == 'down':
            move_from_382 = retracement_382 - e_price  # Downward move
            move_pct = (move_from_382 / retracement_382) * 100
        else:  # 'up' or unknown
            move_from_382 = e_price - retracement_382  # Upward move
            move_pct = (move_from_382 / retracement_382) * 100

        analysis['move_from_382'] = move_from_382
        analysis['move_from_382_pct'] = move_pct

        # Calculate move from D (direction-aware)
        if direction == 'down':
            move_from_d = d_price - e_price  # Downward move from D
            move_from_d_pct = (move_from_d / d_price) * 100
        else:  # 'up' or unknown
            move_from_d = e_price - d_price  # Upward move from D
            move_from_d_pct = (move_from_d / d_price) * 100

        analysis['move_from_d'] = move_from_d
        analysis['move_from_d_pct'] = move_from_d_pct

        print(f"  Final analysis: Move from D = ${move_from_d:.2f} ({move_from_d_pct:.1f}%)")

    return analysis


def plot_pattern_with_extension(analysis, prices, dates, min_change=0.01):
    """
    Plot pattern with 38.2% level and extension - HANDLES BOTH DIRECTIONS
    """
    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

    plt.figure(figsize=(15, 10))

    # Debug information - NOW SHOWING TIMESTAMPS
    print(f"\nDEBUG - {direction.upper()} Pattern timestamps:")
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        print(f"  {point}: {timestamp}, price=${price:.2f}")

    # Convert timestamps to indices for plotting calculations
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)
        print(f"  {point} timestamp {timestamp} → index {idx}")

    min_pattern_idx = min(pattern_indices)
    max_pattern_idx = max(pattern_indices)

    # Include the E point if it exists (works for both high and low)
    e_point = None
    e_label = None

    if analysis.get('lowest_after_d'):
        e_timestamp, e_price = analysis['lowest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx)
        e_point = (e_timestamp, e_price)
        e_label = "E (Low)"
        print(f"  E (Lowest): {e_timestamp} → index {e_idx}, price=${e_price:.2f}")
    elif analysis.get('highest_after_d'):
        e_timestamp, e_price = analysis['highest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx)
        e_point = (e_timestamp, e_price)
        e_label = "E (High)"
        print(f"  E (Highest): {e_timestamp} → index {e_idx}, price=${e_price:.2f}")
    else:
        max_display_idx = max_pattern_idx

    # Calculate display range with padding
    pattern_range = max_pattern_idx - min_pattern_idx
    padding = max(50, int(pattern_range * 0.5))
    start_idx = max(0, min_pattern_idx - padding)
    end_idx = min(len(prices) - 1, max_display_idx + padding)

    print(f"\nDEBUG - Display range: {start_idx} to {end_idx}")

    # Plot the subset of price data using TIMESTAMPS for x-axis
    subset_prices = prices[start_idx:end_idx + 1]
    subset_dates = dates[start_idx:end_idx + 1]

    if len(subset_prices) == 0:
        print("ERROR: No price data in display range!")
        return

    # Plot price data with TIMESTAMPS on x-axis
    plt.plot(subset_dates, subset_prices, 'b-', alpha=0.5, linewidth=0.8, label='Price')

    # Plot pattern points using TIMESTAMPS
    points = ['A', 'B', 'C', 'D']
    point_colors = {'A': 'black', 'B': 'red', 'C': 'orange', 'D': 'blue'}

    for point in points:
        timestamp, price = pattern[point]
        color = point_colors.get(point, 'gray')
        plt.plot(timestamp, price, 'o', color=color, markersize=8, zorder=5)
        plt.text(timestamp, price, f'{point}\n${price:.2f}', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)

    # Draw lines between pattern points using TIMESTAMPS
    for i in range(len(points) - 1):
        timestamp1, price1 = pattern[points[i]]
        timestamp2, price2 = pattern[points[i + 1]]
        plt.plot([timestamp1, timestamp2], [price1, price2], 'r-', linewidth=2, zorder=4)

    # Plot 38.2% retracement level of AB
    retracement_level = analysis['retracement_382_level']
    plt.axhline(y=retracement_level, color='green', linestyle='--', alpha=0.7,
                label=f'38.2% of AB: ${retracement_level:.2f}', zorder=3)

    # Plot E point and fan line if exists
    if e_point:
        e_timestamp, e_price = e_point

        # Choose color based on direction
        e_color = 'purple' if direction == 'down' else 'lime'

        plt.plot(e_timestamp, e_price, 'o', color=e_color, markersize=10, zorder=5)

        # Position text based on direction to avoid overlap
        va_pos = 'top' if direction == 'down' else 'bottom'
        plt.text(e_timestamp, e_price, f'{e_label}\n${e_price:.2f}', ha='center', va=va_pos, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=e_color, alpha=0.3), zorder=6)

        # Draw fan line from 38.2% level to E point
        d_timestamp = pattern['D'][0]
        plt.plot([d_timestamp, e_timestamp], [retracement_level, e_price],
                 color=e_color, linewidth=2, alpha=0.7, label=f'{direction.upper()} Extension', zorder=4)

        # Add percentage text on the fan line
        if 'move_from_382_pct' in analysis:
            # Position text in middle of fan line
            mid_x = d_timestamp + (e_timestamp - d_timestamp) / 2
            mid_y = retracement_level + (e_price - retracement_level) / 2

            plt.text(mid_x, mid_y, f'{analysis["move_from_382_pct"]:.1f}%',
                     ha='center', va='center', fontsize=9, color=e_color, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)

    # Format x-axis for timestamps
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Price ($)')

    title = f'{direction.upper()}TREND Pattern with 38.2% Fan Extension'
    if e_point:
        move_type = "Low" if direction == 'down' else "High"
        title += f'\nExtension to {move_type}: {analysis.get("move_from_382_pct", 0):.1f}% from 38.2% level'

    plt.title(title)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()


def get_completed_patterns_for_date(date_str, min_change=0.01):  # Default 1% change
    """
    Load data for a specific date and get all completed patterns with fan extension analysis

    Args:
        date_str (str): Date in YYYY-MM-DD format
        min_change: Minimum percentage change (e.g., 0.01 for 1%) from D to consider a new point

    Returns:
        dict: All completed patterns and data for that date with extension analysis
    """
    print(f"\n=== FAN EXTENSION ANALYSIS ===")
    print(f"Getting completed patterns for date: {date_str}")

    # Load parameters
    params = load_parameters()
    possible_files = [
        f"C:/Users/admin/Desktop/btc_minute_data/btc_minute_data_{date_str}.csv",
        f"btc_minute_data_{date_str}.csv",
        f"./btc_minute_data_{date_str}.csv"
    ]
    data_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            data_file = file_path
            print(f"Found data file: {data_file}")
            break




    if not data_file:
        print(f"Data file not found in any of these locations:")
        for path in possible_files:
            print(f"  - {path}")
        return {"status": "file_not_found", "date": date_str}

    # Load data using analyze.py
    prices, dates, df = load_and_prepare_data(data_file)
    print(f"Loaded {len(prices)} data points")

    # Get all patterns using analyze.py
    all_patterns = analyze_multiple_windows(
        prices,
        dates,
        window_sizes=params['window_sizes'],
        overlap_percent=params['overlap'],
        min_change_threshold=params['min_change'],
        pattern_config=params['pattern_detection']
    )

    # Filter for completed patterns only
    completed_patterns = []
    for pattern_data in all_patterns:
        pattern, analysis, window_info = pattern_data
        if isinstance(pattern, list):
            for p in pattern:
                if p.get('status') == 'completed':
                    completed_patterns.append(p)
        else:
            if pattern.get('status') == 'completed':
                completed_patterns.append(pattern)

    print(f"Total patterns found: {len(all_patterns)}")
    print(f"Completed patterns: {len(completed_patterns)}")

    # Analyze fan extensions for each completed pattern
    extension_analyses = []

    for i, pattern in enumerate(completed_patterns):
        print(f"\n=== Completed Pattern {i + 1} ===")
        print(f"  Direction: {pattern.get('direction', 'unknown')}")
        print(f"  A: {pattern['A'][0]}, Price ${pattern['A'][1]:.2f}")
        print(f"  B: {pattern['B'][0]}, Price ${pattern['B'][1]:.2f}")
        print(f"  C: {pattern['C'][0]}, Price ${pattern['C'][1]:.2f}")
        print(f"  D: {pattern['D'][0]}, Price ${pattern['D'][1]:.2f}")
        print(f"  Initial Move: {pattern.get('initial_move_pct', 0):.2f}%")
        print(f"  Retracement: {pattern.get('retracement_pct', 0):.2f}%")

        # Analyze fan extension
        extension_analysis = analyze_fan_extension(pattern, prices, dates, min_change)
        extension_analyses.append(extension_analysis)

        print(f"\n  Fan Extension Analysis:")
        print(f"    38.2% Retracement Level of AB: ${extension_analysis['retracement_382_level']:.2f}")
        if extension_analysis.get('lowest_after_d'):
            low_timestamp, low_price = extension_analysis['lowest_after_d']
            print(f"    E (Lowest) After D: {low_timestamp}, Price ${low_price:.2f}")
        elif extension_analysis.get('highest_after_d'):
            high_timestamp, high_price = extension_analysis['highest_after_d']
            print(f"    E (Highest) After D: {high_timestamp}, Price ${high_price:.2f}")
        else:
            print(f"    No significant move after point D within {min_change*100}%")

    return {
        "status": "success",
        "date": date_str,
        "completed_patterns": completed_patterns,
        "extension_analyses": extension_analyses,
        "prices": prices,
        "dates": dates,
        "total_patterns": len(all_patterns),
        "completed_count": len(completed_patterns),
        "ready_for_fan_extension": True
    }
if __name__ == "__main__":
    # Example usage - choose your date here
    date_to_analyze = "2025-07-22"
    params = load_parameters()
    min_change = params.get('min_change', 0.001)

    # Set to True if you want to see plots
    show_plots = True

    results = get_completed_patterns_for_date(date_to_analyze, min_change)

    if results["status"] == "success":
        print(f"\n=== READY FOR FAN EXTENSION ===")
        print(f"Found {results['completed_count']} completed patterns")

        if show_plots and results['extension_analyses']:
            print("\nGenerating plots for patterns with extensions...")
            for i, analysis in enumerate(results['extension_analyses']):
                if analysis.get('lowest_after_d') or analysis.get('highest_after_d'):
                    print(f"\nPlotting pattern {i + 1}...")
                    plot_pattern_with_extension(analysis, results['prices'], results['dates'], min_change)
    else:
        print(f"Error: {results['status']}")