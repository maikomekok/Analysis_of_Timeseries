import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def validate_pattern_structure(pattern):
    """
    Validate that the pattern structure follows harmonic pattern rules.
    Returns True if valid, False otherwise.
    """
    A_idx, A_price = pattern['A']
    B_idx, B_price = pattern['B']
    C_idx, C_price = pattern['C']
    D_idx, D_price = pattern['D']
    direction = pattern['direction']

    # Basic sequence validation
    if not (A_idx < B_idx < C_idx < D_idx):
        print("DEBUG: Invalid point sequence")
        return False

    # For uptrend patterns
    if direction == 'up':
        # B must be higher than A
        if B_price <= A_price:
            print(f"DEBUG: Invalid up-trend - B ({B_price}) not higher than A ({A_price})")
            return False

        # C must be lower than B (retracement)
        if C_price >= B_price:
            print(f"DEBUG: Invalid up-trend - C ({C_price}) not lower than B ({B_price})")
            return False

        # C should not go below A (too deep retracement)
        if C_price < A_price:
            print(f"DEBUG: Invalid up-trend - C ({C_price}) below A ({A_price})")
            return False

    # For downtrend patterns
    else:
        # B must be lower than A
        if B_price >= A_price:
            print(f"DEBUG: Invalid down-trend - B ({B_price}) not lower than A ({A_price})")
            return False

        # C must be higher than B (retracement)
        if C_price <= B_price:
            print(f"DEBUG: Invalid down-trend - C ({C_price}) not higher than B ({B_price})")
            return False

        # C should not go above A (too deep retracement)
        if C_price > A_price:
            print(f"DEBUG: Invalid down-trend - C ({C_price}) above A ({A_price})")
            return False

    return True

def detect_trend_with_ma(prices, short_window=20, long_window=50):
    """
    LEGACY FUNCTION: Kept for backward compatibility.
    Detects trend direction using moving averages.
    """
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


def detect_multiple_timeframe_trends(prices, windows=[(5, 10), (20, 50), (50, 200)]):
    """
    LEGACY FUNCTION: Kept for backward compatibility.
    Detects trends across multiple timeframes.
    """
    trends = {}

    for short_window, long_window in windows:
        if len(prices) < long_window:
            continue

        trend_name = f"{short_window}_{long_window}"
        trends[trend_name] = detect_trend_with_ma(prices, short_window, long_window)

    if trends:
        up_count = sum(1 for t in trends.values() if t == "up")
        down_count = sum(1 for t in trends.values() if t == "down")

        if up_count > down_count:
            trends["overall"] = "up"
        elif down_count > up_count:
            trends["overall"] = "down"
        else:
            trends["overall"] = "conflicting"
    else:
        trends["overall"] = detect_trend_with_ma(prices)

    return trends


def find_significant_price_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Find significant price patterns using configurable parameters.
    Improved version with better D point placement for failed/completed patterns.

    Args:
        prices: List of price values
        dates: List of corresponding dates
        min_change_pct: Minimum percentage change to consider significant (default 0.5%)
        config: Dictionary with configuration parameters (defaults will be used if None)

    Returns:
        List of detected patterns
    """
    # print(f"Minimum required move: {min_change_pct * 100 * min_move_multiplier:.3f}%")
    # Use default configuration if none provided
    if config is None:
        config = {}

    # Get configuration parameters with defaults
    retracement_target = config.get("retracement_target", 0.5)  # Default 50%
    failure_level = config.get("failure_level", 0.8)  # Default 80%
    completion_extension = config.get("completion_extension", 0.236)  # Default -23.6%
    retracement_tolerance = config.get("retracement_tolerance", 0.05)  # Default ±10% of target
    use_strict_swing_points = config.get("use_strict_swing_points", False)  # Default to flexible
    min_move_multiplier = config.get("min_move_multiplier", 1.0)  # Default 2x threshold
    only_show_significant = config.get("only_show_significant", False)  # Default show all patterns
    significant_patterns_only = config.get("significant_patterns_only",
                                           ["completed", "failed"])  # Default significant pattern statuses

    print(f"\nDEBUG: Analyzing {len(prices)} price points with min_change_pct={min_change_pct}")
    print(f"DEBUG: Price range: min={min(prices):.2f}, max={max(prices):.2f}")
    print(f"DEBUG: Percentage difference between min and max: {(max(prices) - min(prices)) / min(prices) * 100:.2f}%")
    print(f"DEBUG: Configuration: {config}")

    if len(prices) < 4:
        print("Not enough data points for pattern detection")
        return []

    # Initialize list of patterns
    patterns = []

    # Find all highs and lows using either strict or flexible criteria
    highs = []  # (index, price)
    lows = []  # (index, price)

    if use_strict_swing_points:
        # Strict criteria: Find significant swing points
        for i in range(1, len(prices) - 1):
            # Get current and adjacent prices
            prev_price = prices[i - 1]
            current_price = prices[i]
            next_price = prices[i + 1]

            # Calculate percentage changes
            change_from_prev = abs(current_price - prev_price) / prev_price
            change_to_next = abs(next_price - current_price) / current_price

            # Check if both changes are significant
            if change_from_prev >= min_change_pct and change_to_next >= min_change_pct:
                # Check if this is a peak (swing high)
                if current_price > prev_price and current_price > next_price:
                    highs.append((i, current_price))

                # Check if this is a valley (swing low)
                elif current_price < prev_price and current_price < next_price:
                    lows.append((i, current_price))
    else:
        # Flexible criteria: Find local highs and lows
        for i in range(1, len(prices) - 1):
            # Check if this is a peak (local high)
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                highs.append((i, prices[i]))

            # Check if this is a valley (local low)
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                lows.append((i, prices[i]))

    print(f"DEBUG: Found {len(highs)} highs and {len(lows)} lows")

    # If we don't have enough swing points, we can't form patterns
    if len(highs) < 1 or len(lows) < 1:
        print("Not enough highs and lows detected")
        return []

    # Try to find uptrend patterns (low to high to low to high)
    for i, (low_idx, low_price) in enumerate(lows):
        # Find the next significant high after this low
        # Find the next significant high after this low (B point)
        next_highs = [(idx, price) for idx, price in highs if idx > low_idx]
        if not next_highs:
            continue

        # Choose the highest high, not just the first one
        high_idx, high_price = max(next_highs, key=lambda x: x[1])

        # Calculate initial move (A to B)
        initial_move = high_price - low_price
        move_pct = initial_move / low_price * 100

        # Only consider moves that exceed the minimum threshold multiplier
        if move_pct < min_change_pct * 100 * min_move_multiplier:
            continue

        # Calculate the target retracement level based on configuration
        target_retrace = high_price - (initial_move * retracement_target)

        # Calculate the allowed range based on tolerance
        min_retrace = high_price - (initial_move * (retracement_target + retracement_tolerance))
        max_retrace = high_price - (initial_move * (retracement_target - retracement_tolerance))

        # Find the next low after B (potential C) that's closest to the target retracement
        potential_cs = [(idx, price) for idx, price in lows if idx > high_idx]

        if not potential_cs:
            continue

        # Find the retracement point closest to our target
        retrace_idx, retrace_price = None, None
        best_distance = float('inf')

        for idx, price in potential_cs:
            if min_retrace <= price <= max_retrace:
                distance = abs(price - target_retrace)
                if distance < best_distance:
                    best_distance = distance
                    retrace_idx, retrace_price = idx, price

        # If we didn't find a close match, just use the first retracement
        if retrace_idx is None:
            retrace_idx, retrace_price = potential_cs[0]


        # Calculate retracement percentage
        retracement = high_price - retrace_price
        retrace_pct = retracement / initial_move * 100 if initial_move != 0 else 0

        # Calculate the failure level
        failure_level_price = high_price - (initial_move * failure_level)

        # Calculate the completion level (-23.6% extension)
        completion_level_price = high_price + (initial_move * completion_extension)

        # Find potential D candidates (next highs after C)
        next_highs_after_c = [(idx, price) for idx, price in highs if idx > retrace_idx]

        # Default values for D point
        d_idx = len(prices) - 1  # Default to last point in window
        d_price = prices[d_idx]  # Default to last price

        # Look for candidate D points
        if next_highs_after_c:
            # Get the first high after C as a candidate
            d_candidate_idx, d_candidate_price = next_highs_after_c[0]

            # Update the default D
            d_idx = d_candidate_idx
            d_price = d_candidate_price

            # Check if we have more highs to see if pattern has progressed further
            if len(next_highs_after_c) > 1:
                # Look for the highest high after C as another candidate
                highest_high_idx, highest_high_price = max(next_highs_after_c, key=lambda x: x[1])

                if highest_high_price > d_price:
                    # If we found a higher high, use that instead
                    d_idx = highest_high_idx
                    d_price = highest_high_price

        # Determine pattern status
        if retrace_price <= failure_level_price:
            status = "failed"
            # IMPROVEMENT: For failed patterns, place D at the failure level
            # Find the closest date index to when price hit the failure level
            for j in range(retrace_idx + 1, len(prices)):
                if j >= len(prices):
                    break
                # If price drops below failure level
                if prices[j] <= failure_level_price:
                    d_idx = j
                    d_price = failure_level_price
                    break
        elif d_price >= high_price:
            if d_price >= completion_level_price:
                status = "completed"
                # IMPROVEMENT: For completed patterns, make sure D is at or beyond completion level
                # If we've already found a high beyond the completion level, use that
                # otherwise place D at the completion level
                if d_price < completion_level_price:
                    # Find where price crosses completion level
                    for j in range(retrace_idx + 1, len(prices)):
                        if j >= len(prices):
                            break
                        if prices[j] >= completion_level_price:
                            d_idx = j
                            d_price = completion_level_price
                            break
            else:
                status = "progressive"
        else:
            status = "in_progress"
        # Create pattern
        pattern = {
            "direction": "up",
            "A": (low_idx, low_price),
            "B": (high_idx, high_price),
            "C": (retrace_idx, retrace_price),
            "D": (d_idx, d_price),
            "initial_move_pct": move_pct,
            "retracement_pct": retrace_pct,
            "status": status,
            "failure_level": failure_level_price,
            "completion_level": completion_level_price,
            "target_level": target_retrace,
            "long_term": False
        }

        # Add pattern validation
        if validate_pattern_structure(pattern):
            patterns.append(pattern)
            print(
                f"Found valid uptrend pattern: A={low_price:.2f}, B={high_price:.2f}, C={retrace_price:.2f}, D={d_price:.2f}")
        else:
            print(
                f"Rejected invalid uptrend pattern: A={low_price:.2f}, B={high_price:.2f}, C={retrace_price:.2f}, D={d_price:.2f}")

        # Filter patterns based on config if requested
        if only_show_significant and status not in significant_patterns_only:
            continue

        patterns.append(pattern)
        print(f"Found uptrend pattern: A={low_price:.2f}, B={high_price:.2f}, C={retrace_price:.2f}, D={d_price:.2f}")
        print(
            f"  Status: {status}, Failure level: {failure_level_price:.2f}, Completion level: {completion_level_price:.2f}")
        print(f"  Move: {move_pct:.2f}%, Retracement: {retrace_pct:.2f}%")

    # Try to find downtrend patterns (high to low to high to low)
    for i, (high_idx, high_price) in enumerate(highs):
        # Find the next significant low after this high
        # Find the next significant low after this high (B point)
        next_lows = [(idx, price) for idx, price in lows if idx > high_idx]
        if not next_lows:
            continue

        # Choose the lowest low, not just the first one
        low_idx, low_price = min(next_lows, key=lambda x: x[1])

        # Calculate initial move (A to B)
        initial_move = high_price - low_price
        move_pct = initial_move / high_price * 100

        # Only consider moves that exceed the minimum threshold multiplier
        if move_pct < min_change_pct * 100 * min_move_multiplier:
            continue

        # Calculate the target retracement level based on configuration
        target_retrace = low_price + (initial_move * retracement_target)

        # Calculate the allowed range based on tolerance
        min_retrace = low_price + (initial_move * (retracement_target - retracement_tolerance))
        max_retrace = low_price + (initial_move * (retracement_target + retracement_tolerance))

        # Find the next high after B (potential C) that's closest to the target retracement
        potential_cs = [(idx, price) for idx, price in highs if idx > low_idx]

        if not potential_cs:
            continue

        # Find the retracement point closest to our target
        retrace_idx, retrace_price = None, None
        best_distance = float('inf')

        for idx, price in potential_cs:
            if min_retrace <= price <= max_retrace:
                distance = abs(price - target_retrace)
                if distance < best_distance:
                    best_distance = distance
                    retrace_idx, retrace_price = idx, price

        # If we didn't find a close match, just use the first retracement
        if retrace_idx is None:
            retrace_idx, retrace_price = potential_cs[0]

        # Calculate retracement percentage
        retracement = retrace_price - low_price
        retrace_pct = retracement / initial_move * 100 if initial_move != 0 else 0

        # Calculate the failure level
        failure_level_price = low_price + (initial_move * failure_level)

        # Calculate the completion level (-23.6% extension)
        completion_level_price = low_price - (initial_move * completion_extension)

        # Find potential D candidates (next lows after C)
        next_lows_after_c = [(idx, price) for idx, price in lows if idx > retrace_idx]

        # Default values for D point
        d_idx = len(prices) - 1  # Default to last point in window
        d_price = prices[d_idx]  # Default to last price

        # Look for candidate D points
        if next_lows_after_c:
            # Get the first low after C as a candidate
            d_candidate_idx, d_candidate_price = next_lows_after_c[0]

            # Update the default D
            d_idx = d_candidate_idx
            d_price = d_candidate_price

            # Check if we have more lows to see if pattern has progressed further
            if len(next_lows_after_c) > 1:
                # Look for the lowest low after C as another candidate
                lowest_low_idx, lowest_low_price = min(next_lows_after_c, key=lambda x: x[1])

                if lowest_low_price < d_price:
                    # If we found a lower low, use that instead
                    d_idx = lowest_low_idx
                    d_price = lowest_low_price

        # Determine pattern status
        if retrace_price >= failure_level_price:
            status = "failed"
            # IMPROVEMENT: For failed patterns, place D at the failure level
            # Find the closest date index to when price hit the failure level
            for j in range(retrace_idx + 1, len(prices)):
                if j >= len(prices):
                    break
                # If price rises above failure level
                if prices[j] >= failure_level_price:
                    d_idx = j
                    d_price = failure_level_price
                    break
        elif d_price <= low_price:
            if d_price <= completion_level_price:
                status = "completed"
                # IMPROVEMENT: For completed patterns, make sure D is at or beyond completion level
                # If we've already found a low beyond the completion level, use that
                # otherwise place D at the completion level
                if d_price > completion_level_price:
                    # Find where price crosses completion level
                    for j in range(retrace_idx + 1, len(prices)):
                        if j >= len(prices):
                            break
                        if prices[j] <= completion_level_price:
                            d_idx = j
                            d_price = completion_level_price
                            break
            else:
                status = "progressive"
        else:
            status = "in_progress"

        # Create pattern
        pattern = {
            "direction": "down",
            "A": (high_idx, high_price),
            "B": (low_idx, low_price),
            "C": (retrace_idx, retrace_price),
            "D": (d_idx, d_price),
            "initial_move_pct": move_pct,
            "retracement_pct": retrace_pct,
            "status": status,
            "failure_level": failure_level_price,
            "completion_level": completion_level_price,
            "target_level": target_retrace,
            "long_term": False
        }

        # Filter patterns based on config if requested
        if only_show_significant and status not in significant_patterns_only:
            continue

        patterns.append(pattern)
        print(f"Found downtrend pattern: A={high_price:.2f}, B={low_price:.2f}, C={retrace_price:.2f}, D={d_price:.2f}")
        print(
            f"  Status: {status}, Failure level: {failure_level_price:.2f}, Completion level: {completion_level_price:.2f}")
        print(f"  Move: {move_pct:.2f}%, Retracement: {retrace_pct:.2f}%")

    # Sort patterns by the significance of their moves (largest first)
    patterns.sort(key=lambda p: p["initial_move_pct"], reverse=True)

    print(f"DEBUG: Found {len(patterns)} patterns total")
    return patterns
# Add these function declarations at the top of your analyze.py file after imports:

def analyze_move(result):
    """Legacy function name kept for compatibility"""
    if result is None:
        return "No valid pattern found."

    # Handle multiple patterns
    if isinstance(result, list):
        analyses = []
        for i, pattern in enumerate(result):
            analysis = analyze_single_pattern(pattern)
            analyses.append(f"Pattern {i + 1} ({pattern['direction']}): {analysis}")
        return "\n".join(analyses)
    else:
        return analyze_single_pattern(result)


def draw_fibonacci_levels(ax, prices, dates, result, direction, is_failed=False):
    """
    Draw Fibonacci retracement levels based on identified pattern.

    Args:
        ax: Matplotlib axis
        prices: List of price values
        dates: List of corresponding dates
        result: Pattern dictionary or None
        direction: Trend direction ('up' or 'down')
        is_failed: Whether the pattern is failed

    Returns:
        Updated matplotlib axis
    """
    # Define points A and B
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

    # Calculate main move
    main_move = B - A if direction == "up" else A - B

    # Updated Fibonacci levels with extended levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 0.854, 1.0, 1.236, 1.618, -0.236, -0.618]

    # Different style for failed patterns
    linestyle = '--' if not is_failed else '-.'
    linewidth = 1.5 if not is_failed else 2.0
    alpha = 0.6 if not is_failed else 0.8

    # Highlight the 76.4% level more for failed patterns
    level_colors = {
        0: '#FF0000',  # Red
        0.236: '#FF7F00',  # Orange
        0.382: '#FFFF00',  # Yellow
        0.5: '#00FF00',  # Green
        0.618: '#0000FF',  # Blue
        0.764: '#FF0000' if is_failed else '#4B0082',  # Red if failed, otherwise Indigo
        0.854: '#708090',  # SlateGray
        1.0: '#8F00FF',  # Violet
        1.236: '#FFA500',  # Orange extension
        1.618: '#32CD32',  # LimeGreen extension
        -0.236: '#FFC0CB',  # Pink
        -0.618: '#800080'  # Purple
    }

    for level in fib_levels:
        fib_price = B - (main_move * level) if direction == "up" else B + (main_move * level)

        # Special formatting for 76.4% level in failed patterns
        if is_failed and level == 0.764:
            ax.axhline(
                y=fib_price,
                color='red',
                linestyle='-',
                alpha=1.0,
                linewidth=2.5,
                label=f'Failed at {level * 100:.1f}%'
            )
            # Add a "FAILED" label
            ax.text(
                dates[0],
                fib_price,
                f'{level * 100:.1f}% FAILED',
                fontsize=12,
                fontweight='bold',
                verticalalignment='center',
                color='red'
            )
        else:
            ax.axhline(
                y=fib_price,
                color=level_colors.get(level, 'gray'),
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
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

    # Add the custom 80% failure level and -23.6% completion level if available
    if result and 'failure_level' in result:
        failure_level = result['failure_level']
        ax.axhline(
            y=failure_level,
            color='red',
            linestyle='--',
            alpha=1.0,
            linewidth=2.5,
            label='80% Failure Level'
        )
        ax.text(
            dates[0],
            failure_level,
            '80% Failure',
            fontsize=12,
            fontweight='bold',
            verticalalignment='center',
            color='red'
        )

    # Add the custom -23.6% completion level
    if result and 'completion_level' in result:
        completion_level = result['completion_level']
        ax.axhline(
            y=completion_level,
            color='green',
            linestyle='--',
            alpha=1.0,
            linewidth=2.5,
            label='-23.6% Completion Level'
        )
        ax.text(
            dates[0],
            completion_level,
            '-23.6% Completion',
            fontsize=12,
            fontweight='bold',
            verticalalignment='center',
            color='green'
        )

    return ax
def analyze_single_pattern(pattern):
    """
    Analyze a single Fibonacci pattern.
    Updated to handle both old and new pattern structures.
    """
    direction = pattern["direction"]

    # Get retracement percentage, handling both old and new pattern formats
    if "retracement_pct" in pattern:
        retracement_pct = pattern["retracement_pct"]
    else:
        # Calculate it for backward compatibility
        A_price = pattern["A"][1]
        B_price = pattern["B"][1]
        C_price = pattern["C"][1]

        if direction == "up":
            initial_move = B_price - A_price
            retracement = B_price - C_price
        else:
            initial_move = A_price - B_price
            retracement = C_price - B_price

        retracement_pct = retracement / initial_move * 100 if initial_move != 0 else 0

    # Get initial move percentage, handling both formats
    if "initial_move_pct" in pattern:
        initial_move_pct = pattern["initial_move_pct"]
    else:
        # Calculate it for backward compatibility
        A_price = pattern["A"][1]
        B_price = pattern["B"][1]

        if direction == "up":
            initial_move = B_price - A_price
            initial_move_pct = initial_move / A_price * 100
        else:
            initial_move = A_price - B_price
            initial_move_pct = initial_move / A_price * 100

    # Get status, handling both formats
    status = pattern.get("status", "unknown")

    A_price = pattern["A"][1]
    B_price = pattern["B"][1]
    C_price = pattern["C"][1]
    D_price = pattern["D"][1]

    # Format status for display
    if status == "failed":
        analysis = f"Failed {direction}trend pattern - Retracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"
    elif status == "progressive":
        analysis = f"Progressive {direction}trend pattern - Retracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"
    elif status == "completed":
        analysis = f"Completed {direction}trend pattern - Retracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"
    else:  # in_progress
        analysis = f"In-progress {direction}trend pattern - Retracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"

    # Add Fibonacci levels to analysis, but only if the keys exist
    if "failure_level" in pattern:
        analysis += f"\nFailure level (80%): {pattern['failure_level']:.2f}"

    if "completion_level" in pattern:
        analysis += f"\nCompletion level (-23.6%): {pattern['completion_level']:.2f}"

    if "target_level" in pattern:
        analysis += f"\nTarget level (50%): {pattern['target_level']:.2f}"
    elif "fifty_pct_level" in pattern:
        analysis += f"\nFifty percent level: {pattern['fifty_pct_level']:.2f}"

    return analysis

def plot_diagnostic_graph(prices, dates, result=None):
    """
    Plot a diagnostic graph showing price series, moving averages, and Fibonacci levels.
    Now using datetime values for the x-axis.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    short_window = 20
    long_window = 50

    if len(prices) >= long_window:
        prices_array = np.array(prices)
        short_ma = np.convolve(prices_array, np.ones(short_window) / short_window, mode='valid')
        long_ma = np.convolve(prices_array, np.ones(long_window) / long_window, mode='valid')

    trends = detect_multiple_timeframe_trends(prices)
    trend_str = ", ".join([f"{k}: {v}" for k, v in trends.items()])

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
    ax.set_title(f'Price Series Diagnostic Plot - {move_type} Move\n(Timeframe Trends: {trend_str})', fontsize=14)

    if result:
        # Check if we have multiple patterns
        if isinstance(result, list):
            # Draw all patterns with different colors
            colors = [('black', 'red', 'green', 'blue'), ('purple', 'orange', 'cyan', 'magenta')]

            for i, pattern in enumerate(result):
                # Check if this is a failed pattern
                is_failed = pattern.get('status') == 'failed'
                marker_size = 150 if is_failed else 100  # Larger marker for failed patterns
                marker_style = 'x' if is_failed else 'o'  # X for failed patterns

                color_set = colors[i % len(colors)]
                points = {'A': (color_set[0], pattern['A']), 'B': (color_set[1], pattern['B']),
                          'C': (color_set[2], pattern['C']), 'D': (color_set[3], pattern['D'])}

                for label, (color, point) in points.items():
                    idx, price = point
                    ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

                    # Add different text for failed patterns
                    if is_failed and label == 'C':
                        ax.text(dates[idx], price, f"{label}{i + 1} (FAILED)", fontsize=14, fontweight='bold',
                                ha='right', va='bottom', color='red')
                    else:
                        ax.text(dates[idx], price, f"{label}{i + 1}", fontsize=14, fontweight='bold',
                                ha='right', va='bottom', color=color)

                # Draw Fibonacci levels using the detected points
                ax = draw_fibonacci_levels(ax, prices, dates, pattern, pattern['direction'], is_failed=is_failed)
        else:
            # Single pattern
            # Check if this is a failed pattern
            is_failed = result.get('status') == 'failed'
            marker_size = 150 if is_failed else 100
            marker_style = 'x' if is_failed else 'o'

            points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                      'C': ('green', result['C']), 'D': ('blue', result['D'])}

            for label, (color, point) in points.items():
                idx, price = point
                ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

                # Add different text for failed patterns
                if is_failed and label == 'C':
                    ax.text(dates[idx], price, f"{label} (FAILED)", fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color='red')
                else:
                    ax.text(dates[idx], price, label, fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color=color)

            # Draw Fibonacci levels using the detected points
            ax = draw_fibonacci_levels(ax, prices, dates, result, direction, is_failed=is_failed)
    else:
        # Draw Fibonacci levels based on min/max if no pattern detected
        ax = draw_fibonacci_levels(ax, prices, dates, None, direction)

    # Add min/max points
    min_idx = np.argmin(prices)
    max_idx = np.argmax(prices)

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels for better readability

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


def analyze_price_data(prices, dates, min_change_threshold=0.005, max_days_between=None,
                       allow_multiple_patterns=True, long_term_window=100, detect_long_failures=True,
                       pattern_config=None):
    """
    Analyze price data for Fibonacci patterns.
    Updated to use configurable pattern parameters.

    Args:
        prices: List of price values
        dates: List of corresponding dates
        min_change_threshold: Minimum price change threshold
        max_days_between: Maximum days between points
        allow_multiple_patterns: Whether to detect multiple patterns
        long_term_window: Window size for detecting longer-term patterns
        detect_long_failures: Whether to specifically look for long-term failed patterns
        pattern_config: Configuration dictionary for pattern detection

    Returns:
        tuple: (result, analysis, trends) - detected patterns, analysis text, and trend info
    """
    # Detect trends at multiple timeframes (kept for compatibility)
    trends = detect_multiple_timeframe_trends(prices)
    print(f"Trends across multiple timeframes: {trends}")

    # Find Fibonacci patterns using configurable parameters
    result = find_significant_price_patterns(
        prices,
        dates,
        min_change_threshold,
        config=pattern_config
    )

    # Analyze the detected patterns
    analysis = analyze_move(result) if result else "No valid pattern found."

    return result, analysis, trends
def display_results(prices, dates, result, analysis, trends):
    """
    Display the results of the Fibonacci pattern analysis.
    Fixed to properly map pattern points to the correct positions.
    """
    # Create and show the diagnostic plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot the price series with small markers to see actual movements
    ax.plot(dates, prices, marker='o', linestyle='-', color='black', alpha=0.7,
            label='Price Series', markersize=3)

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels

    # Determine direction and title
    direction = "up"
    if result:
        if isinstance(result, list):
            direction = result[0]['direction']
        else:
            direction = result['direction']
    else:
        direction = trends.get('overall', 'unknown')

    move_type = "Upward" if direction == "up" else "Downward"
    trend_str = ", ".join([f"{k}: {v}" for k, v in trends.items()])
    ax.set_title(f'Price Series Diagnostic Plot - {move_type} Timeframe Trends: {trend_str})', fontsize=14)

    # Print analysis results
    print("\nAnalysis Results:")
    print("-" * 50)
    print(f"Overall trend direction: {trends.get('overall', 'unknown')}")
    print(f"Analysis: {analysis}")

    # Draw Fibonacci levels and pattern points
    if result:
        if isinstance(result, list):
            # Handle multiple patterns
            print(f"Detected {len(result)} different patterns")
            colors = [('black', 'red', 'green', 'blue'), ('purple', 'orange', 'cyan', 'magenta')]

            for i, pattern in enumerate(result):
                print(f"\nPattern {i + 1}:")
                print(f"Detected trend: {pattern['direction']}")
                print(f"Initial move: {pattern['initial_move_pct']:.2f}%")
                print(f"Retracement: {pattern['retracement_pct']:.2f}%")
                print(f"Status: {pattern['status']}")
                print(f"Identified points:")

                # Check if this is a failed pattern
                is_failed = pattern.get('status') == 'failed'
                marker_size = 150 if is_failed else 100  # Larger marker for failed patterns
                marker_style = 'x' if is_failed else 'o'  # X for failed patterns

                # Use different colors and markers for failed patterns
                color_set = colors[i % len(colors)]
                points = {'A': (color_set[0], pattern['A']), 'B': (color_set[1], pattern['B']),
                          'C': (color_set[2], pattern['C']), 'D': (color_set[3], pattern['D'])}

                # DEBUG: Output raw point data to verify values
                print(f"DEBUG: Raw pattern points:")
                for label, (_, point_data) in points.items():
                    idx, price = point_data
                    print(f"DEBUG: {label} - Index: {idx}, Price: {price:.2f}")

                for label, (color, point) in points.items():
                    idx, price = point

                    # IMPORTANT FIX: Check if the index is within this window's date range
                    if idx >= len(dates):
                        print(f"Warning: Point {label} of pattern {i + 1} has index {idx} which is outside the current window (size {len(dates)}).")
                        # Use the last available date instead
                        idx = len(dates) - 1

                    # Now idx is guaranteed to be within range
                    ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

                    # Add different text for failed patterns
                    if is_failed and label == 'C':
                        ax.text(dates[idx], price, f"{label}{i + 1} (FAILED)", fontsize=14, fontweight='bold',
                                ha='right', va='bottom', color='red')
                    else:
                        ax.text(dates[idx], price, f"{label}{i + 1}", fontsize=14, fontweight='bold',
                                ha='right', va='bottom', color=color)

                    # DEBUG: Print where this point is being placed
                    print(f"{label}: Index {idx}, Date {dates[idx].strftime('%Y-%m-%d %H:%M')}, Price {price}")

                # Draw Fibonacci levels with special styling for failed patterns
                draw_fibonacci_levels(ax, prices, dates, pattern, pattern['direction'], is_failed=is_failed)

                # Print Fibonacci levels
                A, B = pattern['A'][1], pattern['B'][1]
                main_move = B - A if pattern['direction'] == "up" else A - B
                print(f"\nKey Fibonacci levels for Pattern {i + 1}:")
                for level in [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, 1.236, 1.618, -0.236, -0.618]:
                    fib_price = B - (main_move * level) if pattern['direction'] == "up" else B + (main_move * level)
                    print(f"{level * 100:.1f}%: {fib_price:.2f}")

                # Print custom levels
                if 'failure_level' in pattern:
                    print(f"80% Failure level: {pattern['failure_level']:.2f}")
                if 'completion_level' in pattern:
                    print(f"-23.6% Completion level: {pattern['completion_level']:.2f}")
        else:
            # Single pattern
            print(f"Detected trend: {result['direction']}")
            print(f"Initial move: {result['initial_move_pct']:.2f}%")
            print(f"Retracement: {result['retracement_pct']:.2f}%")
            print(f"Status: {result['status']}")
            print(f"Identified points:")

            # Check if this is a failed pattern
            is_failed = result.get('status') == 'failed'
            marker_size = 150 if is_failed else 100
            marker_style = 'x' if is_failed else 'o'

            # Draw pattern points
            points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                     'C': ('green', result['C']), 'D': ('blue', result['D'])}

            # DEBUG: Output raw point data to verify values
            print(f"DEBUG: Raw pattern points:")
            for label, (_, point_data) in points.items():
                idx, price = point_data
                print(f"DEBUG: {label} - Index: {idx}, Price: {price:.2f}")

            for label, (color, point) in points.items():
                idx, price = point

                # IMPORTANT FIX: Check if the index is within this window's date range
                if idx >= len(dates):
                    print(f"Warning: Point {label} has index {idx} which is outside the current window (size {len(dates)}).")
                    # Use the last available date instead
                    idx = len(dates) - 1

                # Now idx is guaranteed to be within range
                ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

                # Add different text for failed patterns
                if is_failed and label == 'C':
                    ax.text(dates[idx], price, f"{label} (FAILED)", fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color='red')
                else:
                    ax.text(dates[idx], price, label, fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color=color)

                print(f"{label}: Index {idx}, Date {dates[idx].strftime('%Y-%m-%d %H:%M')}, Price {price}")

            # Draw Fibonacci levels
            draw_fibonacci_levels(ax, prices, dates, result, result['direction'], is_failed=is_failed)

            # Print Fibonacci levels
            A, B = result['A'][1], result['B'][1]
            main_move = B - A if result['direction'] == "up" else A - B
            print("\nKey Fibonacci levels:")
            for level in [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, 1.236, 1.618, -0.236, -0.618]:
                fib_price = B - (main_move * level) if result['direction'] == "up" else B + (main_move * level)
                print(f"{level * 100:.1f}%: {fib_price:.2f}")

            # Print custom levels
            if 'failure_level' in result:
                print(f"80% Failure level: {result['failure_level']:.2f}")
            if 'completion_level' in result:
                print(f"-23.6% Completion level: {result['completion_level']:.2f}")
    else:
        # No pattern detected
        print("Could not find a valid Fibonacci retracement pattern.")
        print("A diagnostic graph has been generated with Fibonacci levels based on min/max prices.")
        draw_fibonacci_levels(ax, prices, dates, None, direction)

    # Finalize plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    # Return the figure and axis for further customization if needed
    return fig, ax


# Fix 2: Update the create_longterm_pattern_figure function in analyze.py
# This function handles the visualization of long-term patterns across the entire dataset

def create_longterm_pattern_figure(prices, dates, all_patterns):
    """
    Create a dedicated figure for significant patterns across the entire dataset.
    Fixed to properly map pattern points to the correct positions.
    """
    # Extract patterns with significant moves (over 2%)
    significant_patterns = []

    for pattern_idx, (pattern, analysis, window_info) in enumerate(all_patterns):
        # Check if the initial move percentage is at least 2%
        if isinstance(pattern, dict) and pattern.get('initial_move_pct', 0) >= 2.0:
            significant_patterns.append((pattern_idx, pattern, analysis, window_info))
        # Legacy support
        elif pattern.get('long_term', False):
            significant_patterns.append((pattern_idx, pattern, analysis, window_info))

    # If no significant patterns found, return None
    if not significant_patterns:
        print("No significant patterns detected in the dataset")
        return None

    # Create figure for significant patterns
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the full price series
    ax.plot(dates, prices, color='black', alpha=0.5, linewidth=1, label='Price')

    # Colors for different patterns
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal', 'navy', 'olive', 'maroon']

    # DEBUG: Print price range to verify y-axis scale
    print(f"DEBUG: Full price range - Min: {min(prices):.2f}, Max: {max(prices):.2f}")

    # Plot each significant pattern
    for i, (pattern_idx, pattern, analysis, window_info) in enumerate(significant_patterns):
        color = colors[i % len(colors)]

        # Extract pattern points
        A_idx, A_price = pattern['A']
        B_idx, B_price = pattern['B']
        C_idx, C_price = pattern['C']
        D_idx, D_price = pattern['D']

        # DEBUG: Print raw pattern points
        print(f"DEBUG: Pattern {pattern_idx+1} raw points:")
        print(f"DEBUG: A - Index: {A_idx}, Price: {A_price:.2f}")
        print(f"DEBUG: B - Index: {B_idx}, Price: {B_price:.2f}")
        print(f"DEBUG: C - Index: {C_idx}, Price: {C_price:.2f}")
        print(f"DEBUG: D - Index: {D_idx}, Price: {D_price:.2f}")

        # Convert to global indices (since these are within a window)
        start_idx = window_info['start_idx']
        global_A_idx = start_idx + A_idx
        global_B_idx = start_idx + B_idx
        global_C_idx = start_idx + C_idx
        global_D_idx = start_idx + D_idx

        # DEBUG: Print global indices
        print(f"DEBUG: Global A index: {global_A_idx}, Global B index: {global_B_idx}")
        print(f"DEBUG: Global C index: {global_C_idx}, Global D index: {global_D_idx}")

        # Ensure indices are within range
        if global_A_idx >= len(dates) or global_B_idx >= len(dates) or global_C_idx >= len(
                dates) or global_D_idx >= len(dates):
            print(f"Warning: Pattern {pattern_idx + 1} has indices out of range. Skipping.")
            continue

        # Check if pattern is failed
        is_failed = pattern.get('status') == 'failed'
        marker_style = 'x' if is_failed else 'o'

        # Plot points - IMPORTANT FIX: Use the actual price values from the pattern,
        # not the prices at the indices in the global price array
        ax.scatter(dates[global_A_idx], A_price, color=color, marker='o', s=120, zorder=5,
                   label=f'Pattern {pattern_idx + 1} (A)')
        ax.scatter(dates[global_B_idx], B_price, color=color, marker='s', s=120, zorder=5)
        ax.scatter(dates[global_C_idx], C_price, color=color, marker=marker_style, s=150, zorder=5)
        ax.scatter(dates[global_D_idx], D_price, color=color, marker='d', s=120, zorder=5)

        # DEBUG: Print where points are being placed
        print(f"DEBUG: Placing A at Date: {dates[global_A_idx].strftime('%Y-%m-%d %H:%M')}, Price: {A_price:.2f}")
        print(f"DEBUG: Placing B at Date: {dates[global_B_idx].strftime('%Y-%m-%d %H:%M')}, Price: {B_price:.2f}")
        print(f"DEBUG: Placing C at Date: {dates[global_C_idx].strftime('%Y-%m-%d %H:%M')}, Price: {C_price:.2f}")
        print(f"DEBUG: Placing D at Date: {dates[global_D_idx].strftime('%Y-%m-%d %H:%M')}, Price: {D_price:.2f}")

        # Add labels
        ax.text(dates[global_A_idx], A_price, f'A{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')
        ax.text(dates[global_B_idx], B_price, f'B{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')
        ax.text(dates[global_C_idx], C_price, f'C{pattern_idx + 1}{"(F)" if is_failed else ""}', fontsize=12,
                fontweight='bold', ha='right', va='bottom')
        ax.text(dates[global_D_idx], D_price, f'D{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')

        # Connect the points with lines
        if is_failed:
            # For failed patterns, use dashed lines
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx]],
                    [A_price, B_price, C_price],
                    linestyle='--', color=color, linewidth=2, alpha=0.7)
        else:
            # For normal patterns, use solid lines
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx], dates[global_D_idx]],
                    [A_price, B_price, C_price, D_price],
                    linestyle='-', color=color, linewidth=2, alpha=0.7)

        # Add a box with pattern information
        direction = pattern['direction']

        # Get move and retracement info - adapt to both new and legacy formats
        if 'initial_move_pct' in pattern:
            move_pct = pattern['initial_move_pct']
            retrace_pct = pattern['retracement_pct']
        else:
            # Legacy calculation
            main_move = B_price - A_price if direction == "up" else A_price - B_price
            move_pct = abs(main_move / A_price) * 100

            if direction == "up":
                retrace_pct = (B_price - C_price) / main_move * 100 if main_move != 0 else 0
            else:
                retrace_pct = (C_price - B_price) / main_move * 100 if main_move != 0 else 0

        text_y_pos = min(prices) + (i * (max(prices) - min(prices)) * 0.05)
        text_x_pos = dates[int(len(dates) * 0.05)]  # 5% from the left edge

        # Create info box text
        info_text = f"Pattern {pattern_idx + 1}: {direction.upper()}-trend "
        info_text += f"{pattern.get('status', 'unknown').upper()} "
        info_text += f"(Move: {move_pct:.1f}%, Retrace: {retrace_pct:.1f}%)"

        # Add info box
        ax.text(text_x_pos, text_y_pos, info_text,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round'),
                color=color, fontsize=10, fontweight='bold')

    # Set title and labels
    num_patterns = len(significant_patterns)
    ax.set_title(f'Significant Patterns Overview ({num_patterns} patterns detected)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Adjust layout
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    return fig


# Fix 3: Update the code in main.py for showing pattern points in the summary figure
# This is the part in main() function that creates the overview plot

# Replace the pattern point plotting code in the main.py with this:

'''
# Add the pattern points to the overview
if isinstance(result, list):
    # Multiple patterns in this window
    for j, pattern in enumerate(result):
        # Plot each point (A, B, C, D) for this pattern
        for label, marker_shape in markers.items():
            # Get point coordinates (correcting for window offset)
            idx, price = pattern[label]
            global_idx = start_idx + idx  # Convert window index to global index

            # Ensure the index is valid
            if global_idx < len(dates):
                # FIXED: Use the original price value from pattern, not the price from global array
                ax.scatter(dates[global_idx], price, color=color, marker=marker_shape,
                           s=120, zorder=4, edgecolors='black')

                # Add a label with pattern number and point label
                ax.text(dates[global_idx], price, f"{label}{i + 1}", fontsize=12,
                        color=color, fontweight='bold', ha='right', va='bottom')
else:
    # Single pattern in this window
    # Plot each point (A, B, C, D)
    for label, marker_shape in markers.items():
        # Get point coordinates (correcting for window offset)
        idx, price = result[label]
        global_idx = start_idx + idx  # Convert window index to global index

        # Ensure the index is valid
        if global_idx < len(dates):
            # FIXED: Use the original price value from pattern, not the price from global array
            ax.scatter(dates[global_idx], price, color=color, marker=marker_shape,
                       s=120, zorder=4, edgecolors='black')

            # Add a label
            ax.text(dates[global_idx], price, f"{label}{i + 1}", fontsize=12,
                    color=color, fontweight='bold', ha='right', va='bottom')
'''

# Fix 4: Add debug output to analyze_multiple_windows function
# This will help identify any issues with the pattern detection itself

def debug_analyze_multiple_windows(prices, dates, window_sizes=[50, 100, 200],
                             overlap_percent=50, min_change_threshold=0.001,
                             allow_multiple_patterns=True, detect_long_failures=True):
    """
    Debug version of analyze_multiple_windows function with additional logging.
    """
    all_patterns = []

    # Ensure window sizes are sorted from smallest to largest
    window_sizes = sorted(window_sizes)

    for window_size in window_sizes:
        if window_size >= len(prices):
            print(f"Window size {window_size} is larger than available data ({len(prices)} points). Skipping.")
            continue

        # Calculate step size based on overlap percentage
        step_size = max(1, int(window_size * (1 - overlap_percent / 100)))

        # Iterate through the data with the current window size
        for start_idx in range(0, len(prices) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Get the window data
            window_prices = prices[start_idx:end_idx]
            window_dates = dates[start_idx:end_idx]

            # DEBUG: Print window price range
            print(f"DEBUG: Window {start_idx}:{end_idx} - Price range: Min={min(window_prices):.2f}, Max={max(window_prices):.2f}")

            # Detect patterns in this window
            window_patterns = find_significant_price_patterns(
                window_prices,
                window_dates,
                min_change_threshold
            )

            # If patterns were found, analyze them and add to results
            if window_patterns:
                # Add the window info
                window_info = {
                    "window_size": window_size,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": dates[start_idx],
                    "end_date": dates[end_idx - 1],
                    "window_prices": window_prices,
                    "window_dates": window_dates
                }

                # DEBUG: Print detected pattern details
                for p_idx, p in enumerate(window_patterns):
                    print(f"DEBUG: Window {start_idx}:{end_idx} - Pattern {p_idx+1} details:")
                    print(f"  Direction: {p['direction']}")
                    print(f"  Status: {p.get('status', 'unknown')}")
                    print(f"  A: Index={p['A'][0]}, Price={p['A'][1]:.2f}")
                    print(f"  B: Index={p['B'][0]}, Price={p['B'][1]:.2f}")
                    print(f"  C: Index={p['C'][0]}, Price={p['C'][1]:.2f}")
                    print(f"  D: Index={p['D'][0]}, Price={p['D'][1]:.2f}")

                # For compatibility with the existing code, we'll take only the most significant pattern
                # from each window unless multiple patterns are explicitly requested
                if allow_multiple_patterns:
                    for pattern in window_patterns:
                        pattern_analysis = analyze_single_pattern(pattern)
                        all_patterns.append((pattern, pattern_analysis, window_info))
                else:
                    top_pattern = window_patterns[0]  # Most significant pattern (largest price move)
                    pattern_analysis = analyze_single_pattern(top_pattern)
                    all_patterns.append((top_pattern, pattern_analysis, window_info))

    # Sort patterns by significance
    all_patterns.sort(key=lambda x: pattern_significance(x[0]), reverse=True)

    return all_patterns


def analyze_multiple_windows(prices, dates, window_sizes=[50, 100, 200],
                             overlap_percent=50, min_change_threshold=0.001,
                             allow_multiple_patterns=True, detect_long_failures=True):
    """
    Analyze the same price series with different window sizes to capture both short and long-term patterns.
    Updated to use the new pattern detection approach.
    """
    all_patterns = []

    # Ensure window sizes are sorted from smallest to largest
    window_sizes = sorted(window_sizes)

    for window_size in window_sizes:
        if window_size >= len(prices):
            print(f"Window size {window_size} is larger than available data ({len(prices)} points). Skipping.")
            continue

        # Calculate step size based on overlap percentage
        step_size = max(1, int(window_size * (1 - overlap_percent / 100)))

        # Iterate through the data with the current window size
        for start_idx in range(0, len(prices) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Get the window data
            window_prices = prices[start_idx:end_idx]
            window_dates = dates[start_idx:end_idx]

            # Detect patterns in this window using the new approach
            window_patterns = find_significant_price_patterns(
                window_prices,
                window_dates,
                min_change_threshold
            )

            # If patterns were found, analyze them and add to results
            if window_patterns:
                # Add the window info
                window_info = {
                    "window_size": window_size,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": dates[start_idx],
                    "end_date": dates[end_idx - 1],
                    "window_prices": window_prices,
                    "window_dates": window_dates
                }

                # For compatibility with the existing code, we'll take only the most significant pattern
                # from each window unless multiple patterns are explicitly requested
                if allow_multiple_patterns:
                    for pattern in window_patterns:
                        pattern_analysis = analyze_single_pattern(pattern)
                        all_patterns.append((pattern, pattern_analysis, window_info))
                else:
                    top_pattern = window_patterns[0]  # Most significant pattern (largest price move)
                    pattern_analysis = analyze_single_pattern(top_pattern)
                    all_patterns.append((top_pattern, pattern_analysis, window_info))

    # Sort patterns by significance
    all_patterns.sort(key=lambda x: pattern_significance(x[0]), reverse=True)

    return all_patterns


def adjust_pattern_indices(pattern, offset):
    """
    Adjust pattern indices to correspond to the full dataset.
    This prevents the 'list index out of range' error.
    """
    adjusted_pattern = pattern.copy()

    # Adjust the point indices
    for point in ['A', 'B', 'C', 'D']:
        idx, price = pattern[point]
        adjusted_pattern[point] = (idx + offset, price)

    return adjusted_pattern


def pattern_significance(pattern):
    """
    Calculate a significance score for a pattern.
    Higher score = more significant pattern.
    Updated to use the new pattern structure.
    """
    # For the new simplified pattern detection, we prioritize:
    # 1. Larger initial moves
    # 2. Patterns that are completed or progressive
    # 3. Patterns with retracement close to 50%

    # Start with the initial move significance
    if 'initial_move_pct' in pattern:
        score = min(pattern['initial_move_pct'] / 5, 5)  # Cap at 5 for moves >= 25%

        # Add bonus for completed or progressive patterns
        if pattern.get('status') == 'completed':
            score += 3
        elif pattern.get('status') == 'progressive':
            score += 2
        elif pattern.get('status') == 'failed':
            score -= 1

        # Add bonus for retracement close to 50%
        retrace_pct = pattern.get('retracement_pct', 0)
        retrace_quality = 1.0 - abs(retrace_pct - 50) / 50  # 1.0 for perfect 50%, 0 for 0% or 100%
        score += retrace_quality * 2

        return score

    # Legacy method for backward compatibility
    # Handle multiple patterns
    if isinstance(pattern, list):
        return max(pattern_significance(p) for p in pattern)

    A, B, C, D = pattern['A'][1], pattern['B'][1], pattern['C'][1], pattern['D'][1]
    direction = pattern['direction']

    # Calculate move sizes
    if direction == "up":
        initial_move = B - A
        retracement = B - C
        extension = D - C
    else:
        initial_move = A - B
        retracement = C - B
        extension = C - D

    # Calculate retracement ratio
    retracement_ratio = retracement / initial_move if initial_move != 0 else 0

    # Calculate extension ratio
    extension_ratio = extension / initial_move if initial_move != 0 else 0

    # Ideal retracement is around 50-61.8%
    retracement_quality = 1.0 - abs(retracement_ratio - 0.618)

    # Ideal extension is around 100-161.8%
    extension_quality = 1.0 - min(abs(extension_ratio - 1.0), abs(extension_ratio - 1.618))

    # Size of the overall move matters
    move_size = abs(D - A)

    # Check if pattern is failed
    is_failed = pattern.get('status') == 'failed'

    # Failed patterns might be very interesting
    failed_bonus = 0.5 if is_failed else 0

    # Long-term patterns often more important
    long_term_bonus = 0.3 if pattern.get('long_term', False) else 0

    # Combine factors into a single score
    score = (
            move_size * 0.4 +
            retracement_quality * 0.2 +
            extension_quality * 0.2 +
            failed_bonus +
            long_term_bonus
    )

    return score


def create_longterm_pattern_figure(prices, dates, all_patterns):
    """
    Create a dedicated figure for significant patterns across the entire dataset.

    Modified to work with the new pattern detection approach.

    Args:
        prices: Full list of price values
        dates: Full list of dates
        all_patterns: List of (result, analysis, window_info) from the multi-window analysis

    Returns:
        Figure with long-term patterns highlighted
    """
    # Extract patterns with significant moves (over 2%)
    significant_patterns = []

    for pattern_idx, (pattern, analysis, window_info) in enumerate(all_patterns):
        # Check if the initial move percentage is at least 2%
        if isinstance(pattern, dict) and pattern.get('initial_move_pct', 0) >= 2.0:
            significant_patterns.append((pattern_idx, pattern, analysis, window_info))
        # Legacy support
        elif pattern.get('long_term', False):
            significant_patterns.append((pattern_idx, pattern, analysis, window_info))

    # If no significant patterns found, return None
    if not significant_patterns:
        print("No significant patterns detected in the dataset")
        return None

    # Create figure for significant patterns
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the full price series
    ax.plot(dates, prices, color='black', alpha=0.5, linewidth=1, label='Price')

    # Colors for different patterns
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal', 'navy', 'olive', 'maroon']

    # Plot each significant pattern
    for i, (pattern_idx, pattern, analysis, window_info) in enumerate(significant_patterns):
        color = colors[i % len(colors)]

        # Extract pattern points
        A_idx, A_price = pattern['A']
        B_idx, B_price = pattern['B']
        C_idx, C_price = pattern['C']
        D_idx, D_price = pattern['D']

        # Convert to global indices (since these are within a window)
        start_idx = window_info['start_idx']
        global_A_idx = start_idx + A_idx
        global_B_idx = start_idx + B_idx
        global_C_idx = start_idx + C_idx
        global_D_idx = start_idx + D_idx

        # Ensure indices are within range
        if global_A_idx >= len(dates) or global_B_idx >= len(dates) or global_C_idx >= len(
                dates) or global_D_idx >= len(dates):
            print(f"Warning: Pattern {pattern_idx + 1} has indices out of range. Skipping.")
            continue

        # Check if pattern is failed
        is_failed = pattern.get('status') == 'failed'
        marker_style = 'x' if is_failed else 'o'

        # Plot points
        ax.scatter(dates[global_A_idx], A_price, color=color, marker='o', s=120, zorder=5,
                   label=f'Pattern {pattern_idx + 1} (A)')
        ax.scatter(dates[global_B_idx], B_price, color=color, marker='s', s=120, zorder=5)
        ax.scatter(dates[global_C_idx], C_price, color=color, marker=marker_style, s=150, zorder=5)
        ax.scatter(dates[global_D_idx], D_price, color=color, marker='d', s=120, zorder=5)

        # Add labels
        ax.text(dates[global_A_idx], A_price, f'A{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')
        ax.text(dates[global_B_idx], B_price, f'B{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')
        ax.text(dates[global_C_idx], C_price, f'C{pattern_idx + 1}{"(F)" if is_failed else ""}', fontsize=12,
                fontweight='bold', ha='right', va='bottom')
        ax.text(dates[global_D_idx], D_price, f'D{pattern_idx + 1}', fontsize=12, fontweight='bold', ha='right',
                va='bottom')

        # Connect the points with lines
        if is_failed:
            # For failed patterns, use dashed lines
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx]],
                    [A_price, B_price, C_price],
                    linestyle='--', color=color, linewidth=2, alpha=0.7)
        else:
            # For normal patterns, use solid lines
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx], dates[global_D_idx]],
                    [A_price, B_price, C_price, D_price],
                    linestyle='-', color=color, linewidth=2, alpha=0.7)

        # Add a box with pattern information
        direction = pattern['direction']

        # Get move and retracement info - adapt to both new and legacy formats
        if 'initial_move_pct' in pattern:
            move_pct = pattern['initial_move_pct']
            retrace_pct = pattern['retracement_pct']
        else:
            # Legacy calculation
            main_move = B_price - A_price if direction == "up" else A_price - B_price
            move_pct = abs(main_move / A_price) * 100

            if direction == "up":
                retrace_pct = (B_price - C_price) / main_move * 100 if main_move != 0 else 0
            else:
                retrace_pct = (C_price - B_price) / main_move * 100 if main_move != 0 else 0

        text_y_pos = min(prices) + (i * (max(prices) - min(prices)) * 0.05)
        text_x_pos = dates[int(len(dates) * 0.05)]  # 5% from the left edge

        # Create info box text
        info_text = f"Pattern {pattern_idx + 1}: {direction.upper()}-trend "
        info_text += f"{pattern.get('status', 'unknown').upper()} "
        info_text += f"(Move: {move_pct:.1f}%, Retrace: {retrace_pct:.1f}%)"

        # Add info box
        ax.text(text_x_pos, text_y_pos, info_text,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round'),
                color=color, fontsize=10, fontweight='bold')

    # Set title and labels
    num_patterns = len(significant_patterns)
    ax.set_title(f'Significant Patterns Overview ({num_patterns} patterns detected)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Adjust layout
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    return fig


def create_detailed_pattern_graph(pattern_number, result, analysis, window_info, trends):
    """Create a detailed graph for a specific pattern with more information."""
    window_prices = window_info['window_prices']
    window_dates = window_info['window_dates']

    # Create a figure with 2 subplots: main chart and information panel
    fig = plt.figure(figsize=(18, 14))  # Larger figure

    # Main chart - use 80% of the figure height
    chart_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Info panel - use 20% of the figure height
    info_ax = plt.subplot2grid((5, 1), (4, 0))
    info_ax.axis('off')  # Hide axes for info panel

    # --- Main Chart ---
    # Plot the price series
    chart_ax.plot(window_dates, window_prices, marker='o', linestyle='-', color='black', alpha=0.7,
                  label='Price Series', markersize=3)

    # Get basic pattern info, supporting both new and legacy formats
    if isinstance(result, list):
        # Handle multiple patterns (use the first one for simplicity)
        pattern = result[0]
    else:
        pattern = result

    direction = pattern['direction']
    status = pattern.get('status', 'unknown')

    # Determine if the pattern is failed
    is_failed = status == 'failed'

    # Get move and retracement percentages - adapt to both formats
    if 'initial_move_pct' in pattern:
        move_pct = pattern['initial_move_pct']
        retrace_pct = pattern['retracement_pct']
    else:
        # Legacy calculation
        A_price = pattern['A'][1]
        B_price = pattern['B'][1]
        C_price = pattern['C'][1]
        main_move = B_price - A_price if direction == "up" else A_price - B_price
        move_pct = abs(main_move / A_price) * 100

        if direction == "up":
            retrace_pct = (B_price - C_price) / main_move * 100 if main_move != 0 else 0
        else:
            retrace_pct = (C_price - B_price) / main_move * 100 if main_move != 0 else 0

    # Plot each point (A, B, C, D)
    colors = {'A': 'black', 'B': 'red', 'C': ('darkred' if is_failed else 'green'), 'D': 'blue'}
    markers = {'A': 'o', 'B': 's', 'C': ('X' if is_failed else '^'), 'D': 'd'}
    sizes = {'A': 120, 'B': 120, 'C': 160 if is_failed else 120, 'D': 120}

    for label in ['A', 'B', 'C', 'D']:
        idx, price = pattern[label]

        # Ensure index is within range
        if idx < len(window_dates):
            chart_ax.scatter(window_dates[idx], price, color=colors[label],
                             marker=markers[label], s=sizes[label], zorder=4, edgecolors='black')

            # Add label with extra emphasis on failed points
            if is_failed and label == 'C':
                chart_ax.text(window_dates[idx], price, f"{label} (FAILED)", fontsize=14,
                              fontweight='bold', color='darkred', ha='right', va='bottom')
            else:
                chart_ax.text(window_dates[idx], price, label, fontsize=14,
                              fontweight='bold', color=colors[label], ha='right', va='bottom')

    # Draw Fibonacci levels
    draw_fibonacci_levels(chart_ax, window_prices, window_dates, pattern, direction, is_failed=is_failed)

    # Connect the points with lines to show the pattern more clearly
    if is_failed:
        # For failed patterns, use red dashed lines
        x_points = [window_dates[pattern['A'][0]], window_dates[pattern['B'][0]],
                    window_dates[pattern['C'][0]]]
        y_points = [pattern['A'][1], pattern['B'][1], pattern['C'][1]]
        chart_ax.plot(x_points, y_points, 'r--', linewidth=2, alpha=0.7)
    else:
        # For regular patterns, use green solid lines
        x_points = [window_dates[pattern['A'][0]], window_dates[pattern['B'][0]],
                    window_dates[pattern['C'][0]], window_dates[pattern['D'][0]]]
        y_points = [pattern['A'][1], pattern['B'][1], pattern['C'][1], pattern['D'][1]]
        chart_ax.plot(x_points, y_points, 'g-', linewidth=2, alpha=0.7)

    # Set title and axis labels
    title_text = f"Pattern {pattern_number} - {status.upper()} - "
    title_text += f"{direction.upper()}TREND"
    chart_ax.set_title(title_text, fontsize=16, fontweight='bold', color='darkred' if is_failed else 'darkgreen')

    chart_ax.set_xlabel('Date', fontsize=12)
    chart_ax.set_ylabel('Price', fontsize=12)
    chart_ax.grid(True, alpha=0.3)
    chart_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    chart_ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # --- Information Panel ---
    # Create a detailed info text
    info_text = f"PATTERN #{pattern_number} ANALYSIS\n\n"
    info_text += f"Type: {direction.upper()}TREND\n"
    info_text += f"Status: {status.upper()}\n"
    info_text += f"Time Period: {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}\n\n"

    # Add point information
    info_text += "KEY POINTS:\n"
    for label in ['A', 'B', 'C', 'D']:
        idx, price = pattern[label]
        if idx < len(window_dates):
            point_date = window_dates[idx].strftime('%Y-%m-%d %H:%M')
            info_text += f"• Point {label}: Price ${price:.2f} on {point_date}\n"

    # Add price movement information
    info_text += f"\nINITIAL MOVE: {move_pct:.2f}%\n"
    info_text += f"RETRACEMENT: {retrace_pct:.2f}%\n"

    # Add Fibonacci levels
    A_price, B_price = pattern['A'][1], pattern['B'][1]
    main_move = B_price - A_price if direction == "up" else A_price - B_price

    info_text += "\nKEY FIBONACCI LEVELS:\n"
    for level in [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, 1.618]:
        fib_price = B_price - (main_move * level) if direction == "up" else B_price + (main_move * level)
        info_text += f"• {level * 100:.1f}%: ${fib_price:.2f}\n"

    # Add custom levels if they exist
    if 'failure_level' in pattern:
        info_text += f"• 80% Failure level: ${pattern['failure_level']:.2f}\n"
    if 'completion_level' in pattern:
        info_text += f"• -23.6% Completion level: ${pattern['completion_level']:.2f}\n"
    if 'target_level' in pattern:
        info_text += f"• 50% Target level: ${pattern['target_level']:.2f}\n"

    # Add pattern analysis summary
    info_text += f"\nANALYSIS SUMMARY:\n{analysis}\n"

    # Add trends
    trend_str = ", ".join([f"{k}: {v}" for k, v in trends.items()])
    info_text += f"\nTRENDS: {trend_str}"

    # Display the info text
    info_ax.text(0.01, 0.99, info_text, fontsize=11, va='top', family='monospace')

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2)

    return fig

def load_and_prepare_data(csv_file, date_range=None, index_range=None):
    """
    Load and prepare price data from CSV file.

    Args:
        csv_file (str): Path to the CSV file
        date_range (tuple): Optional (start_date, end_date) for filtering
        index_range (tuple): Optional (start_idx, end_idx) for filtering

    Returns:
        tuple: (prices, dates, df) - price data, dates, and full dataframe
    """
    # Read the CSV file with datetime parsing
    df = pd.read_csv(csv_file)

    # Check first few rows to understand the structure
    print("First few rows of the CSV file:")
    print(df.head(2))

    # Detect if the timestamp is the first column or has a specific name
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        # Assume it's the first column
        timestamp_col = df.columns[0]

    print(f"Using column '{timestamp_col}' as timestamp")

    # Resolve error when reading the timestamp column
    try:
        print("Trying to parse timestamp with mixed format...")
        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='mixed')
    except Exception as e:
        print(f"Error parsing timestamp with mixed format: {e}")
        try:
            print("Trying to parse timestamp with auto-inferred format...")
            df['timestamp'] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            print(f"Error with auto-inferred format: {e}")
            print("Trying various common formats...")

            # Try common date formats
            formats = [
                '%m/%d/%Y %H:%M',  # 08/17/2017 5:00
                '%m/%d/%Y %H:%M:%S',  # 08/17/2017 05:00:00
                '%Y-%m-%d %H:%M:%S',  # 2017-08-17 05:00:00
                '%Y-%m-%d %H:%M',  # 2017-08-17 05:00
                '%Y/%m/%d %H:%M',  # 2017/08/17 05:00
                '%d/%m/%Y %H:%M',  # 17/08/2017 05:00
                '%d-%m-%Y %H:%M'  # 17-08-2017 05:00
            ]

            for fmt in formats:
                try:
                    print(f"Trying format: {fmt}")
                    df['timestamp'] = pd.to_datetime(df[timestamp_col], format=fmt)
                    print(f"Successfully parsed with format: {fmt}")
                    break
                except:
                    continue

            # If all formats fail, create a generic timestamp
            if 'timestamp' not in df.columns:
                print("WARNING: Could not parse timestamps. Creating generic time index.")
                df['timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='H')

    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        subset = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        print(f"Filtered data by date range: {start_date} to {end_date}")
    # Or filter by index range if provided
    elif index_range:
        start_idx, end_idx = index_range
        end_idx = min(end_idx, len(df))  # Ensure we don't exceed dataset length
        subset = df.iloc[start_idx:end_idx]
        print(f"Filtered data by index range: {start_idx} to {end_idx}")
    else:
        # Default: use all available data
        subset = df
        print(f"Using all data: {len(df)} rows")

    # Extract prices and timestamps together, dropping rows where price is NaN
    clean_subset = subset.dropna(subset=["price"])

    # Check if we have enough data points
    if len(clean_subset) < 5:
        print("Warning: Not enough valid data points after filtering!")
    else:
        print(f"Working with {len(clean_subset)} data points")

    prices = clean_subset["price"].tolist()
    dates = clean_subset["timestamp"].tolist()

    return prices, dates, df


# Version identification
__version__ = "2.0.0"
__description__ = "Fibonacci Pattern Analysis with Custom Criteria"