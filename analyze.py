import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta


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


# Updated analyze.py - Modified pattern detection functions to only return "failed" or "completed"

def detect_uptrend_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Extract existing uptrend logic but only return patterns with definitive status
    """
    if config is None:
        config = {}

    retracement_target = config.get("retracement_target", 0.5)
    retracement_tolerance = config.get("retracement_tolerance", 0.02)
    completion_extension = config.get("completion_extension", 0.236)
    failure_level_pct = config.get("failure_level", 0.764)
    min_move_multiplier = config.get("min_move_multiplier", 2.0)

    patterns = []

    # Step 1: A is ALWAYS the absolute lowest point in the entire dataset
    A_idx = prices.index(min(prices))
    A_price = min(prices)

    print(f"UPTREND - A point (absolute low): Index {A_idx}, Price ${A_price:.2f}")

    # Step 2: Find all potential B points after A and test for valid C
    valid_AB_pairs = []

    for B_idx in range(A_idx + 1, len(prices)):
        B_price = prices[B_idx]
        move_AB = B_price - A_price

        # Skip if move is too small or negative
        if move_AB <= 0:
            continue

        move_pct = (move_AB / A_price) * 100
        if move_pct < min_change_pct * min_move_multiplier * 100:
            continue

        # Calculate 50% retracement level
        target_C_price = B_price - move_AB * retracement_target
        tolerance_range = move_AB * retracement_tolerance
        min_C_price = target_C_price - tolerance_range
        max_C_price = target_C_price + tolerance_range

        # Look for a valid C point after B
        C_candidates = []
        for i in range(B_idx + 1, len(prices)):
            if min_C_price <= prices[i] <= max_C_price:
                C_candidates.append((i, prices[i]))

        if C_candidates:
            # Take the first valid C point (chronologically)
            C_idx, C_price = C_candidates[0]
            valid_AB_pairs.append((move_AB, B_idx, B_price, C_idx, C_price))

    if not valid_AB_pairs:
        print("UPTREND - No valid A-B-C combinations found")
        return patterns

    # Step 3: Select the best B point (largest move that has valid C)
    valid_AB_pairs.sort(key=lambda x: x[0], reverse=True)

    # Take the top few candidates
    for move_AB, B_idx, B_price, C_idx, C_price in valid_AB_pairs[:3]:

        # Calculate metrics
        move_pct = (move_AB / A_price) * 100
        actual_retracement = B_price - C_price
        retracement_pct = (actual_retracement / move_AB) * 100

        # Calculate key levels
        failure_level = B_price - move_AB * failure_level_pct
        completion_level = B_price + move_AB * completion_extension

        # Step 4: Determine D point and pattern status - ONLY failed or completed
        pattern_status = None
        D_idx = None
        D_price = None

        # Analyze price action after C to find definitive completion or failure
        if C_idx + 1 < len(prices):
            prices_after_C = prices[C_idx + 1:]
            indices_after_C = list(range(C_idx + 1, len(prices)))

            # Check each price after C for definitive status
            for i, price in enumerate(prices_after_C):
                current_idx = indices_after_C[i]

                # Check if price breaks below failure level (76.4%)
                if price < failure_level:
                    D_idx = current_idx
                    D_price = price
                    pattern_status = "failed"
                    break

                # Check if price reaches completion level (-23.6%)
                elif price >= completion_level:
                    D_idx = current_idx
                    D_price = price
                    pattern_status = "completed"
                    break

        # Only create pattern if we have a definitive status
        if pattern_status is not None and D_price is not None:

            # Strict requirement: D must be above A for uptrend
            if pattern_status == "failed" or D_price > A_price:
                # Create pattern
                pattern = {
                    "direction": "up",
                    "A": (A_idx, A_price),
                    "B": (B_idx, B_price),
                    "C": (C_idx, C_price),
                    "D": (D_idx, D_price),
                    "initial_move_pct": move_pct,
                    "retracement_pct": retracement_pct,
                    "target_level": target_C_price,
                    "failure_level": failure_level,
                    "completion_level": completion_level,
                    "status": pattern_status
                }

                patterns.append(pattern)

                print(f"UPTREND pattern created:")
                print(f"  A (absolute low): Index {A_idx}, Price ${A_price:.2f}")
                print(f"  B (high): Index {B_idx}, Price ${B_price:.2f} (move: {move_pct:.2f}%)")
                print(f"  C (50% retrace): Index {C_idx}, Price ${C_price:.2f} (retrace: {retracement_pct:.2f}%)")
                print(f"  D: Index {D_idx}, Price ${D_price:.2f} (status: {pattern_status})")
                print(f"  Failure level (76.4%): ${failure_level:.2f}")
                print(f"  Completion level (-23.6%): ${completion_level:.2f}")
                print()

    return patterns


def detect_downtrend_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Detect downtrend patterns: A (absolute high) → B (low) → C (50% retrace) → D (below A)
    Only return patterns with definitive "failed" or "completed" status
    """
    if config is None:
        config = {}

    retracement_target = config.get("retracement_target", 0.5)
    retracement_tolerance = config.get("retracement_tolerance", 0.02)
    completion_extension = config.get("completion_extension", 0.236)
    failure_level_pct = config.get("failure_level", 0.764)
    min_move_multiplier = config.get("min_move_multiplier", 2.0)

    patterns = []

    # Step 1: A is ALWAYS the absolute highest point in the entire dataset
    A_idx = prices.index(max(prices))
    A_price = max(prices)

    print(f"DOWNTREND - A point (absolute high): Index {A_idx}, Price ${A_price:.2f}")

    valid_AB_pairs = []

    for B_idx in range(A_idx + 1, len(prices)):
        B_price = prices[B_idx]
        move_AB = A_price - B_price  # Downward move

        # Skip if move is too small or negative
        if move_AB <= 0:
            continue

        move_pct = (move_AB / A_price) * 100
        if move_pct < min_change_pct * min_move_multiplier * 100:
            continue

        target_C_price = B_price + move_AB * retracement_target
        tolerance_range = move_AB * retracement_tolerance
        min_C_price = target_C_price - tolerance_range
        max_C_price = target_C_price + tolerance_range

        C_candidates = []
        for i in range(B_idx + 1, len(prices)):
            if min_C_price <= prices[i] <= max_C_price:
                C_candidates.append((i, prices[i]))

        if C_candidates:
            C_idx, C_price = C_candidates[0]
            valid_AB_pairs.append((move_AB, B_idx, B_price, C_idx, C_price))

    if not valid_AB_pairs:
        print("DOWNTREND - No valid A-B-C combinations found")
        return patterns

    # Select the best B point (largest move that has valid C)
    valid_AB_pairs.sort(key=lambda x: x[0], reverse=True)

    # Take the top few candidates
    for move_AB, B_idx, B_price, C_idx, C_price in valid_AB_pairs[:3]:

        move_pct = (move_AB / A_price) * 100
        actual_retracement = C_price - B_price
        retracement_pct = (actual_retracement / move_AB) * 100

        # Calculate key levels
        failure_level = B_price + move_AB * failure_level_pct  # 76.4% back up toward A
        completion_level = B_price - move_AB * completion_extension  # -23.6% extension below B

        # Step 4: Determine D point and pattern status - ONLY failed or completed
        pattern_status = None
        D_idx = None
        D_price = None

        # Analyze price action after C to find definitive completion or failure
        if C_idx + 1 < len(prices):
            prices_after_C = prices[C_idx + 1:]
            indices_after_C = list(range(C_idx + 1, len(prices)))

            # Check each price after C for definitive status
            for i, price in enumerate(prices_after_C):
                current_idx = indices_after_C[i]

                # Check if price breaks above failure level (76.4%)
                if price > failure_level:
                    D_idx = current_idx
                    D_price = price
                    pattern_status = "failed"
                    break

                # Check if price reaches completion level (-23.6%)
                elif price <= completion_level:
                    D_idx = current_idx
                    D_price = price
                    pattern_status = "completed"
                    break

        # Only create pattern if we have a definitive status
        if pattern_status is not None and D_price is not None:

            # Strict requirement: D must be below A for downtrend
            if pattern_status == "failed" or D_price < A_price:
                # Create pattern
                pattern = {
                    "direction": "down",
                    "A": (A_idx, A_price),
                    "B": (B_idx, B_price),
                    "C": (C_idx, C_price),
                    "D": (D_idx, D_price),
                    "initial_move_pct": move_pct,
                    "retracement_pct": retracement_pct,
                    "target_level": target_C_price,
                    "failure_level": failure_level,
                    "completion_level": completion_level,
                    "status": pattern_status
                }

                patterns.append(pattern)

                print(f"DOWNTREND pattern created:")
                print(f"  A (absolute high): Index {A_idx}, Price ${A_price:.2f}")
                print(f"  B (low): Index {B_idx}, Price ${B_price:.2f} (move: {move_pct:.2f}%)")
                print(f"  C (50% retrace): Index {C_idx}, Price ${C_price:.2f} (retrace: {retracement_pct:.2f}%)")
                print(f"  D: Index {D_idx}, Price ${D_price:.2f} (status: {pattern_status})")
                print(f"  Failure level (76.4%): ${failure_level:.2f}")
                print(f"  Completion level (-23.6%): ${completion_level:.2f}")
                print()

    return patterns


# Updated main pattern detection function
def find_significant_price_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Find significant price patterns with only "failed" or "completed" status.
    No more "in_progress" patterns.

    UPTREND:
    1. A is the absolute lowest point in the dataset
    2. B is the highest point after A that has a valid C point (50% retracement)
    3. C is the retracement point at exactly 50% (within tolerance)
    4. D is determined by completion/failure levels - must reach one or the other

    DOWNTREND:
    1. A is the absolute highest point in the dataset
    2. B is the lowest point after A that has a valid C point (50% retracement)
    3. C is the retracement point at exactly 50% (within tolerance)
    4. D is determined by completion/failure levels - must reach one or the other

    Pattern is "failed" when price breaks 76.4% level after C.
    Pattern is "completed" when price reaches -23.6% extension level.
    """

    if config is None:
        config = {}

    patterns = []

    # UPTREND PATTERN DETECTION
    print("=== DETECTING UPTREND PATTERNS (Only Failed/Completed) ===")
    uptrend_patterns = detect_uptrend_patterns(prices, dates, min_change_pct, config)
    patterns.extend(uptrend_patterns)

    # DOWNTREND PATTERN DETECTION
    print("=== DETECTING DOWNTREND PATTERNS (Only Failed/Completed) ===")
    downtrend_patterns = detect_downtrend_patterns(prices, dates, min_change_pct, config)
    patterns.extend(downtrend_patterns)

    print(
        f"Total definitive patterns found: {len(patterns)} ({len(uptrend_patterns)} uptrend, {len(downtrend_patterns)} downtrend)")
    print("All patterns have definitive status: failed or completed")
    return patterns
def find_significant_price_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Find significant price patterns with specific criteria for both uptrend and downtrend:

    UPTREND:
    1. A is the absolute lowest point in the dataset
    2. B is the highest point after A that has a valid C point (50% retracement)
    3. C is the retracement point at exactly 50% (within tolerance)
    4. D is determined by completion/failure levels

    DOWNTREND:
    1. A is the absolute highest point in the dataset
    2. B is the lowest point after A that has a valid C point (50% retracement)
    3. C is the retracement point at exactly 50% (within tolerance)
    4. D is determined by completion/failure levels

    Pattern fails when price breaks 76.4% level after C and D cannot reach completion level.
    """

    if config is None:
        config = {}

    patterns = []

    # UPTREND PATTERN DETECTION
    print("=== DETECTING UPTREND PATTERNS ===")
    uptrend_patterns = detect_uptrend_patterns(prices, dates, min_change_pct, config)
    patterns.extend(uptrend_patterns)

    # DOWNTREND PATTERN DETECTION
    print("=== DETECTING DOWNTREND PATTERNS ===")
    downtrend_patterns = detect_downtrend_patterns(prices, dates, min_change_pct, config)
    patterns.extend(downtrend_patterns)

    print(
        f"Total patterns found: {len(patterns)} ({len(uptrend_patterns)} uptrend, {len(downtrend_patterns)} downtrend)")
    return patterns
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

    # Fibonacci levels with extended levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 0.854, 1.0, 1.236, 1.618, -0.236, -0.618]

    # Different style for failed patterns
    linestyle = '--' if not is_failed else '-.'
    linewidth = 1.5 if not is_failed else 2.0
    alpha = 0.6 if not is_failed else 0.8

    # Level colors
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

    # Add custom levels
    if result and 'failure_level' in result:
        failure_level = result['failure_level']
        ax.axhline(
            y=failure_level,
            color='red',
            linestyle='--',
            alpha=1.0,
            linewidth=2.5,
            label='76.4% Failure Level'
        )
        ax.text(
            dates[0],
            failure_level,
            '76.4% Failure',
            fontsize=12,
            fontweight='bold',
            verticalalignment='center',
            color='red'
        )

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
    Analyze a single Fibonacci pattern - simplified for failed/completed only.
    """
    direction = pattern["direction"]
    retracement_pct = pattern.get("retracement_pct", 0)
    initial_move_pct = pattern.get("initial_move_pct", 0)
    status = pattern.get("status", "unknown")

    A_price = pattern["A"][1]
    B_price = pattern["B"][1]
    C_price = pattern["C"][1]
    D_price = pattern["D"][1]

    if status == "failed":
        analysis = f"FAILED {direction}trend pattern - Price broke 76.4% level"
        analysis += f"\nRetracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"
        analysis += f"\nPattern failed when price reached ${D_price:.2f}"
    elif status == "completed":
        analysis = f"COMPLETED {direction}trend pattern - Price reached -23.6% extension"
        analysis += f"\nRetracement: {retracement_pct:.1f}%, Initial Move: {initial_move_pct:.1f}%"
        analysis += f"\nPattern completed when price reached ${D_price:.2f}"
    else:
        analysis = f"UNKNOWN pattern status: {status}"

    if "failure_level" in pattern:
        analysis += f"\nFailure level (76.4%): ${pattern['failure_level']:.2f}"

    if "completion_level" in pattern:
        analysis += f"\nCompletion level (-23.6%): ${pattern['completion_level']:.2f}"

    return analysis


def plot_diagnostic_graph(prices, dates, result=None):
    """
    Plot a diagnostic graph showing price series, moving averages, and Fibonacci levels.
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
                marker_size = 150 if is_failed else 100
                marker_style = 'x' if is_failed else 'o'

                color_set = colors[i % len(colors)]
                points = {'A': (color_set[0], pattern['A']), 'B': (color_set[1], pattern['B']),
                          'C': (color_set[2], pattern['C']), 'D': (color_set[3], pattern['D'])}

                for label, (color, point) in points.items():
                    idx, price = point
                    ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

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
            is_failed = result.get('status') == 'failed'
            marker_size = 150 if is_failed else 100
            marker_style = 'x' if is_failed else 'o'

            points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                      'C': ('green', result['C']), 'D': ('blue', result['D'])}

            for label, (color, point) in points.items():
                idx, price = point
                ax.scatter(dates[idx], price, color=color, s=marker_size, zorder=3, marker=marker_style)

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

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


def analyze_price_data(prices, dates, min_change_threshold=0.005, pattern_config=None):
    trends = detect_multiple_timeframe_trends(prices)
    print(f"Trends across multiple timeframes: {trends}")

    result = find_significant_price_patterns(
        prices,
        dates,
        min_change_threshold,
        config=pattern_config
    )

    analysis = analyze_move(result) if result else "No valid pattern found."

    return result, analysis, trends


def display_results(prices, dates, result, analysis, trends):
    """
    Display the results of the Fibonacci pattern analysis - updated for failed/completed only.
    """
    # Create and show the diagnostic plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot the price series with small markers to see actual movements
    ax.plot(dates, prices, marker='o', linestyle='-', color='black', alpha=0.7,
            label='Price Series', markersize=3)

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

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
    ax.set_title(f'Pattern Analysis - {move_type} Move (Only Failed/Completed Patterns)\nTrends: {trend_str}',
                 fontsize=14)

    # Print analysis results
    print("\nPattern Analysis Results (Failed/Completed Only):")
    print("-" * 60)
    print(f"Overall trend direction: {trends.get('overall', 'unknown')}")
    print(f"Analysis: {analysis}")

    # Draw Fibonacci levels and pattern points
    if result:
        if isinstance(result, list):
            print(f"Detected {len(result)} definitive patterns")
            colors = [('black', 'red', 'green', 'blue'), ('purple', 'orange', 'cyan', 'magenta')]

            for i, pattern in enumerate(result):
                print(f"\nPattern {i + 1}:")
                print(f"Direction: {pattern['direction']}")
                print(f"Status: {pattern['status'].upper()}")
                print(f"Initial move: {pattern['initial_move_pct']:.2f}%")
                print(f"Retracement: {pattern['retracement_pct']:.2f}%")

                is_failed = pattern.get('status') == 'failed'
                is_completed = pattern.get('status') == 'completed'

                # Use different styling for failed vs completed
                if is_failed:
                    marker_size = 200
                    marker_style = 'X'  # Large X for failed
                    line_style = '--'
                    line_color = 'red'
                    line_width = 3
                elif is_completed:
                    marker_size = 150
                    marker_style = 'o'  # Circle for completed
                    line_style = '-'
                    line_color = 'green'
                    line_width = 2
                else:
                    marker_size = 100
                    marker_style = 'o'
                    line_style = '-'
                    line_color = 'gray'
                    line_width = 1

                color_set = colors[i % len(colors)]
                points = {'A': (color_set[0], pattern['A']), 'B': (color_set[1], pattern['B']),
                          'C': (color_set[2], pattern['C']), 'D': (color_set[3], pattern['D'])}

                for label, (color, point) in points.items():
                    idx, price = point

                    if idx >= len(dates):
                        print(f"Warning: Point {label} of pattern {i + 1} has index {idx} outside range.")
                        idx = len(dates) - 1

                    # Special styling for D point based on status
                    if label == 'D':
                        ax.scatter(dates[idx], price, color=line_color, s=marker_size,
                                   zorder=3, marker=marker_style, edgecolors='black', linewidth=2)

                        if is_failed:
                            ax.text(dates[idx], price, f"{label}{i + 1} (FAILED)", fontsize=14, fontweight='bold',
                                    ha='right', va='bottom', color='red',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', alpha=0.8))
                        elif is_completed:
                            ax.text(dates[idx], price, f"{label}{i + 1} (COMPLETED)", fontsize=14, fontweight='bold',
                                    ha='right', va='bottom', color='green',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
                    else:
                        ax.scatter(dates[idx], price, color=color, s=100, zorder=3, marker='o')
                        ax.text(dates[idx], price, f"{label}{i + 1}", fontsize=14, fontweight='bold',
                                ha='right', va='bottom', color=color)

                    print(f"{label}: Index {idx}, Date {dates[idx].strftime('%Y-%m-%d %H:%M')}, Price ${price:.2f}")

                # Draw connecting lines with status-based styling
                points_coords = [(pattern['A'][0], pattern['A'][1]), (pattern['B'][0], pattern['B'][1]),
                                 (pattern['C'][0], pattern['C'][1]), (pattern['D'][0], pattern['D'][1])]
                x_points, y_points = zip(*[(dates[idx], price) for idx, price in points_coords if idx < len(dates)])
                ax.plot(x_points, y_points, color=line_color, linestyle=line_style,
                        linewidth=line_width, alpha=0.8,
                        label=f'Pattern {i + 1} ({pattern["status"].upper()})')

                # Draw key levels
                draw_fibonacci_levels(ax, prices, dates, pattern, pattern['direction'], is_failed=is_failed)

                # Print key levels
                print(f"\nKey levels for Pattern {i + 1}:")
                print(f"Failure level (76.4%): ${pattern['failure_level']:.2f}")
                print(f"Completion level (-23.6%): ${pattern['completion_level']:.2f}")
                print(f"Status: {pattern['status'].upper()}")

        else:
            # Single pattern
            print(f"Detected trend: {result['direction']}")
            print(f"Status: {result['status'].upper()}")
            print(f"Initial move: {result['initial_move_pct']:.2f}%")
            print(f"Retracement: {result['retracement_pct']:.2f}%")

            is_failed = result.get('status') == 'failed'
            is_completed = result.get('status') == 'completed'

            # Status-based styling
            if is_failed:
                marker_size = 200
                marker_style = 'X'
                line_color = 'red'
                line_style = '--'
            elif is_completed:
                marker_size = 150
                marker_style = 'o'
                line_color = 'green'
                line_style = '-'
            else:
                marker_size = 100
                marker_style = 'o'
                line_color = 'gray'
                line_style = '-'

            points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                      'C': ('green', result['C']), 'D': (line_color, result['D'])}

            for label, (color, point) in points.items():
                idx, price = point

                if idx >= len(dates):
                    print(f"Warning: Point {label} has index {idx} outside range.")
                    idx = len(dates) - 1

                if label == 'D':
                    ax.scatter(dates[idx], price, color=line_color, s=marker_size,
                               zorder=3, marker=marker_style, edgecolors='black', linewidth=2)

                    status_text = "FAILED" if is_failed else "COMPLETED" if is_completed else "UNKNOWN"
                    text_color = 'red' if is_failed else 'green' if is_completed else 'gray'
                    bg_color = 'pink' if is_failed else 'lightgreen' if is_completed else 'lightgray'

                    ax.text(dates[idx], price, f"{label} ({status_text})", fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color=text_color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.8))
                else:
                    ax.scatter(dates[idx], price, color=color, s=100, zorder=3, marker='o')
                    ax.text(dates[idx], price, label, fontsize=14, fontweight='bold',
                            ha='right', va='bottom', color=color)

                print(f"{label}: Index {idx}, Date {dates[idx].strftime('%Y-%m-%d %H:%M')}, Price ${price:.2f}")

            # Draw connecting line
            points_coords = [(result['A'][0], result['A'][1]), (result['B'][0], result['B'][1]),
                             (result['C'][0], result['C'][1]), (result['D'][0], result['D'][1])]
            x_points, y_points = zip(*[(dates[idx], price) for idx, price in points_coords if idx < len(dates)])
            ax.plot(x_points, y_points, color=line_color, linestyle=line_style,
                    linewidth=3, alpha=0.8, label=f'Pattern ({result["status"].upper()})')

            draw_fibonacci_levels(ax, prices, dates, result, result['direction'], is_failed=is_failed)

            print(f"\nKey levels:")
            print(f"Failure level (76.4%): ${result['failure_level']:.2f}")
            print(f"Completion level (-23.6%): ${result['completion_level']:.2f}")
            print(f"Final status: {result['status'].upper()}")

    else:
        print("No definitive patterns detected (failed or completed).")
        draw_fibonacci_levels(ax, prices, dates, None, direction)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


def analyze_multiple_windows(prices, dates, window_sizes=[50, 100, 200],
                             overlap_percent=50, min_change_threshold=0.001,
                             allow_multiple_patterns=True, detect_long_failures=True,
                             pattern_config=None):
    """
    Analyze the same price series with different window sizes using updated pattern detection.
    """
    all_patterns = []

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

            # Use the updated pattern detection method
            window_patterns = find_significant_price_patterns(
                window_prices,
                window_dates,
                min_change_threshold,
                config=pattern_config
            )

            # If patterns were found, analyze them and add to results
            if window_patterns:
                window_info = {
                    "window_size": window_size,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": dates[start_idx],
                    "end_date": dates[end_idx - 1],
                    "window_prices": window_prices,
                    "window_dates": window_dates
                }

                for pattern in window_patterns:
                    pattern_analysis = analyze_single_pattern(pattern)
                    all_patterns.append((pattern, pattern_analysis, window_info))

    # Sort patterns by significance
    all_patterns.sort(key=lambda x: pattern_significance(x[0]), reverse=True)

    return all_patterns


def pattern_significance(pattern):
    """
    Calculate a significance score for a pattern.
    Prioritize completed and failed patterns since we only have those now.
    """
    if 'initial_move_pct' in pattern:
        score = min(pattern['initial_move_pct'] / 5, 5)  # Cap at 5 for moves >= 25%

        # Add bonus for completed patterns (they reached target)
        if pattern.get('status') == 'completed':
            score += 5  # High bonus for completion
        elif pattern.get('status') == 'failed':
            score += 3  # Significant bonus for failure (still important)

        # Add bonus for retracement close to 50%
        retrace_pct = pattern.get('retracement_pct', 0)
        retrace_quality = 1.0 - abs(retrace_pct - 50) / 50  # 1.0 for perfect 50%, 0 for 0% or 100%
        score += retrace_quality * 2

        return score

    # Fallback calculation
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

    # Ideal retracement is around 50%
    retracement_quality = 1.0 - abs(retracement_ratio - 0.5)

    # Size of the overall move matters
    move_size = abs(D - A)

    # Bonus for definitive status
    status_bonus = 2.0 if pattern.get('status') in ['failed', 'completed'] else 0

    # Combine factors into a single score
    score = (
            move_size * 0.4 +
            retracement_quality * 0.3 +
            status_bonus
    )

    return score

def create_longterm_pattern_figure(prices, dates, all_patterns):
    """
    Create a dedicated figure for significant patterns across the entire dataset.
    """
    # Extract patterns with significant moves (over 2%)
    significant_patterns = []

    for pattern_idx, (pattern, analysis, window_info) in enumerate(all_patterns):
        if isinstance(pattern, dict) and pattern.get('initial_move_pct', 0) >= 2.0:
            significant_patterns.append((pattern_idx, pattern, analysis, window_info))

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

        # Convert to global indices
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
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx]],
                    [A_price, B_price, C_price],
                    linestyle='--', color=color, linewidth=2, alpha=0.7)
        else:
            ax.plot([dates[global_A_idx], dates[global_B_idx], dates[global_C_idx], dates[global_D_idx]],
                    [A_price, B_price, C_price, D_price],
                    linestyle='-', color=color, linewidth=2, alpha=0.7)

        # Add pattern information
        direction = pattern['direction']
        move_pct = pattern.get('initial_move_pct', 0)
        retrace_pct = pattern.get('retracement_pct', 0)

        text_y_pos = min(prices) + (i * (max(prices) - min(prices)) * 0.05)
        text_x_pos = dates[int(len(dates) * 0.05)]

        info_text = f"Pattern {pattern_idx + 1}: {direction.upper()}-trend "
        info_text += f"{pattern.get('status', 'unknown').upper()} "
        info_text += f"(Move: {move_pct:.1f}%, Retrace: {retrace_pct:.1f}%)"

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
    fig = plt.figure(figsize=(18, 14))

    # Main chart
    chart_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Info panel
    info_ax = plt.subplot2grid((5, 1), (4, 0))
    info_ax.axis('off')

    # Plot the price series
    chart_ax.plot(window_dates, window_prices, marker='o', linestyle='-', color='black', alpha=0.7,
                  label='Price Series', markersize=3)

    # Get pattern info
    if isinstance(result, list):
        pattern = result[0]
    else:
        pattern = result

    direction = pattern['direction']
    status = pattern.get('status', 'unknown')
    is_failed = status == 'failed'
    move_pct = pattern.get('initial_move_pct', 0)
    retrace_pct = pattern.get('retracement_pct', 0)

    # Plot each point (A, B, C, D)
    colors = {'A': 'black', 'B': 'red', 'C': ('darkred' if is_failed else 'green'), 'D': 'blue'}
    markers = {'A': 'o', 'B': 's', 'C': ('X' if is_failed else '^'), 'D': 'd'}
    sizes = {'A': 120, 'B': 120, 'C': 160 if is_failed else 120, 'D': 120}

    for label in ['A', 'B', 'C', 'D']:
        idx, price = pattern[label]

        if idx < len(window_dates):
            chart_ax.scatter(window_dates[idx], price, color=colors[label],
                             marker=markers[label], s=sizes[label], zorder=4, edgecolors='black')

            if is_failed and label == 'C':
                chart_ax.text(window_dates[idx], price, f"{label} (FAILED)", fontsize=14,
                              fontweight='bold', color='darkred', ha='right', va='bottom')
            else:
                chart_ax.text(window_dates[idx], price, label, fontsize=14,
                              fontweight='bold', color=colors[label], ha='right', va='bottom')

    # Draw Fibonacci levels
    draw_fibonacci_levels(chart_ax, window_prices, window_dates, pattern, direction, is_failed=is_failed)

    # Connect points with lines
    if is_failed:
        x_points = [window_dates[pattern['A'][0]], window_dates[pattern['B'][0]],
                    window_dates[pattern['C'][0]]]
        y_points = [pattern['A'][1], pattern['B'][1], pattern['C'][1]]
        chart_ax.plot(x_points, y_points, 'r--', linewidth=2, alpha=0.7)
    else:
        x_points = [window_dates[pattern['A'][0]], window_dates[pattern['B'][0]],
                    window_dates[pattern['C'][0]], window_dates[pattern['D'][0]]]
        y_points = [pattern['A'][1], pattern['B'][1], pattern['C'][1], pattern['D'][1]]
        chart_ax.plot(x_points, y_points, 'g-', linewidth=2, alpha=0.7)

    # Set title and axis labels
    title_text = f"Pattern {pattern_number} - {status.upper()} - {direction.upper()}TREND"
    chart_ax.set_title(title_text, fontsize=16, fontweight='bold', color='darkred' if is_failed else 'darkgreen')

    chart_ax.set_xlabel('Date', fontsize=12)
    chart_ax.set_ylabel('Price', fontsize=12)
    chart_ax.grid(True, alpha=0.3)
    chart_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    chart_ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Information Panel
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

    # Add custom levels
    if 'failure_level' in pattern:
        info_text += f"• 76.4% Failure level: ${pattern['failure_level']:.2f}\n"
    if 'completion_level' in pattern:
        info_text += f"• -23.6% Completion level: ${pattern['completion_level']:.2f}\n"
    if 'target_level' in pattern:
        info_text += f"• 50% Target level: ${pattern['target_level']:.2f}\n"

    # Add analysis summary
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
    """
    # Read the CSV file with datetime parsing
    df = pd.read_csv(csv_file)

    # Check first few rows to understand the structure
    print("First few rows of the CSV file:")
    print(df.head(2))

    # Detect timestamp column
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        timestamp_col = df.columns[0]

    print(f"Using column '{timestamp_col}' as timestamp")

    # Parse timestamps
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

            formats = [
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M',
                '%d/%m/%Y %H:%M',
                '%d-%m-%Y %H:%M'
            ]

            for fmt in formats:
                try:
                    print(f"Trying format: {fmt}")
                    df['timestamp'] = pd.to_datetime(df[timestamp_col], format=fmt)
                    print(f"Successfully parsed with format: {fmt}")
                    break
                except:
                    continue

            if 'timestamp' not in df.columns:
                print("WARNING: Could not parse timestamps. Creating generic time index.")
                df['timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='H')

    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        subset = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        print(f"Filtered data by date range: {start_date} to {end_date}")
    elif index_range:
        start_idx, end_idx = index_range
        end_idx = min(end_idx, len(df))
        subset = df.iloc[start_idx:end_idx]
        print(f"Filtered data by index range: {start_idx} to {end_idx}")
    else:
        subset = df
        print(f"Using all data: {len(df)} rows")

    # Extract prices and timestamps, dropping NaN values
    clean_subset = subset.dropna(subset=["price"])

    if len(clean_subset) < 5:
        print("Warning: Not enough valid data points after filtering!")
    else:
        print(f"Working with {len(clean_subset)} data points")

    prices = clean_subset["price"].tolist()
    dates = clean_subset["timestamp"].tolist()

    return prices, dates, df


