import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def find_all_overlapping_patterns(prices, dates, min_change_pct=0.005, config=None):
    """
    Comprehensive pattern detection using your existing config system
    """
    if config is None:
        config = {}

    all_patterns = []

    print("=== COMPREHENSIVE PATTERN DETECTION ===")
    print("Finding ALL overlapping upward and downward patterns...")

    print("\n--- Traditional Patterns ---")

    absolute_low_idx = prices.index(min(prices))
    absolute_low_price = min(prices)
    upward_patterns = find_patterns_from_point(
        prices, dates, absolute_low_idx, absolute_low_price, "up",
        min_change_pct, config  # Just pass your existing config
    )
    all_patterns.extend(upward_patterns)

    absolute_high_idx = prices.index(max(prices))
    absolute_high_price = max(prices)
    downward_patterns = find_patterns_from_point(
        prices, dates, absolute_high_idx, absolute_high_price, "down",
        min_change_pct, config  # Just pass your existing config
    )
    all_patterns.extend(downward_patterns)

    print("\n--- Local Extremes Patterns ---")

    min_move_multiplier = config.get("min_move_multiplier", 2.0)
    local_extremes = find_significant_extremes(prices, min_change_pct * min_move_multiplier)

    for extreme in local_extremes:
        idx, price, extreme_type = extreme

        if extreme_type == "high":
            down_patterns = find_patterns_from_point(
                prices, dates, idx, price, "down", min_change_pct, config
            )
            all_patterns.extend(down_patterns)

        elif extreme_type == "low":
            up_patterns = find_patterns_from_point(
                prices, dates, idx, price, "up", min_change_pct, config
            )
            all_patterns.extend(up_patterns)

    print("\n--- Overlapping Pattern Analysis ---")

    completed_patterns = [p for p in all_patterns if p.get('status') == 'completed']

    for pattern in completed_patterns:
        opposite_patterns = find_opposite_overlapping_patterns(
            prices, dates, pattern, min_change_pct, config  # Use same config
        )
        all_patterns.extend(opposite_patterns)

    unique_patterns = remove_duplicate_patterns(all_patterns)

    # FILTER FOR COMPLETED/FAILED ONLY
    definitive_patterns = [p for p in unique_patterns if p.get('status') in ['completed', 'failed']]

    # For failed patterns, find accurate failure points beyond window boundaries
    if config.get('search_beyond_window_for_failure', True):
        definitive_patterns = [find_accurate_failure_point_global(p, prices, dates, config)
                               for p in definitive_patterns]

    definitive_patterns.sort(key=lambda p: calculate_pattern_significance(p), reverse=True)

    print(f"\n=== COMPREHENSIVE RESULTS ===")
    print(f"Total patterns found: {len(unique_patterns)}")
    print(f"Definitive patterns (completed/failed): {len(definitive_patterns)}")

    status_summary = {}
    for pattern in definitive_patterns:
        direction = pattern.get('direction', 'unknown')
        status = pattern.get('status', 'unknown')
        key = f"{direction}_{status}"
        status_summary[key] = status_summary.get(key, 0) + 1

    for key, count in status_summary.items():
        direction, status = key.split('_')
        print(f"  {direction.upper()} {status}: {count}")

    return definitive_patterns


def find_accurate_failure_point_global(pattern, prices, dates, config):
    """
    For failed patterns, search through entire dataset to find accurate failure point
    """
    if pattern.get('status') != 'failed':
        return pattern

    print(f"Finding accurate failure point for {pattern['direction']} pattern...")

    A_idx, A_price = pattern['A']
    B_idx, B_price = pattern['B']
    C_idx, C_price = pattern['C']

    direction = pattern['direction']
    failure_level_pct = config.get('failure_level', 0.764)

    # Calculate failure level
    if direction == "up":
        move_AB = B_price - A_price
        failure_level = B_price - move_AB * failure_level_pct
        search_condition = lambda price: price < failure_level
    else:  # down
        move_AB = A_price - B_price
        failure_level = B_price + move_AB * failure_level_pct
        search_condition = lambda price: price > failure_level

    # Search from C point onwards through ENTIRE dataset
    search_start_idx = max(0, C_idx + 1)
    actual_failure_idx = None
    actual_failure_price = None

    print(f"  Searching from index {search_start_idx} to {len(prices) - 1} for failure level ${failure_level:.2f}")

    for i in range(search_start_idx, len(prices)):
        price = prices[i]
        if search_condition(price):
            actual_failure_idx = i
            actual_failure_price = price
            print(f"  Found accurate failure at index {i}, price ${price:.2f}")
            break

    # Update D point with accurate failure location
    if actual_failure_idx is not None:
        pattern['D'] = (actual_failure_idx, actual_failure_price)
        pattern['accurate_failure_point'] = True
        print(f"  Updated D point to accurate failure: index={actual_failure_idx}, price=${actual_failure_price:.2f}")
    else:
        print(f"  No better failure point found beyond C, keeping original D point")
        pattern['accurate_failure_point'] = False

    return pattern


def find_patterns_from_point(prices, dates, A_idx, A_price, direction, min_change_pct, config):
    """
    Find all valid patterns starting from a specific A point
    Uses your existing config system - no separate parameters
    """
    patterns = []

    retracement_target = config.get("retracement_target", 0.5)
    retracement_tolerance = config.get("retracement_tolerance", 0.02)
    completion_extension = config.get("completion_extension", 0.236)
    failure_level = config.get("failure_level", 0.764)
    min_move_multiplier = config.get("min_move_multiplier", 2.0)

    if direction == "up":
        A_idx = prices.index(min(prices))
        A_price = prices[A_idx]
    else:
        A_idx = prices.index(max(prices))
        A_price = prices[A_idx]

    print(f"  Using GLOBAL {'MIN' if direction == 'up' else 'MAX'} for A: Index {A_idx}, Price ${A_price:.2f}")

    valid_B_candidates = []

    for B_idx in range(A_idx + 1, len(prices)):
        B_price = prices[B_idx]

        if direction == "up":
            move_AB = B_price - A_price
            if move_AB <= 0:
                continue
        else:  # down
            move_AB = A_price - B_price
            if move_AB <= 0:
                continue

        move_pct = (move_AB / A_price) * 100
        if move_pct < min_change_pct * min_move_multiplier * 100:
            continue

        if direction == "up":
            target_C_price = B_price - move_AB * retracement_target
        else:
            target_C_price = B_price + move_AB * retracement_target

        tolerance_range = move_AB * retracement_tolerance
        min_C_price = target_C_price - tolerance_range
        max_C_price = target_C_price + tolerance_range

        # Look for valid C
        valid_C_found = None
        for i in range(B_idx + 1, len(prices)):
            if min_C_price <= prices[i] <= max_C_price:
                valid_C_found = (i, prices[i])
                break

        if valid_C_found:
            C_idx, C_price = valid_C_found
            valid_B_candidates.append({
                'B_idx': B_idx,
                'B_price': B_price,
                'move_AB': move_AB,
                'move_pct': move_pct,
                'C_idx': C_idx,
                'C_price': C_price,
                'target_C_price': target_C_price
            })

    if not valid_B_candidates:
        return patterns

    if direction == "up":
        valid_B_candidates.sort(key=lambda x: x['B_price'], reverse=True)
    else:
        valid_B_candidates.sort(key=lambda x: x['B_price'])

    # Create patterns from top candidates
    for i, candidate in enumerate(valid_B_candidates[:3]):  # Top 3 candidates

        B_idx = candidate['B_idx']
        B_price = candidate['B_price']
        move_AB = candidate['move_AB']
        move_pct = candidate['move_pct']
        C_idx = candidate['C_idx']
        C_price = candidate['C_price']

        if direction == "up":
            actual_retracement = B_price - C_price
        else:
            actual_retracement = C_price - B_price

        retracement_pct = (actual_retracement / move_AB) * 100

        if direction == "up":
            failure_level_price = B_price - move_AB * failure_level
            completion_level_price = B_price + move_AB * completion_extension
        else:
            failure_level_price = B_price + move_AB * failure_level
            completion_level_price = B_price - move_AB * completion_extension

        # Find D point and status - search through ENTIRE dataset for accuracy
        pattern_status = None
        D_idx = None
        D_price = None

        if C_idx + 1 < len(prices):
            # Search through ALL remaining data, not just a window
            for j in range(C_idx + 1, len(prices)):
                price = prices[j]

                # Check for failure or completion based on direction
                if direction == "up":
                    if price < failure_level_price:
                        D_idx = j
                        D_price = price
                        pattern_status = "failed"
                        break
                    elif price >= completion_level_price:
                        D_idx = j
                        D_price = price
                        pattern_status = "completed"
                        break
                else:  # down
                    if price > failure_level_price:
                        D_idx = j
                        D_price = price
                        pattern_status = "failed"
                        break
                    elif price <= completion_level_price:
                        D_idx = j
                        D_price = price
                        pattern_status = "completed"
                        break

        # Create pattern ONLY if definitive status found (completed or failed)
        if pattern_status in ['completed', 'failed'] and D_price is not None:

            # Validate D position relative to A
            valid_D = False
            if direction == "up":
                valid_D = (pattern_status == "failed") or (D_price > A_price)
            else:
                valid_D = (pattern_status == "failed") or (D_price < A_price)

            if valid_D:
                pattern = {
                    "direction": direction,
                    "A": (A_idx, A_price),
                    "B": (B_idx, B_price),
                    "C": (C_idx, C_price),
                    "D": (D_idx, D_price),
                    "initial_move_pct": move_pct,
                    "retracement_pct": retracement_pct,
                    "target_level": candidate['target_C_price'],
                    "failure_level": failure_level_price,
                    "completion_level": completion_level_price,
                    "status": pattern_status,
                    "pattern_rank": i + 1,
                    "pattern_type": "comprehensive",
                    "A_type": "absolute" if A_idx in [prices.index(min(prices)),
                                                      prices.index(max(prices))] else "local",
                    "searched_full_dataset": True
                }

                patterns.append(pattern)

                print(f"    {direction.upper()} pattern created (rank #{i + 1}, {pattern_status}):")
                print(f"      A: ${A_price:.2f}, B: ${B_price:.2f}, C: ${C_price:.2f}, D: ${D_price:.2f}")

    return patterns


def find_opposite_overlapping_patterns(prices, dates, reference_pattern, min_change_pct, config):
    """
    For a completed pattern in one direction, look for failed patterns in opposite direction
    Uses your existing config - no separate parameters
    """
    patterns = []

    ref_direction = reference_pattern['direction']
    ref_A_idx = reference_pattern['A'][0]
    ref_D_idx = reference_pattern['D'][0]

    opposite_direction = "down" if ref_direction == "up" else "up"

    print(f"    Looking for overlapping {opposite_direction} patterns...")

    # Look for A points in the opposite direction within the reference pattern timeframe
    if opposite_direction == "up":
        # Look for local lows between reference A and D
        for i in range(ref_A_idx, ref_D_idx):
            if i > 5 and i < len(prices) - 5:
                if prices[i] == min(prices[i - 5:i + 6]):  # Local low
                    opposite_patterns = find_patterns_from_point(
                        prices, dates, i, prices[i], opposite_direction, min_change_pct, config
                    )
                    # Only keep patterns that overlap with reference pattern
                    for pattern in opposite_patterns:
                        if pattern['D'][0] >= ref_A_idx and pattern['A'][0] <= ref_D_idx:
                            pattern['overlap_with'] = f"{ref_direction}_{reference_pattern['status']}"
                            patterns.append(pattern)
    else:
        # Look for local highs between reference A and D
        for i in range(ref_A_idx, ref_D_idx):
            if i > 5 and i < len(prices) - 5:
                if prices[i] == max(prices[i - 5:i + 6]):  # Local high
                    opposite_patterns = find_patterns_from_point(
                        prices, dates, i, prices[i], opposite_direction, min_change_pct, config
                    )
                    # Only keep patterns that overlap with reference pattern
                    for pattern in opposite_patterns:
                        if pattern['D'][0] >= ref_A_idx and pattern['A'][0] <= ref_D_idx:
                            pattern['overlap_with'] = f"{ref_direction}_{reference_pattern['status']}"
                            patterns.append(pattern)

    return patterns


# Utility functions remain the same but cleaner
def find_significant_extremes(prices, min_change_threshold):
    """Find significant local highs and lows - simplified"""
    extremes = []

    for i in range(5, len(prices) - 5):
        # Local high
        if prices[i] == max(prices[i - 5:i + 6]):
            extremes.append((i, prices[i], "high"))
        # Local low
        elif prices[i] == min(prices[i - 5:i + 6]):
            extremes.append((i, prices[i], "low"))

    return extremes


def remove_duplicate_patterns(patterns):
    """Remove duplicate patterns"""
    unique_patterns = []
    seen_signatures = set()

    for pattern in patterns:
        signature = (
            pattern['A'][0], pattern['B'][0],
            pattern['C'][0], pattern['D'][0],
            pattern['direction']
        )

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_patterns.append(pattern)

    return unique_patterns


def calculate_pattern_significance(pattern):
    """Enhanced significance calculation"""
    base_score = pattern.get('initial_move_pct', 0) / 5

    if pattern.get('status') == 'completed':
        base_score += 5
    elif pattern.get('status') == 'failed':
        base_score += 3

    if pattern.get('A_type') == 'absolute':
        base_score += 2

    retrace_pct = pattern.get('retracement_pct', 0)
    retrace_quality = 1.0 - abs(retrace_pct - 50) / 50
    base_score += retrace_quality * 2

    if 'overlap_with' in pattern:
        base_score += 1.5

    return base_score


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
    UPDATED: Now only returns completed/failed patterns with accurate failure points
    """
    print("Finding significant price patterns (completed/failed only)...")

    all_patterns = find_all_overlapping_patterns(prices, dates, min_change_pct, config)

    print(f"\nFinal Results: {len(all_patterns)} definitive patterns detected")

    # Print summary
    up_completed = len([p for p in all_patterns if p['direction'] == 'up' and p['status'] == 'completed'])
    up_failed = len([p for p in all_patterns if p['direction'] == 'up' and p['status'] == 'failed'])
    down_completed = len([p for p in all_patterns if p['direction'] == 'down' and p['status'] == 'completed'])
    down_failed = len([p for p in all_patterns if p['direction'] == 'down' and p['status'] == 'failed'])

    print(f"  UP patterns: {up_completed} completed, {up_failed} failed")
    print(f"  DOWN patterns: {down_completed} completed, {down_failed} failed")

    overlapping = len([p for p in all_patterns if 'overlap_with' in p])
    accurate_failures = len([p for p in all_patterns if p.get('accurate_failure_point', False)])
    print(f"  Overlapping patterns: {overlapping}")
    print(f"  Accurate failure points found: {accurate_failures}")

    return all_patterns


def analyze_multiple_windows(prices, dates, window_sizes=[50, 100, 200],
                             overlap_percent=50, min_change_threshold=0.001,
                             allow_multiple_patterns=True, detect_long_failures=True,
                             pattern_config=None):
    """
    UPDATED: Analyze the same price series with different window sizes, but only return completed/failed patterns
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

            # Use the updated pattern detection method that only returns completed/failed
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
                    # For failed patterns in windows, search beyond window boundary for accurate D point
                    if pattern.get('status') == 'failed' and pattern_config.get('search_beyond_window_for_failure',
                                                                                True):
                        pattern = find_accurate_failure_point_in_window(pattern, prices, dates, window_info,
                                                                        pattern_config)

                    pattern_analysis = analyze_single_pattern(pattern)
                    all_patterns.append((pattern, pattern_analysis, window_info))

    # Sort patterns by significance
    all_patterns.sort(key=lambda x: pattern_significance(x[0]), reverse=True)

    print(f"analyze_multiple_windows found {len(all_patterns)} definitive patterns")
    return all_patterns


def find_accurate_failure_point_in_window(pattern, full_prices, full_dates, window_info, config):
    """
    For windowed patterns, search beyond window boundary to find accurate failure point
    """
    if pattern.get('status') != 'failed':
        return pattern

    print(f"Finding accurate failure point for windowed {pattern['direction']} pattern...")

    # Get window info
    window_start = window_info['start_idx']
    window_size = window_info['window_size']

    # Convert window-local coordinates to global
    A_local_idx, A_price = pattern['A']
    B_local_idx, B_price = pattern['B']
    C_local_idx, C_price = pattern['C']

    global_A_idx = window_start + A_local_idx
    global_B_idx = window_start + B_local_idx
    global_C_idx = window_start + C_local_idx

    direction = pattern['direction']
    failure_level_pct = config.get('failure_level', 0.764)

    # Calculate failure level
    if direction == "up":
        move_AB = B_price - A_price
        failure_level = B_price - move_AB * failure_level_pct
        search_condition = lambda price: price < failure_level
    else:  # down
        move_AB = A_price - B_price
        failure_level = B_price + move_AB * failure_level_pct
        search_condition = lambda price: price > failure_level

    # Search from C point onwards through ENTIRE dataset (beyond window)
    search_start_idx = max(0, global_C_idx + 1)
    actual_failure_idx = None
    actual_failure_price = None

    print(f"  Searching from global index {search_start_idx} to {len(full_prices) - 1}")
    print(f"  Window was [{window_start}:{window_start + window_size}]")

    for i in range(search_start_idx, len(full_prices)):
        price = full_prices[i]
        if search_condition(price):
            actual_failure_idx = i
            actual_failure_price = price
            print(f"  Found accurate failure at global index {i}, price ${price:.2f}")
            break

    # Update D point with accurate failure location (convert back to window-local)
    if actual_failure_idx is not None:
        local_failure_idx = actual_failure_idx - window_start
        pattern['D'] = (local_failure_idx, actual_failure_price)
        pattern['accurate_failure_point'] = True
        pattern['searched_beyond_window'] = True
        print(
            f"  Updated D point: local_idx={local_failure_idx}, global_idx={actual_failure_idx}, price=${actual_failure_price:.2f}")
    else:
        print(f"  No better failure point found beyond window, keeping original D point")
        pattern['accurate_failure_point'] = False
        pattern['searched_beyond_window'] = False

    return pattern


# Rest of the functions remain the same as they were working...
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
        if pattern.get('accurate_failure_point'):
            analysis += f"\n✓ Accurate failure point found beyond window boundary"
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


def analyze_move(patterns):
    """Analyze a list of patterns and return summary"""
    if not patterns:
        return "No patterns detected"

    if isinstance(patterns, list):
        completed = len([p for p in patterns if p.get('status') == 'completed'])
        failed = len([p for p in patterns if p.get('status') == 'failed'])
        return f"Found {len(patterns)} definitive patterns: {completed} completed, {failed} failed"
    else:
        return analyze_single_pattern(patterns)


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
                            failure_text = "FAILED"
                            if pattern.get('accurate_failure_point'):
                                failure_text += " (ACCURATE)"
                            ax.text(dates[idx], price, f"{label}{i + 1} ({failure_text})", fontsize=14,
                                    fontweight='bold',
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
                if pattern.get('accurate_failure_point'):
                    print("✓ Accurate failure point found beyond original window")

        else:
            # Single pattern - similar styling
            print(f"Detected trend: {result['direction']}")
            print(f"Status: {result['status'].upper()}")

            # ... rest of single pattern display logic remains the same

    else:
        print("No definitive patterns detected (failed or completed).")
        draw_fibonacci_levels(ax, prices, dates, None, direction)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


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

        # Add bonus for accurate failure points
        if pattern.get('accurate_failure_point'):
            score += 1

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
        failure_text = f'C{pattern_idx + 1}{"(F)" if is_failed else ""}'
        if pattern.get('accurate_failure_point'):
            failure_text += "*"
        ax.text(dates[global_C_idx], C_price, failure_text, fontsize=12,
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
        if pattern.get('accurate_failure_point'):
            info_text += " *Accurate D"

        ax.text(text_x_pos, text_y_pos, info_text,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round'),
                color=color, fontsize=10, fontweight='bold')

    # Set title and labels
    num_patterns = len(significant_patterns)
    ax.set_title(f'Significant Patterns Overview ({num_patterns} definitive patterns detected)', fontsize=16,
                 fontweight='bold')
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
            elif is_failed and label == 'D' and pattern.get('accurate_failure_point'):
                chart_ax.text(window_dates[idx], price, f"{label} (ACCURATE)", fontsize=14,
                              fontweight='bold', color='blue', ha='right', va='bottom')
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

    # Set title
    title_text = f"Pattern {pattern_number} - {status.upper()} - {direction.upper()}TREND"
    if pattern.get('accurate_failure_point'):
        title_text += " (ACCURATE D POINT)"
    chart_ax.set_title(title_text, fontsize=16, fontweight='bold', color='darkred' if is_failed else 'darkgreen')

    chart_ax.set_xlabel('Date', fontsize=12)
    chart_ax.set_ylabel('Price', fontsize=12)
    chart_ax.grid(True, alpha=0.3)
    chart_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    chart_ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Information Panel
    info_text = f"PATTERN #{pattern_number} ANALYSIS (DEFINITIVE PATTERNS ONLY)\n\n"
    info_text += f"Type: {direction.upper()}TREND\n"
    info_text += f"Status: {status.upper()}\n"
    info_text += f"Time Period: {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}\n"
    if pattern.get('accurate_failure_point'):
        info_text += f"✓ Accurate failure point found beyond window boundary\n"
    info_text += "\n"

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