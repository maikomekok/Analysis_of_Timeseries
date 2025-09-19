import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import json
import os
from analyze import (
    load_and_prepare_data,
    analyze_multiple_windows,
    find_patterns_progressive
)
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle


def load_parameters():
    """Load ALL parameters from parameters.json"""
    with open('parameters.json', 'r') as f:
        params = json.load(f)
    return params


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


def calculate_fibonacci_levels(pattern):
    """Calculate key Fibonacci levels for the pattern"""
    a_price = pattern['A'][1]
    b_price = pattern['B'][1]
    direction = pattern.get('direction', 'unknown')

    if direction == 'up':
        move_size = b_price - a_price
        levels = {
            '38.2': b_price - (move_size * 0.382),
            '50.0': b_price - (move_size * 0.5),
            '61.8': b_price - (move_size * 0.618)
        }
    else:
        move_size = a_price - b_price
        levels = {
            '38.2': b_price + (move_size * 0.382),
            '50.0': b_price + (move_size * 0.5),
            '61.8': b_price + (move_size * 0.618)
        }

    print(f"  Fibonacci levels: 38.2%=${levels['38.2']:.2f}, 50%=${levels['50.0']:.2f}, 61.8%=${levels['61.8']:.2f}")
    return levels


def calculate_80_percent_retracement(f_price, e_price, direction):
    """Calculate the failure retracement level between F and E points"""
    params = load_parameters()
    failure_percentage = params['pattern_detection']['failure_level']

    if direction == 'up':
        fe_move = e_price - f_price
        retracement_80 = e_price - (fe_move * failure_percentage)
    else:
        fe_move = f_price - e_price
        retracement_80 = e_price + (fe_move * failure_percentage)

    return retracement_80


def find_local_extremes_after_d(ohlc_data, dates, d_timestamp, direction, window_size=10):
    """Find all local extremes after D point"""
    d_index = find_index_from_timestamp(dates, d_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    extremes = []

    # Start searching from D+1
    for i in range(d_index + window_size, len(dates) - window_size):
        if direction == 'up':
            # Look for local highs in uptrend
            if highs[i] == max(highs[i - window_size:i + window_size + 1]):
                extremes.append((dates[i], highs[i], 'high'))
        else:
            # Look for local lows in downtrend
            if lows[i] == min(lows[i - window_size:i + window_size + 1]):
                extremes.append((dates[i], lows[i], 'low'))

    return extremes


def check_s_bounce(ohlc_data, dates, e_timestamp, e_price, f_price, direction):
    """Check for 50% retracement bounce (S point) after E"""
    params = load_parameters()
    retracement_target = params['pattern_detection']['retracement_target']
    retracement_tolerance = params['pattern_detection']['retracement_tolerance']

    e_idx = find_index_from_timestamp(dates, e_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if e_idx >= len(highs) - 1:
        return None

    # Calculate S level (50% retracement of FE move)
    if direction == 'up':
        fe_move = e_price - f_price
        s_level = e_price - (fe_move * retracement_target)
    else:
        fe_move = f_price - e_price
        s_level = e_price + (fe_move * retracement_target)

    tolerance = abs(fe_move) * retracement_tolerance

    # Search for S bounce
    for i in range(e_idx + 1, len(dates)):
        if direction == 'up':
            # Check if low touches S level
            if abs(lows[i] - s_level) <= tolerance:
                return {
                    'timestamp': dates[i],
                    'price': lows[i],
                    's_level': s_level,
                    'index': i
                }
        else:
            # Check if high touches S level
            if abs(highs[i] - s_level) <= tolerance:
                return {
                    'timestamp': dates[i],
                    'price': highs[i],
                    's_level': s_level,
                    'index': i
                }

    return None


def check_pattern_completion(ohlc_data, dates, s_bounce_idx, e_price, f_price, direction):
    """Check if pattern completes at -23.6% extension after S bounce"""
    params = load_parameters()
    completion_extension = params['pattern_detection']['completion_extension']
    failure_level = params['pattern_detection']['failure_level']

    highs = ohlc_data['high']
    lows = ohlc_data['low']

    # Calculate target and failure levels
    if direction == 'up':
        fe_move = e_price - f_price
        target_level = e_price + (fe_move * completion_extension)
        failure_level_price = e_price - (fe_move * failure_level)
    else:
        fe_move = f_price - e_price
        target_level = e_price - (fe_move * completion_extension)
        failure_level_price = e_price + (fe_move * failure_level)

    # Search for completion or failure
    for i in range(s_bounce_idx + 1, len(dates)):
        # Check failure first
        if direction == 'up':
            if lows[i] <= failure_level_price:
                return {
                    'status': 'failed',
                    'timestamp': dates[i],
                    'price': lows[i],
                    'target_level': target_level,
                    'failure_level': failure_level_price,
                    's_failure_point': (dates[i], failure_level_price)  # S point at failure level
                }
            if highs[i] >= target_level:
                return {
                    'status': 'completed',
                    'timestamp': dates[i],
                    'price': highs[i],
                    'target_level': target_level,
                    'failure_level': failure_level_price
                }
        else:
            if highs[i] >= failure_level_price:
                return {
                    'status': 'failed',
                    'timestamp': dates[i],
                    'price': highs[i],
                    'target_level': target_level,
                    'failure_level': failure_level_price,
                    's_failure_point': (dates[i], failure_level_price)  # S point at failure level
                }
            if lows[i] <= target_level:
                return {
                    'status': 'completed',
                    'timestamp': dates[i],
                    'price': lows[i],
                    'target_level': target_level,
                    'failure_level': failure_level_price
                }

    return {
        'status': 'pending',
        'target_level': target_level,
        'failure_level': failure_level_price
    }


def find_fan_extension_pattern(pattern, ohlc_data, dates):
    """Main function to find fan extension pattern with trailing E points"""
    fib_levels = calculate_fibonacci_levels(pattern)
    d_timestamp = pattern['D'][0]
    d_price = pattern['D'][1]
    direction = pattern.get('direction', 'unknown')
    f_price = fib_levels['38.2']

    print(f"\n{'=' * 70}")
    print(f"SEARCHING FOR FAN EXTENSION - {direction.upper()} PATTERN")
    print(f"  D: {d_timestamp}, ${d_price:.2f}")
    print(f"  F (38.2%): ${f_price:.2f}")

    # Get D index
    d_index = find_index_from_timestamp(dates, d_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    # Find all local extremes after D
    extremes = find_local_extremes_after_d(ohlc_data, dates, d_timestamp, direction)

    if not extremes:
        print("  No local extremes found after D")
        return None

    print(f"  Found {len(extremes)} potential E points")

    # Try each extreme as E point
    for idx, (e_candidate_timestamp, e_candidate_price, e_type) in enumerate(extremes):
        print(f"\n  Testing E candidate #{idx + 1}: {e_candidate_timestamp}, ${e_candidate_price:.2f}")

        # Check for S bounce using this candidate
        s_bounce = check_s_bounce(ohlc_data, dates, e_candidate_timestamp, e_candidate_price, f_price, direction)

        if s_bounce:
            print(f"    ✓ S bounce found at {s_bounce['timestamp']}, ${s_bounce['price']:.2f}")

            # NOW find the actual extreme between D and S
            s_index = s_bounce['index']

            if direction == 'up':
                # Find the highest high between D and S
                search_highs = highs[d_index:s_index + 1]
                actual_e_price = max(search_highs)
                actual_e_idx = d_index + search_highs.index(actual_e_price)
            else:
                # Find the lowest low between D and S
                search_lows = lows[d_index:s_index + 1]
                actual_e_price = min(search_lows)
                actual_e_idx = d_index + search_lows.index(actual_e_price)

            actual_e_timestamp = dates[actual_e_idx]

            print(f"    📍 Actual E point (absolute extreme): {actual_e_timestamp}, ${actual_e_price:.2f}")

            # Re-calculate S level based on actual E
            if direction == 'up':
                fe_move = actual_e_price - f_price
                actual_s_level = actual_e_price - (fe_move * 0.5)
            else:
                fe_move = f_price - actual_e_price
                actual_s_level = actual_e_price + (fe_move * 0.5)

            # Update S bounce info with recalculated level
            s_bounce['s_level'] = actual_s_level

            # Check for pattern completion using actual E
            completion = check_pattern_completion(
                ohlc_data, dates, s_bounce['index'],
                actual_e_price, f_price, direction
            )

            if completion['status'] == 'completed':
                print(f"    ✓ Pattern COMPLETED at {completion['timestamp']}, ${completion['price']:.2f}")
            elif completion['status'] == 'failed':
                print(f"    ✗ Pattern FAILED at {completion['timestamp']}, ${completion['price']:.2f}")
            else:
                print(f"    ~ Pattern pending (target: ${completion['target_level']:.2f})")

            # Return the successful pattern with actual E
            return {
                'pattern': pattern,
                'fibonacci_levels': fib_levels,
                'f_level': f_price,
                'e_point': (actual_e_timestamp, actual_e_price),
                'e_candidate': (e_candidate_timestamp, e_candidate_price),  # Keep track of original candidate
                's_bounce': s_bounce,
                'completion': completion,
                'direction': direction,
                'is_valid': True,
                'attempts': idx + 1
            }
        else:
            print(f"    ✗ No S bounce found")

    print("\n  No valid fan extension found")
    return None


def plot_fan_extension_pattern(analysis, ohlc_data, dates):
    """Enhanced plotting with all pattern points labeled consistently"""
    pattern = analysis['pattern']
    direction = analysis['direction']

    fig, ax = plt.subplots(figsize=(24, 14))

    # Determine display range
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp = pattern[point][0]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)

    # Include E point
    if analysis.get('e_point'):
        e_idx = find_index_from_timestamp(dates, analysis['e_point'][0])
        pattern_indices.append(e_idx)

    # Include completion point if exists
    if analysis.get('completion') and analysis['completion'].get('timestamp'):
        comp_idx = find_index_from_timestamp(dates, analysis['completion']['timestamp'])
        pattern_indices.append(comp_idx)

    min_idx = min(pattern_indices)
    max_idx = max(pattern_indices)

    # Add padding
    padding = 50
    start_idx = max(0, min_idx - padding)
    end_idx = min(len(dates) - 1, max_idx + padding)

    # Plot candlesticks
    for i in range(start_idx, end_idx):
        date = dates[i]
        open_price = ohlc_data['open'][i]
        high_price = ohlc_data['high'][i]
        low_price = ohlc_data['low'][i]
        close_price = ohlc_data['close'][i]

        color = '#00CC00' if close_price >= open_price else '#CC0000'

        # High-low line
        ax.plot([date, date], [low_price, high_price],
                color='black', linewidth=0.5, alpha=0.6, zorder=1)

        # Body rectangle
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)

        if i < len(dates) - 1:
            width = (mdates.date2num(dates[i + 1]) - mdates.date2num(date)) * 0.7
        else:
            width = 0.0005

        rect = Rectangle((mdates.date2num(date) - width / 2, body_bottom),
                         width, body_height,
                         facecolor=color, edgecolor='black',
                         alpha=0.7, linewidth=0.3, zorder=2)
        ax.add_patch(rect)

    # Plot ABCD pattern points with consistent style
    point_style = {
        'A': {'color': '#2F2F2F', 'marker': 'o', 'size': 14},
        'B': {'color': '#D62728', 'marker': 'o', 'size': 14},
        'C': {'color': '#FF7F0E', 'marker': 'o', 'size': 14},
        'D': {'color': '#1F77B4', 'marker': 'o', 'size': 14}
    }

    # Plot and label ABCD points
    for point_name in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point_name]
        style = point_style[point_name]

        # Plot point with white border
        ax.plot(timestamp, price, style['marker'],
                color=style['color'], markersize=style['size'],
                markeredgecolor='white', markeredgewidth=2.5, zorder=10)

        # Add label
        ax.text(timestamp, price, point_name,
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=11)

    # Plot F point (38.2% retracement)
    f_price = analysis['f_level']
    f_timestamp = pattern['D'][0]  # F is at same time as D

    ax.plot(f_timestamp, f_price, 's',
            color='#2CA02C', markersize=14,
            markeredgecolor='white', markeredgewidth=2.5, zorder=10)
    ax.text(f_timestamp, f_price, 'F',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=11)

    # Draw horizontal line from F
    ax.axhline(y=f_price, color='#2CA02C', linestyle='--',
               linewidth=2, alpha=0.6, label='F Level (38.2%)')

    # Plot E point
    if analysis.get('e_point'):
        e_timestamp, e_price = analysis['e_point']

        ax.plot(e_timestamp, e_price, 'D',
                color='#9467BD', markersize=14,
                markeredgecolor='white', markeredgewidth=2.5, zorder=10)
        ax.text(e_timestamp, e_price, 'E',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=11)

        # Draw line from F to E
        ax.plot([f_timestamp, e_timestamp], [f_price, e_price],
                'b-', linewidth=2, alpha=0.7, label='F-E Extension')

    # Plot S bounce point or S failure point
    if analysis.get('s_bounce'):
        s_bounce = analysis['s_bounce']
        s_timestamp = s_bounce['timestamp']
        s_price = s_bounce['price']
        s_level = s_bounce['s_level']

        # Check if pattern failed and we have an S failure point
        if analysis.get('completion') and analysis['completion'].get('s_failure_point'):
            # Plot S at the failure level (80% retracement)
            s_fail_timestamp, s_fail_price = analysis['completion']['s_failure_point']

            # Plot S at failure level
            ax.plot(s_fail_timestamp, s_fail_price, '^',
                    color='#FF4444', markersize=14,
                    markeredgecolor='white', markeredgewidth=2.5, zorder=10)
            ax.text(s_fail_timestamp, s_fail_price, 'S',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=11)

            # Still show the original S bounce attempt
            ax.plot(s_timestamp, s_price, '^',
                    color='#17BECF', markersize=10, alpha=0.5,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=9)

            # Draw both S levels
            ax.axhline(y=s_level, color='#17BECF', linestyle=':',
                       linewidth=1.5, alpha=0.4, label='S Level attempted (50% FE)')
            ax.axhline(y=analysis['completion']['failure_level'], color='#FF4444', linestyle=':',
                       linewidth=2, alpha=0.8, label='S Level failed (80% FE)')
        else:
            # Normal S bounce (pattern didn't fail at 80%)
            ax.plot(s_timestamp, s_price, '^',
                    color='#17BECF', markersize=14,
                    markeredgecolor='white', markeredgewidth=2.5, zorder=10)
            ax.text(s_timestamp, s_price, 'S',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=11)

            # Draw S level line
            ax.axhline(y=s_level, color='#17BECF', linestyle=':',
                       linewidth=2, alpha=0.6, label='S Level (50% FE)')

    # Plot completion/failure point
    if analysis.get('completion'):
        completion = analysis['completion']
        if completion.get('timestamp'):
            comp_timestamp = completion['timestamp']
            comp_price = completion['price']
            comp_status = completion['status']

            if comp_status == 'completed':
                marker_color = '#00FF00'
                label_text = 'T'  # Target reached
            else:
                marker_color = '#FF0000'
                label_text = 'X'  # Failed

            ax.plot(comp_timestamp, comp_price, 'v',
                    color=marker_color, markersize=14,
                    markeredgecolor='white', markeredgewidth=2.5, zorder=10)
            ax.text(comp_timestamp, comp_price, label_text,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=11)

        # Draw target level
        if 'target_level' in completion:
            ax.axhline(y=completion['target_level'], color='green',
                       linestyle='--', linewidth=2, alpha=0.6,
                       label='Target (-23.6% ext)')

        # Draw failure level
        if 'failure_level' in completion:
            ax.axhline(y=completion['failure_level'], color='red',
                       linestyle='--', linewidth=2, alpha=0.6,
                       label='Failure (80% retr)')

    # Connect pattern lines
    # ABCD lines
    for i in range(len(['A', 'B', 'C', 'D']) - 1):
        point1 = ['A', 'B', 'C', 'D'][i]
        point2 = ['A', 'B', 'C', 'D'][i + 1]
        t1, p1 = pattern[point1]
        t2, p2 = pattern[point2]
        ax.plot([t1, t2], [p1, p2], 'gray', linewidth=1.5, alpha=0.5)

    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)

    # Title based on status
    if analysis.get('completion'):
        status = analysis['completion']['status']
        if status == 'completed':
            title = f'✓ COMPLETED Fan Extension - {direction.upper()} Pattern'
            ax.set_title(title, fontsize=14, fontweight='bold', color='green')
        elif status == 'failed':
            title = f'✗ FAILED Fan Extension - {direction.upper()} Pattern'
            ax.set_title(title, fontsize=14, fontweight='bold', color='red')
        else:
            title = f'~ PENDING Fan Extension - {direction.upper()} Pattern'
            ax.set_title(title, fontsize=14, fontweight='bold', color='orange')
    else:
        title = f'Fan Extension Analysis - {direction.upper()} Pattern'
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(loc='best', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()

    return fig


def analyze_patterns_for_date(date_str):
    """Main function to analyze patterns for a given date"""
    params = load_parameters()

    print(f"\n{'=' * 70}")
    print(f"FAN EXTENSION PATTERN ANALYSIS")
    print(f"Date: {date_str}")
    print(f"{'=' * 70}")

    # Find data file
    possible_files = [
        f"btc_1minute_data_{date_str}.csv",
        f"./btc_1minute_data_{date_str}.csv",
        f"C:/Users/admin/Desktop/btc_1minute_data/btc_1minute_data_1minute_{date_str}.csv"
    ]

    data_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            data_file = file_path
            print(f"Found data file: {data_file}")
            break

    if not data_file:
        print(f"ERROR: Data file not found for {date_str}")
        return None

    # Load OHLC data
    prices, dates, df, ohlc_data = load_and_prepare_data(data_file)

    if ohlc_data is None:
        print("ERROR: No OHLC data found")
        return None

    print(f"Loaded {len(prices)} data points")

    # Find base ABCD patterns
    use_progressive = params["pattern_detection"].get("use_progressive_search", True)

    if use_progressive:
        patterns = find_patterns_progressive(ohlc_data, dates)
        print(f"Found {len(patterns)} patterns using progressive search")
    else:
        all_patterns = analyze_multiple_windows(prices, dates, ohlc_data)
        patterns = [p[0] for p in all_patterns]
        print(f"Found {len(patterns)} patterns using window search")

    # Filter for completed patterns
    completed_patterns = [p for p in patterns if p.get('status') == 'completed']
    print(f"Completed ABCD patterns: {len(completed_patterns)}")

    # Analyze fan extensions
    fan_extensions = []
    successful_count = 0
    failed_count = 0
    pending_count = 0

    for i, pattern in enumerate(completed_patterns):
        print(f"\n{'=' * 50}")
        print(f"PATTERN {i + 1}/{len(completed_patterns)}")

        result = find_fan_extension_pattern(pattern, ohlc_data, dates)

        if result:
            fan_extensions.append(result)

            if result['completion']['status'] == 'completed':
                successful_count += 1
                print(f"✓ Fan extension COMPLETED")
            elif result['completion']['status'] == 'failed':
                failed_count += 1
                print(f"✗ Fan extension FAILED")
            else:
                pending_count += 1
                print(f"~ Fan extension PENDING")
        else:
            print(f"No valid fan extension found")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY RESULTS")
    print(f"{'=' * 70}")
    print(f"Total ABCD patterns: {len(patterns)}")
    print(f"Completed ABCD: {len(completed_patterns)}")
    print(f"Valid fan extensions: {len(fan_extensions)}")
    print(f"  - Completed: {successful_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Pending: {pending_count}")

    if len(fan_extensions) > 0:
        success_rate = (successful_count / len(fan_extensions)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    # Plot results
    if params.get('output_settings', {}).get('show_plots', True):
        for extension in fan_extensions:
            plot_fan_extension_pattern(extension, ohlc_data, dates)

    return {
        'date': date_str,
        'patterns': patterns,
        'completed_patterns': completed_patterns,
        'fan_extensions': fan_extensions,
        'stats': {
            'total_patterns': len(patterns),
            'completed_abcd': len(completed_patterns),
            'valid_extensions': len(fan_extensions),
            'successful': successful_count,
            'failed': failed_count,
            'pending': pending_count
        }
    }


if __name__ == "__main__":
    date_to_analyze = "2025-09-01"

    results = analyze_patterns_for_date(date_to_analyze)

    if results:
        print(f"\n✓ Analysis complete!")
        print(f"Found {results['stats']['valid_extensions']} valid fan extension patterns")
    else:
        print(f"\n✗ Analysis failed")