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


def find_all_e_points_after_d(ohlc_data, dates, d_timestamp, d_price, direction):
    """Find ALL potential E points after D using OHLC data"""
    params = load_parameters()
    min_change = params['min_change']
    # If min_threshold_pct not in JSON, use min_change * 100
    min_threshold_pct = params.get('min_threshold_pct', min_change * 100)

    d_index = find_index_from_timestamp(dates, d_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if d_index >= len(highs) - 1:
        return []

    e_points = []
    min_distance_threshold = d_price * (min_threshold_pct / 100)

    print(f"  Searching for E points after D:")
    print(f"    D point: {d_timestamp}, ${d_price:.2f}")
    print(f"    Min change threshold: {min_change * 100:.1f}%")

    for i in range(d_index + 1, len(highs)):
        current_timestamp = dates[i]

        if direction == 'up':
            # Use HIGHS for uptrend E points
            current_price = highs[i]
            price_change = (current_price - d_price) / d_price if d_price != 0 else 0
            distance_from_d = current_price - d_price

            if price_change >= min_change and distance_from_d >= min_distance_threshold:
                e_points.append((current_timestamp, current_price))

        elif direction == 'down':
            # Use LOWS for downtrend E points
            current_price = lows[i]
            price_change = (d_price - current_price) / d_price if d_price != 0 else 0
            distance_from_d = d_price - current_price

            if price_change >= min_change and distance_from_d >= min_distance_threshold:
                e_points.append((current_timestamp, current_price))

    print(f"  Found {len(e_points)} potential E points")
    return e_points


def check_80_percent_failure(ohlc_data, dates, e_timestamp, e_price, f_price, direction, start_from_timestamp=None):
    """Check if price crosses retracement failure level using OHLC"""
    params = load_parameters()
    failure_percentage = params['pattern_detection']['failure_level']

    if start_from_timestamp:
        start_idx = find_index_from_timestamp(dates, start_from_timestamp)
    else:
        start_idx = find_index_from_timestamp(dates, e_timestamp)

    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if start_idx >= len(highs) - 1:
        return {'failed_80_percent': False, 'reason': 'At end of data'}

    retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction)

    for i in range(start_idx + 1, len(highs)):
        current_timestamp = dates[i]

        if direction == 'up':
            # Check LOWS for failure in uptrend
            if lows[i] <= retracement_80:
                return {
                    'failed_80_percent': True,
                    'failure_point': (current_timestamp, lows[i]),
                    'failure_level': retracement_80,
                    'reason': f'UPTREND: Low went below {failure_percentage * 100:.0f}% level'
                }
        else:
            # Check HIGHS for failure in downtrend
            if highs[i] >= retracement_80:
                return {
                    'failed_80_percent': True,
                    'failure_point': (current_timestamp, highs[i]),
                    'failure_level': retracement_80,
                    'reason': f'DOWNTREND: High went above {failure_percentage * 100:.0f}% level'
                }

    return {
        'failed_80_percent': False,
        'failure_level': retracement_80,
        'reason': f'No excessive retracement beyond {failure_percentage * 100:.0f}% level'
    }


def check_50_percent_bounce_after_e_correct(ohlc_data, dates, e_timestamp, e_price, fib_levels, direction):
    """Check for 50% retracement (S point) using OHLC data"""
    params = load_parameters()
    retracement_target = params['pattern_detection']['retracement_target']
    retracement_tolerance = params['pattern_detection']['retracement_tolerance']

    e_idx = find_index_from_timestamp(dates, e_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if e_idx >= len(highs) - 1:
        return {'valid_50_bounce': False, 'reason': 'E point at end of data', 'failed_80_percent': False}

    f_price = fib_levels['38.2']

    # Calculate S level (50% retracement)
    if direction == 'up':
        s_level = e_price - ((e_price - f_price) * retracement_target)
    else:
        s_level = e_price + ((f_price - e_price) * retracement_target)

    tolerance = s_level * retracement_tolerance

    # Look for S bounce using OHLC
    bounce_found = False
    bounce_point = None

    for i in range(e_idx + 1, len(highs)):
        if direction == 'up':
            # Check if LOW touches S level for uptrend
            if abs(lows[i] - s_level) <= tolerance:
                bounce_timestamp = dates[i]
                bounce_point = (bounce_timestamp, lows[i])
                bounce_found = True
                print(f"    S bounce (LOW) at {bounce_timestamp}, ${lows[i]:.2f}")
                break
        else:
            # Check if HIGH touches S level for downtrend
            if abs(highs[i] - s_level) <= tolerance:
                bounce_timestamp = dates[i]
                bounce_point = (bounce_timestamp, highs[i])
                bounce_found = True
                print(f"    S bounce (HIGH) at {bounce_timestamp}, ${highs[i]:.2f}")
                break

    # Check for 80% failure
    if bounce_found:
        failure_check = check_80_percent_failure(ohlc_data, dates, bounce_point[0], e_price, f_price, direction)
    else:
        failure_check = check_80_percent_failure(ohlc_data, dates, e_timestamp, e_price, f_price, direction)

    if failure_check['failed_80_percent']:
        return {
            'valid_50_bounce': bounce_found,
            'reason': '80% retracement failure',
            'failed_80_percent': True,
            'failure_info': failure_check,
            's_level': s_level,
            'bounce_point': bounce_point if bounce_found else None
        }

    if bounce_found:
        return {
            'valid_50_bounce': True,
            'reason': 'S bounce found',
            'bounce_point': bounce_point,
            's_level': s_level,
            'f_price': f_price,
            'e_price': e_price,
            'fe_move': abs(e_price - f_price),
            'direction': direction,
            'failed_80_percent': False
        }

    return {
        'valid_50_bounce': False,
        'reason': 'No bounce at 50% retracement',
        's_level': s_level,
        'failed_80_percent': False
    }


def check_pattern_completion_236_extension(ohlc_data, dates, bounce_timestamp, e_price, f_price, direction):
    """Check for pattern completion at -23.6% extension using OHLC"""
    params = load_parameters()
    completion_extension = params['pattern_detection']['completion_extension']

    bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if bounce_idx >= len(highs) - 1:
        return {'pattern_completed': False, 'reason': 'S bounce at end of data', 'failed_80_percent': False}

    # Calculate extension target
    if direction == 'up':
        fe_move_distance = e_price - f_price
        extension_236 = e_price + (fe_move_distance * completion_extension)
    else:
        fe_move_distance = f_price - e_price
        extension_236 = e_price - (fe_move_distance * completion_extension)

    retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction)

    for i in range(bounce_idx + 1, len(highs)):
        current_timestamp = dates[i]

        # Check 80% failure first
        if direction == 'up' and lows[i] <= retracement_80:
            return {
                'pattern_completed': False,
                'target_price': extension_236,
                'failed_80_percent': True,
                'failure_point': (current_timestamp, lows[i]),
                'failure_level': retracement_80
            }
        elif direction == 'down' and highs[i] >= retracement_80:
            return {
                'pattern_completed': False,
                'target_price': extension_236,
                'failed_80_percent': True,
                'failure_point': (current_timestamp, highs[i]),
                'failure_level': retracement_80
            }

        # Check for target completion
        if direction == 'up' and highs[i] >= extension_236:
            return {
                'pattern_completed': True,
                'completion_point': (current_timestamp, highs[i]),
                'target_price': extension_236,
                'actual_price': highs[i],
                'failed_80_percent': False
            }
        elif direction == 'down' and lows[i] <= extension_236:
            return {
                'pattern_completed': True,
                'completion_point': (current_timestamp, lows[i]),
                'target_price': extension_236,
                'actual_price': lows[i],
                'failed_80_percent': False
            }

    return {
        'pattern_completed': False,
        'target_price': extension_236,
        'reason': 'Target not reached yet',
        'failed_80_percent': False
    }


def find_valid_e_with_trailing_strategy_complete(ohlc_data, dates, d_timestamp, d_price, fib_levels, direction):
    """Enhanced trailing strategy with OHLC data"""
    all_e_points = find_all_e_points_after_d(ohlc_data, dates, d_timestamp, d_price, direction)

    if not all_e_points:
        return {
            'e_point': None,
            'validation': {'valid_50_bounce': False, 'reason': 'No E points found'},
            'is_valid': False,
            'pattern_completed': False,
            'failed_80_percent': False,
            'trailing_attempts': []
        }

    highs = ohlc_data['high']
    lows = ohlc_data['low']
    trailing_attempts = []

    for i, (initial_e_timestamp, initial_e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']

        print(f"\n--- Trailing Attempt {i + 1}/{len(all_e_points)} ---")
        print(f"Testing E: {initial_e_timestamp}, ${initial_e_price:.2f}")

        validation = check_50_percent_bounce_after_e_correct(
            ohlc_data, dates, initial_e_timestamp, initial_e_price, fib_levels, direction
        )

        if validation.get('failed_80_percent', False):
            print(f"    ❌ Pattern failed: 80% retracement violation")
            trailing_attempts.append({
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'validation': validation,
                'found_bounce': validation['valid_50_bounce'],
                'failed_80_percent': True
            })
            return {
                'e_point': (initial_e_timestamp, initial_e_price),
                'validation': validation,
                'is_valid': False,
                'pattern_completed': False,
                'failed_80_percent': True,
                'trailing_attempts': trailing_attempts
            }

        if validation['valid_50_bounce']:
            print(f"    ✅ S bounce found! Finding {direction.upper()} extreme before bounce...")

            bounce_timestamp = validation['bounce_point'][0]
            bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
            d_idx = find_index_from_timestamp(dates, d_timestamp)

            # Find extreme BEFORE bounce using OHLC
            if direction == 'up':
                # Find highest HIGH before bounce
                search_highs = highs[d_idx:bounce_idx]
                if search_highs:
                    max_price = max(search_highs)
                    max_idx = d_idx + search_highs.index(max_price)
                    final_e_timestamp = dates[max_idx]
                    final_e_price = max_price
                else:
                    final_e_timestamp = initial_e_timestamp
                    final_e_price = initial_e_price
            else:
                # Find lowest LOW before bounce
                search_lows = lows[d_idx:bounce_idx]
                if search_lows:
                    min_price = min(search_lows)
                    min_idx = d_idx + search_lows.index(min_price)
                    final_e_timestamp = dates[min_idx]
                    final_e_price = min_price
                else:
                    final_e_timestamp = initial_e_timestamp
                    final_e_price = initial_e_price

            # Re-validate with final E
            final_validation = check_50_percent_bounce_after_e_correct(
                ohlc_data, dates, final_e_timestamp, final_e_price, fib_levels, direction
            )

            if not final_validation['valid_50_bounce']:
                print(f"    ❌ No valid S bounce after final E")
                trailing_attempts.append({
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'validation': final_validation,
                    'found_bounce': False
                })
                continue

            # Check completion
            final_bounce_timestamp = final_validation['bounce_point'][0]
            completion_result = check_pattern_completion_236_extension(
                ohlc_data, dates, final_bounce_timestamp, final_e_price, f_price, direction
            )

            if completion_result.get('failed_80_percent', False):
                print(f"    ❌ Pattern failed during target approach")
                trailing_attempts.append({
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'validation': final_validation,
                    'completion': completion_result,
                    'failed_80_percent': True
                })
                return {
                    'e_point': (final_e_timestamp, final_e_price),
                    'validation': final_validation,
                    'completion': completion_result,
                    'is_valid': False,
                    'pattern_completed': False,
                    'failed_80_percent': True,
                    'trailing_attempts': trailing_attempts
                }

            print(f"    ✅ Valid pattern found!")
            trailing_attempts.append({
                'candidate_number': i + 1,
                'e_point': (final_e_timestamp, final_e_price),
                'validation': final_validation,
                'completion': completion_result,
                'found_bounce': True,
                'pattern_completed': completion_result['pattern_completed']
            })

            return {
                'e_point': (final_e_timestamp, final_e_price),
                'validation': final_validation,
                'completion': completion_result,
                'is_valid': True,
                'pattern_completed': completion_result['pattern_completed'],
                'failed_80_percent': False,
                's_level': final_validation['s_level'],
                'trailing_attempts': trailing_attempts,
                'successful_attempt': i + 1
            }
        else:
            print(f"    ❌ No S bounce found")
            trailing_attempts.append({
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'validation': validation,
                'found_bounce': False
            })

    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': {'valid_50_bounce': False, 'reason': 'No valid S bounce found'},
        'is_valid': False,
        'pattern_completed': False,
        'failed_80_percent': False,
        'trailing_attempts': trailing_attempts
    }


def analyze_fan_extension_with_completion(pattern, ohlc_data, dates):
    """Main fan extension analysis with OHLC data"""
    params = load_parameters()

    fib_levels = calculate_fibonacci_levels(pattern)
    d_timestamp = pattern['D'][0]
    d_price = pattern['D'][1]
    direction = pattern.get('direction', 'unknown')

    print(f"\n{'=' * 70}")
    print(f"ANALYZING {direction.upper()} PATTERN WITH OHLC DATA")
    print(f"  D point: {d_timestamp}, ${d_price:.2f}")
    print(f"  F point (38.2%): ${fib_levels['38.2']:.2f}")

    # Use OHLC-aware trailing strategy
    e_result = find_valid_e_with_trailing_strategy_complete(
        ohlc_data, dates, d_timestamp, d_price, fib_levels, direction
    )

    # Build analysis results
    analysis = {
        'pattern': pattern,
        'fibonacci_levels': fib_levels,
        'f_level': fib_levels['38.2'],
        'd_timestamp': d_timestamp,
        'd_price': d_price,
        'pattern_direction': direction,
        'is_valid': False,
        'pattern_completed': False,
        'failed_80_percent': False,
        'strategy': 'trailing_ohlc'
    }

    if e_result and e_result.get('failed_80_percent', False):
        analysis['failed_80_percent'] = True
        analysis['is_valid'] = False
        analysis['pattern_completed'] = False
        if e_result.get('e_point'):
            analysis['e_point'] = e_result['e_point']
            if direction == 'up':
                analysis['highest_after_d'] = e_result['e_point']
            else:
                analysis['lowest_after_d'] = e_result['e_point']
        analysis['validation_info'] = e_result.get('validation', {})
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

    elif e_result and e_result['is_valid']:
        e_point = e_result['e_point']
        analysis['e_point'] = e_point
        if direction == 'up':
            analysis['highest_after_d'] = e_point
        else:
            analysis['lowest_after_d'] = e_point

        analysis['validation_info'] = e_result['validation']
        analysis['completion_info'] = e_result.get('completion', {})
        analysis['is_valid'] = True
        analysis['pattern_completed'] = e_result.get('pattern_completed', False)
        analysis['s_level'] = e_result.get('s_level')
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

        # Calculate completion target
        e_price = e_point[1]
        f_price = fib_levels['38.2']
        if direction == 'up':
            fe_move = e_price - f_price
            analysis['completion_target'] = e_price + (fe_move * params['pattern_detection']['completion_extension'])
        else:
            fe_move = f_price - e_price
            analysis['completion_target'] = e_price - (fe_move * params['pattern_detection']['completion_extension'])

        analysis['fe_move'] = fe_move
        analysis['retracement_80_level'] = calculate_80_percent_retracement(f_price, e_price, direction)

    else:
        analysis['validation_info'] = e_result.get('validation', {})
        analysis['is_valid'] = False
        analysis['pattern_completed'] = False
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

    return analysis


def plot_pattern_with_ohlc(analysis, ohlc_data, dates):
    """Plot pattern with OHLC candlesticks"""
    params = load_parameters()
    output_settings = params.get('output_settings', {})

    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

    fig, ax = plt.subplots(figsize=(22, 12))

    # Get display range
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)

    min_idx = min(pattern_indices)
    max_idx = max(pattern_indices)

    # Include E point if exists
    if analysis.get('e_point'):
        e_timestamp = analysis['e_point'][0]
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_idx = max(max_idx, e_idx)

    # Add padding
    padding = 100
    start_idx = max(0, min_idx - padding)
    end_idx = min(len(dates) - 1, max_idx + padding)

    # Plot OHLC candlesticks
    for i in range(start_idx, end_idx):
        date = dates[i]
        open_price = ohlc_data['open'][i]
        high_price = ohlc_data['high'][i]
        low_price = ohlc_data['low'][i]
        close_price = ohlc_data['close'][i]

        # Determine color
        color = '#00AA00' if close_price >= open_price else '#AA0000'

        # Draw high-low line
        ax.plot([date, date], [low_price, high_price],
                color='black', linewidth=0.5, alpha=0.7, zorder=1)

        # Draw body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)

        if i < len(dates) - 1:
            width = (mdates.date2num(dates[i + 1]) - mdates.date2num(date)) * 0.6
        else:
            width = 0.0004

        rect = Rectangle((mdates.date2num(date) - width / 2, body_bottom),
                         width, body_height,
                         facecolor=color, edgecolor='black',
                         alpha=0.8, linewidth=0.3, zorder=2)
        ax.add_patch(rect)

    # Plot pattern points
    points = ['A', 'B', 'C', 'D']
    point_colors = {'A': '#2F2F2F', 'B': '#D62728', 'C': '#FF7F0E', 'D': '#1f77b4'}

    for point in points:
        timestamp, price = pattern[point]
        ax.plot(timestamp, price, 'o', color=point_colors[point],
                markersize=12, zorder=10,
                markeredgecolor='white', markeredgewidth=2)
        ax.text(timestamp, price, point, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=15)

    # F level
    if analysis.get('f_level'):
        f_price = analysis['f_level']
        ax.axhline(y=f_price, color='#2CA02C', linestyle='--',
                   linewidth=3, alpha=0.8, zorder=3)
        ax.plot(pattern['D'][0], f_price, 's', color='#2CA02C',
                markersize=16, zorder=15,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(pattern['D'][0], f_price, 'F', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#2CA02C', zorder=16)

    # E point
    if analysis.get('e_point'):
        e_timestamp, e_price = analysis['e_point']
        ax.plot(e_timestamp, e_price, 'D', color='#228B22',
                markersize=18, zorder=15,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(e_timestamp, e_price, 'E', ha='center', va='top',
                fontsize=12, fontweight='bold', color='#228B22', zorder=16)

    # S level
    if analysis.get('s_level'):
        s_level = analysis['s_level']
        ax.axhline(y=s_level, color='#9467BD', linewidth=4,
                   alpha=0.9, zorder=3)

    # Format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)

    # Title
    if analysis.get('is_valid'):
        if analysis.get('pattern_completed'):
            title = f'COMPLETED {direction.upper()} Pattern (OHLC)'
        else:
            title = f'VALID {direction.upper()} Pattern (OHLC)'
    else:
        title = f'INVALID {direction.upper()} Pattern (OHLC)'

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_settings.get('save_plots', False):
        filename = f'pattern_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=output_settings.get('plot_dpi', 300))

    plt.show()


def get_completed_patterns_for_date(date_str):
    """Load OHLC data for a specific date and analyze patterns"""
    params = load_parameters()

    print(f"\n{'=' * 70}")
    print(f"FAN EXTENSION ANALYSIS WITH OHLC DATA")
    print(f"Date: {date_str}")

    # Find data file
    possible_files = [
        f"C:/Users/admin/Desktop/btc_1minute_data/btc_1minute_data_1minute_{date_str}.csv",
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
        print(f"Data file not found")
        return {"status": "file_not_found", "date": date_str}

    # Load OHLC data
    prices, dates, df, ohlc_data = load_and_prepare_data(data_file)

    if ohlc_data is None:
        print("ERROR: No OHLC data found in file")
        return {"status": "no_ohlc_data", "date": date_str}

    print(f"Loaded {len(prices)} OHLC data points")

    # Get patterns using analyze_multiple_windows
    # all_patterns = analyze_multiple_windows(prices, dates, ohlc_data)
    params = load_parameters()
    use_progressive = params["pattern_detection"].get("use_progressive_search", True)

    if use_progressive:
        progressive_patterns = find_patterns_progressive(ohlc_data, dates)
        all_patterns = [(p, "progressive search", {"method": "progressive"}) for p in progressive_patterns]
        print(f"✅ Using progressive search: {len(progressive_patterns)} patterns found")
    else:
        all_patterns = analyze_multiple_windows(prices, dates, ohlc_data)
        print(f"✅ Using window-based search: {len(all_patterns)} patterns found")

    # Filter for completed patterns
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

    # Analyze fan extensions with OHLC
    extension_analyses = []
    successful_patterns = 0
    failed_patterns = 0

    for i, pattern in enumerate(completed_patterns):
        print(f"\n{'=' * 50}")
        print(f"PATTERN {i + 1}/{len(completed_patterns)}")
        print(f"Direction: {pattern.get('direction', 'unknown')}")

        # Analyze with OHLC data
        extension_analysis = analyze_fan_extension_with_completion(pattern, ohlc_data, dates)
        extension_analyses.append(extension_analysis)

        if extension_analysis['is_valid']:
            successful_patterns += 1
            print(f"✅ Pattern {i + 1}: VALID")
            if extension_analysis.get('pattern_completed'):
                print(f"    Status: COMPLETED")
            else:
                print(f"    Status: VALID (awaiting completion)")
        else:
            failed_patterns += 1
            if extension_analysis.get('failed_80_percent'):
                print(f"❌ Pattern {i + 1}: FAILED (80% retracement)")
            else:
                print(f"❌ Pattern {i + 1}: INVALID (no S bounce)")

        # Show E point info
        if extension_analysis.get('e_point'):
            e_timestamp, e_price = extension_analysis['e_point']
            print(f"    E point: {e_timestamp} at ${e_price:.2f}")

        # Show trailing attempts
        attempts = extension_analysis.get('trailing_attempts', [])
        if attempts:
            print(f"    Tested {len(attempts)} E candidates")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Total completed ABCD patterns: {len(completed_patterns)}")
    print(f"Valid fan extensions: {successful_patterns}")
    print(f"Failed/Invalid extensions: {failed_patterns}")

    if len(completed_patterns) > 0:
        success_rate = (successful_patterns / len(completed_patterns)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    return {
        "status": "success",
        "date": date_str,
        "completed_patterns": completed_patterns,
        "extension_analyses": extension_analyses,
        "ohlc_data": ohlc_data,
        "dates": dates,
        "total_patterns": len(all_patterns),
        "completed_count": len(completed_patterns),
        "valid_extensions": successful_patterns,
        "failed_extensions": failed_patterns
    }


if __name__ == "__main__":
    params = load_parameters()
    date_to_analyze = "2025-09-01"
    show_plots = params.get('output_settings', {}).get('show_plots', True)

    print(f"🚀 Starting OHLC fan extension analysis for {date_to_analyze}")
    print(f"Using parameters from parameters.json")

    results = get_completed_patterns_for_date(date_to_analyze)

    if results["status"] == "success":
        print(f"\n✅ ANALYSIS COMPLETED!")
        print(f"Found {results['completed_count']} completed ABCD patterns")
        print(f"Valid fan extensions: {results['valid_extensions']}")
        print(f"Failed extensions: {results['failed_extensions']}")

        if show_plots and results['extension_analyses']:
            print("\n📈 Generating OHLC plots...")
            for i, analysis in enumerate(results['extension_analyses']):
                if analysis.get('e_point'):
                    print(f"\nPlotting pattern {i + 1}...")
                    plot_pattern_with_ohlc(analysis, results['ohlc_data'], results['dates'])
    else:
        print(f"❌ Error: {results['status']}")