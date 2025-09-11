import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
import json


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


def find_local_extremes(ohlc_data, window_size=5):
    """Find all local highs and lows in OHLC data, ensuring A is highest high or lowest low"""
    highs = ohlc_data['high']
    lows = ohlc_data['low']
    extremes = []

    params = load_parameters()
    min_points_separation = params['pattern_detection']['validation_rules'].get('min_points_separation', 5)
    lookback_window = params['pattern_detection']['validation_rules'].get('lookback_window', window_size * 2)

    for i in range(lookback_window, len(highs) - window_size):
        # Check for local high (potential A for downtrend)
        window_highs = highs[max(0, i - lookback_window):i + window_size + 1]
        if highs[i] == max(window_highs):
            extremes.append((i, highs[i], 'high'))
        # Check for local low (potential A for uptrend)
        window_lows = lows[max(0, i - lookback_window):i + window_size + 1]
        if lows[i] == min(window_lows):
            extremes.append((i, lows[i], 'low'))

    return extremes


def find_all_patterns_ohlc(ohlc_data, dates):
    """
    Find ALL ABCD patterns (up & down) with fixed A and trailing B.
    - A is locked as initial swing extreme
    - B trails to new extremes
    - Each valid retracement (C) is tested for completion (D)
    - After completion/failure, keep A, keep trailing B
    """

    params = load_parameters()
    pattern_config = params['pattern_detection']

    min_change = params['min_change']
    retracement_target = pattern_config['retracement_target']
    retracement_tolerance = pattern_config['retracement_tolerance']
    completion_extension = pattern_config['completion_extension']
    failure_level = pattern_config['failure_level']
    min_points_separation = pattern_config['validation_rules']['min_points_separation']
    use_only_absolute_extremes = pattern_config.get('use_only_absolute_extremes', True)

    patterns = []
    highs, lows = ohlc_data['high'], ohlc_data['low']

    # Step 1: pick A
    if use_only_absolute_extremes:
        potential_A_points = [
            (lows.index(min(lows)), min(lows), 'low'),   # Uptrend
            (highs.index(max(highs)), max(highs), 'high')  # Downtrend
        ]
    # else:
    #     potential_A_points = find_local_extremes(ohlc_data)

    # Step 2: loop over possible A points
    for A_idx, A_price, A_type in potential_A_points:
        direction = 'up' if A_type == 'low' else 'down'

        # Initialize trailing B
        if direction == 'up':
            current_B_price = float('-inf')
            current_B_idx = None
        else:
            current_B_price = float('inf')
            current_B_idx = None

        # Step 3: scan forward from A
        for i in range(A_idx + min_points_separation, len(dates)):

            # Invalidation: if price crosses beyond A, stop
            if direction == 'up' and lows[i] < A_price:
                break
            if direction == 'down' and highs[i] > A_price:
                break

            # Update trailing B
            if direction == 'up' and highs[i] > current_B_price:
                current_B_price = highs[i]
                current_B_idx = i
            elif direction == 'down' and lows[i] < current_B_price:
                current_B_price = lows[i]
                current_B_idx = i

            # Skip if no B yet
            if current_B_idx is None:
                continue

            # Step 4: check AB move big enough
            move_pct = abs(current_B_price - A_price) / abs(A_price)
            if move_pct < min_change:
                continue

            move_AB = abs(current_B_price - A_price)

            # Step 5: look for valid C (local extreme + ~50% retracement)
            left = max(0, i - min_points_separation)
            right = min(len(dates), i + min_points_separation + 1)

            if direction == 'up' and lows[i] == min(lows[left:right]):
                C_price = lows[i]
                target_C = current_B_price - (move_AB * retracement_target)
            elif direction == 'down' and highs[i] == max(highs[left:right]):
                C_price = highs[i]
                target_C = current_B_price + (move_AB * retracement_target)
            else:
                continue

            tolerance = move_AB * retracement_tolerance
            if abs(C_price - target_C) > tolerance:
                continue  # invalid retracement

            # Step 6: search for D (completion or failure)
            D_found = False
            D_price, D_idx, pattern_status = None, None, None
            for k in range(i + min_points_separation, len(dates)):
                if direction == 'up':
                    if lows[k] < current_B_price - (move_AB * failure_level):
                        D_price, D_idx, pattern_status = lows[k], k, "failed"
                        D_found = True
                        break
                    elif highs[k] >= current_B_price + (move_AB * completion_extension):
                        D_price, D_idx, pattern_status = highs[k], k, "completed"
                        D_found = True
                        break
                else:
                    if highs[k] > current_B_price + (move_AB * failure_level):
                        D_price, D_idx, pattern_status = highs[k], k, "failed"
                        D_found = True
                        break
                    elif lows[k] <= current_B_price - (move_AB * completion_extension):
                        D_price, D_idx, pattern_status = lows[k], k, "completed"
                        D_found = True
                        break

            # Step 7: record pattern if D found
            if D_found:
                pattern = {
                    "direction": direction,
                    "A": (dates[A_idx], A_price),
                    "B": (dates[current_B_idx], current_B_price),
                    "C": (dates[i], C_price),
                    "D": (dates[D_idx], D_price),
                    "status": pattern_status,
                    "pattern_type": "progressive",
                    "price_types": {
                        "A": A_type,
                        "B": "high" if direction == "up" else "low",
                        "C": "low" if direction == "up" else "high",
                        "D": ("low" if pattern_status == "failed" else "high") if direction == "up"
                              else ("high" if pattern_status == "failed" else "low")
                    }
                }
                patterns.append(pattern)

                # ✅ Don’t reset A — keep trailing B and continue searching

    return patterns

def analyze_multiple_windows(prices, dates, ohlc_data=None):
    """
    Analyze with multiple window sizes using parameters from JSON
    """
    params = load_parameters()
    pattern_config = params['pattern_detection']

    # Get all parameters from JSON
    window_sizes = params['window_sizes']
    overlap_percent = params['overlap']
    min_change_threshold = params['min_change']

    # Advanced options
    advanced = params.get('advanced_options', {})
    detect_long_failures = pattern_config.get('search_beyond_window_for_failure', True)

    all_patterns = []
    window_sizes = sorted(window_sizes)

    for window_size in window_sizes:
        if window_size >= len(prices):
            print(f"Window size {window_size} is larger than available data. Skipping.")
            continue

        step_size = max(1, int(window_size * (1 - overlap_percent / 100)))

        for start_idx in range(0, len(prices) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Get window data
            window_dates = dates[start_idx:end_idx]

            if ohlc_data:
                # Create OHLC window
                window_ohlc = {
                    'open': ohlc_data['open'][start_idx:end_idx],
                    'high': ohlc_data['high'][start_idx:end_idx],
                    'low': ohlc_data['low'][start_idx:end_idx],
                    'close': ohlc_data['close'][start_idx:end_idx]
                }

                # Find patterns in this window
                window_patterns = find_all_patterns_ohlc(window_ohlc, window_dates)

                # If configured, search beyond window for accurate failure points
                if detect_long_failures and ohlc_data:
                    for pattern in window_patterns:
                        if pattern['status'] == 'failed':
                            pattern = find_accurate_failure_beyond_window(
                                pattern, ohlc_data, dates, start_idx, end_idx
                            )
            else:
                # No OHLC data available
                print("WARNING: No OHLC data available for pattern detection")
                window_patterns = []

            if window_patterns:
                window_info = {
                    "window_size": window_size,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": dates[start_idx],
                    "end_date": dates[end_idx - 1]
                }

                for pattern in window_patterns:
                    pattern_analysis = analyze_single_pattern(pattern)
                    all_patterns.append((pattern, pattern_analysis, window_info))

    # Sort by significance using scoring weights from JSON
    all_patterns.sort(key=lambda x: calculate_pattern_score(x[0]), reverse=True)

    # Limit to top patterns if configured
    top_patterns = params.get('top_patterns', 50)
    if len(all_patterns) > top_patterns:
        all_patterns = all_patterns[:top_patterns]

    print(f"Found {len(all_patterns)} definitive patterns")
    return all_patterns


def find_accurate_failure_beyond_window(pattern, full_ohlc_data, full_dates, window_start, window_end):
    """
    For failed patterns, search beyond window boundary for accurate failure point
    """
    if pattern['status'] != 'failed':
        return pattern

    params = load_parameters()
    pattern_config = params['pattern_detection']
    failure_level = pattern_config['failure_level']

    # Get pattern details
    C_timestamp = pattern['C'][0]
    C_idx_in_window = find_index_from_timestamp(full_dates[window_start:window_end], C_timestamp)
    global_C_idx = window_start + C_idx_in_window

    B_price = pattern['B'][1]
    A_price = pattern['A'][1]
    direction = pattern['direction']

    # Calculate failure level
    move_AB = abs(B_price - A_price)
    if direction == 'up':
        failure_level_price = B_price - (move_AB * failure_level)
    else:
        failure_level_price = B_price + (move_AB * failure_level)

    # Search beyond window for actual failure
    highs = full_ohlc_data['high']
    lows = full_ohlc_data['low']

    for i in range(global_C_idx + 1, len(full_dates)):
        if direction == 'up' and lows[i] < failure_level_price:
            pattern['D'] = (full_dates[i], lows[i])
            pattern['accurate_failure_point'] = True
            break
        elif direction == 'down' and highs[i] > failure_level_price:
            pattern['D'] = (full_dates[i], highs[i])
            pattern['accurate_failure_point'] = True
            break

    return pattern


def calculate_pattern_score(pattern):
    """Calculate pattern score using weights from JSON"""
    params = load_parameters()
    weights = params['pattern_detection']['scoring_weights']
    priority = params['pattern_detection']['pattern_priority']

    score = 0

    # Initial move score
    initial_move = pattern.get('initial_move_pct', 0)
    score += initial_move * weights['initial_move']

    # Retracement quality (closer to 50% is better)
    retracement = pattern.get('retracement_pct', 0)
    retracement_quality = 1.0 - abs(retracement - 50) / 50
    score += retracement_quality * weights['retracement_quality']

    # Status priority
    status = pattern.get('status', 'unknown')
    score += priority.get(status, 0)

    # Absolute extreme bonus
    if pattern.get('price_types', {}).get('A') in ['low', 'high']:
        if pattern['A'][1] == min(pattern['A'][1], pattern['B'][1], pattern['C'][1], pattern['D'][1]):
            score += weights.get('absolute_extreme_bonus', 0)
        elif pattern['A'][1] == max(pattern['A'][1], pattern['B'][1], pattern['C'][1], pattern['D'][1]):
            score += weights.get('absolute_extreme_bonus', 0)

    # Accurate failure bonus
    if pattern.get('accurate_failure_point', False):
        score += weights.get('accurate_failure_bonus', 0)

    return score


def analyze_single_pattern(pattern):
    """Analyze a single pattern"""
    direction = pattern["direction"]
    retracement_pct = pattern.get("retracement_pct", 0)
    initial_move_pct = pattern.get("initial_move_pct", 0)
    status = pattern.get("status", "unknown")
    price_types = pattern.get("price_types", {})

    D_price = pattern["D"][1]

    if status == "failed":
        analysis = f"FAILED {direction}trend pattern - Price broke {retracement_pct:.1f}% level"
        analysis += f"\nInitial Move: {initial_move_pct:.1f}%"
        analysis += f"\nPattern failed at D (${D_price:.2f})"
    elif status == "completed":
        analysis = f"COMPLETED {direction}trend pattern - Target reached"
        analysis += f"\nInitial Move: {initial_move_pct:.1f}%"
        analysis += f"\nPattern completed at D (${D_price:.2f})"
    else:
        analysis = f"Pattern status: {status}"

    if price_types:
        analysis += f"\nOHLC used: A={price_types['A']}, B={price_types.get('B', 'N/A')}, "
        analysis += f"C={price_types.get('C', 'N/A')}, D={price_types.get('D', 'N/A')}"

    return analysis


def load_and_prepare_data(csv_file, date_range=None, index_range=None):
    """
    Load and prepare OHLC data from CSV file
    """
    params = load_parameters()
    data_processing = params.get('data_processing', {})

    df = pd.read_csv(csv_file)

    # Detect timestamp column
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        timestamp_col = df.columns[0]

    # Parse timestamps
    try:
        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='mixed')
    except:
        df['timestamp'] = pd.to_datetime(df[timestamp_col])

    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        subset = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    elif index_range:
        start_idx, end_idx = index_range
        subset = df.iloc[start_idx:end_idx]
    else:
        subset = df

    # Handle gaps if configured
    if data_processing.get('handle_gaps', True):
        subset = subset.sort_values('timestamp')
        if data_processing.get('interpolate_missing', False):
            subset = subset.interpolate(method='linear')

    # Check for OHLC columns
    if all(col in subset.columns for col in ['open', 'high', 'low', 'close']):
        clean_subset = subset.dropna(subset=['open', 'high', 'low', 'close'])

        # Remove outliers if configured
        if data_processing.get('outlier_removal', False):
            threshold = data_processing.get('outlier_threshold', 3.0)
            for col in ['open', 'high', 'low', 'close']:
                z_scores = np.abs((clean_subset[col] - clean_subset[col].mean()) / clean_subset[col].std())
                clean_subset = clean_subset[z_scores < threshold]

        # Check minimum data points
        min_points = data_processing.get('min_data_points', 50)
        if len(clean_subset) < min_points:
            print(f"WARNING: Only {len(clean_subset)} data points (minimum: {min_points})")

        ohlc_data = {
            'open': clean_subset['open'].tolist(),
            'high': clean_subset['high'].tolist(),
            'low': clean_subset['low'].tolist(),
            'close': clean_subset['close'].tolist()
        }
        dates = clean_subset['timestamp'].tolist()
        prices = ohlc_data['close']

        print(f"Loaded {len(clean_subset)} OHLC data points")
        return prices, dates, df, ohlc_data
    else:
        print("ERROR: OHLC columns not found in data")
        return None, None, df, None

def find_patterns_progressive(ohlc_data, dates):
    """
    Progressive ABCD pattern finder:
    Start from A, search for B (up/down), if AB >= min_change, check for C.
    If C invalid, expand horizon and retry.
    """
    params = load_parameters()
    min_change = params['min_change']
    retracement_target = params['pattern_detection']['retracement_target']
    retracement_tolerance = params['pattern_detection']['retracement_tolerance']
    completion_extension = params['pattern_detection']['completion_extension']
    failure_level = params['pattern_detection']['failure_level']

    patterns = []
    highs, lows = ohlc_data['high'], ohlc_data['low']

    for A_idx, A_price, A_type in find_local_extremes(ohlc_data):
        direction = 'up' if A_type == 'low' else 'down'
        best_B_idx = None
        best_B_price = None

        search_idx = A_idx + 1
        while search_idx < len(dates):
            # candidate B is best extreme so far
            if direction == 'up':
                candidate_B_idx = np.argmax(highs[A_idx:search_idx+1]) + A_idx
                candidate_B_price = highs[candidate_B_idx]
            else:
                candidate_B_idx = np.argmin(lows[A_idx:search_idx+1]) + A_idx
                candidate_B_price = lows[candidate_B_idx]

            # check AB move
            move_pct = abs(candidate_B_price - A_price) / abs(A_price)
            if move_pct >= min_change:
                # look for C retracement
                for C_idx in range(candidate_B_idx+1, search_idx+1):
                    if direction == 'up':
                        C_price = lows[C_idx]
                        target_C = candidate_B_price - (abs(candidate_B_price - A_price) * retracement_target)
                    else:
                        C_price = highs[C_idx]
                        target_C = candidate_B_price + (abs(candidate_B_price - A_price) * retracement_target)

                    tolerance = abs(candidate_B_price - A_price) * retracement_tolerance
                    if abs(C_price - target_C) <= tolerance:
                        # we have A, B, C → now D
                        D_idx, D_price, status = None, None, None
                        for k in range(C_idx+1, len(dates)):
                            if direction == 'up':
                                if lows[k] < candidate_B_price - (abs(candidate_B_price - A_price) * failure_level):
                                    D_idx, D_price, status = k, lows[k], "failed"
                                    break
                                elif highs[k] >= candidate_B_price + (abs(candidate_B_price - A_price) * completion_extension):
                                    D_idx, D_price, status = k, highs[k], "completed"
                                    break
                            else:
                                if highs[k] > candidate_B_price + (abs(candidate_B_price - A_price) * failure_level):
                                    D_idx, D_price, status = k, highs[k], "failed"
                                    break
                                elif lows[k] <= candidate_B_price - (abs(candidate_B_price - A_price) * completion_extension):
                                    D_idx, D_price, status = k, lows[k], "completed"
                                    break

                        if D_idx:
                            patterns.append({
                                "A": (dates[A_idx], A_price),
                                "B": (dates[candidate_B_idx], candidate_B_price),
                                "C": (dates[C_idx], C_price),
                                "D": (dates[D_idx], D_price),
                                "direction": direction,
                                "status": status
                            })
                            break  # stop after finding valid C/D
            search_idx += 1
    return patterns
