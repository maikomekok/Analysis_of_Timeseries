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


def find_local_extremes(ohlc_data, window_size=100):
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


def find_patterns_from_progressive_lows(ohlc_data, dates, min_change, retracement_target,
                                        retracement_tolerance, completion_extension,
                                        failure_level, min_points_separation, seen_patterns):
    """
    Find uptrend patterns where A is a low.
    A updates dynamically when a new low is found.
    """
    patterns = []
    highs, lows = ohlc_data['high'], ohlc_data['low']
    direction = 'up'

    # Track the current lowest point as A
    current_A_idx = 0
    current_A_price = lows[0]

    # Scan through the data
    i = 0
    while i < len(dates):
        # Check if we have a new low (new A point)
        if lows[i] < current_A_price:
            # New absolute low found - this becomes the new A
            current_A_idx = i
            current_A_price = lows[i]

            # Reset and start pattern search from this new A
            # Continue scanning from next point
            i += 1
            continue

        # Only look for patterns starting from current A
        if i >= current_A_idx + min_points_separation:
            # Find the highest point between A and current position (trailing B)
            if i > current_A_idx:
                B_idx = current_A_idx + 1 + np.argmax(highs[current_A_idx + 1:i + 1])
                B_price = highs[B_idx]

                # Check if AB move is significant
                move_pct = abs(B_price - current_A_price) / abs(current_A_price)
                if move_pct >= min_change:
                    move_AB = abs(B_price - current_A_price)

                    # Look for C retracement at current position
                    left = max(0, i - min_points_separation)
                    right = min(len(dates), i + min_points_separation + 1)

                    # Check if current position is a local low (potential C)
                    if lows[i] == min(lows[left:right]):
                        C_price = lows[i]
                        C_idx = i
                        target_C = B_price - (move_AB * retracement_target)
                        tolerance = move_AB * retracement_tolerance

                        if abs(C_price - target_C) <= tolerance:
                            # Valid C found, look for D
                            D_found = False
                            D_price, D_idx, pattern_status = None, None, None

                            for k in range(C_idx + min_points_separation, len(dates)):
                                # Check if price goes below A (invalidates pattern)
                                if lows[k] < current_A_price:
                                    # This will become a new A in the next iteration
                                    break

                                # Check for failure or completion
                                if lows[k] < B_price - (move_AB * failure_level):
                                    D_price, D_idx, pattern_status = lows[k], k, "failed"
                                    D_found = True
                                    break
                                elif highs[k] >= B_price + (move_AB * completion_extension):
                                    D_price, D_idx, pattern_status = highs[k], k, "completed"
                                    D_found = True
                                    break

                            if D_found:
                                # Create unique key for deduplication
                                pattern_key = (direction, C_idx, D_idx, pattern_status)

                                if pattern_key not in seen_patterns:
                                    seen_patterns.add(pattern_key)

                                    pattern = {
                                        "direction": direction,
                                        "A": (dates[current_A_idx], current_A_price),
                                        "B": (dates[B_idx], B_price),
                                        "C": (dates[C_idx], C_price),
                                        "D": (dates[D_idx], D_price),
                                        "status": pattern_status,
                                        "pattern_type": "progressive",
                                        "retracement_pct": abs(C_price - B_price) / move_AB * 100,
                                        "initial_move_pct": move_pct * 100,
                                        "price_types": {
                                            "A": "low",
                                            "B": "high",
                                            "C": "low",
                                            "D": "low" if pattern_status == "failed" else "high"
                                        }
                                    }
                                    patterns.append(pattern)

        i += 1

    return patterns


def find_patterns_from_progressive_highs(ohlc_data, dates, min_change, retracement_target,
                                         retracement_tolerance, completion_extension,
                                         failure_level, min_points_separation, seen_patterns):
    """
    Find downtrend patterns where A is a high.
    A updates dynamically when a new high is found.
    """
    patterns = []
    highs, lows = ohlc_data['high'], ohlc_data['low']
    direction = 'down'

    # Track the current highest point as A
    current_A_idx = 0
    current_A_price = highs[0]

    # Scan through the data
    i = 0
    while i < len(dates):
        # Check if we have a new high (new A point)
        if highs[i] > current_A_price:
            # New absolute high found - this becomes the new A
            current_A_idx = i
            current_A_price = highs[i]

            # Reset and start pattern search from this new A
            i += 1
            continue

        # Only look for patterns starting from current A
        if i >= current_A_idx + min_points_separation:
            # Find the lowest point between A and current position (trailing B)
            if i > current_A_idx:
                B_idx = current_A_idx + 1 + np.argmin(lows[current_A_idx + 1:i + 1])
                B_price = lows[B_idx]

                # Check if AB move is significant
                move_pct = abs(current_A_price - B_price) / abs(current_A_price)
                if move_pct >= min_change:
                    move_AB = abs(current_A_price - B_price)

                    # Look for C retracement at current position
                    left = max(0, i - min_points_separation)
                    right = min(len(dates), i + min_points_separation + 1)

                    # Check if current position is a local high (potential C)
                    if highs[i] == max(highs[left:right]):
                        C_price = highs[i]
                        C_idx = i
                        target_C = B_price + (move_AB * retracement_target)
                        tolerance = move_AB * retracement_tolerance

                        if abs(C_price - target_C) <= tolerance:
                            # Valid C found, look for D
                            D_found = False
                            D_price, D_idx, pattern_status = None, None, None

                            for k in range(C_idx + min_points_separation, len(dates)):
                                # Check if price goes above A (invalidates pattern)
                                if highs[k] > current_A_price:
                                    # This will become a new A in the next iteration
                                    break

                                # Check for failure or completion
                                if highs[k] > B_price + (move_AB * failure_level):
                                    D_price, D_idx, pattern_status = highs[k], k, "failed"
                                    D_found = True
                                    break
                                elif lows[k] <= B_price - (move_AB * completion_extension):
                                    D_price, D_idx, pattern_status = lows[k], k, "completed"
                                    D_found = True
                                    break

                            if D_found:
                                # Create unique key for deduplication
                                pattern_key = (direction, C_idx, D_idx, pattern_status)

                                if pattern_key not in seen_patterns:
                                    seen_patterns.add(pattern_key)

                                    pattern = {
                                        "direction": direction,
                                        "A": (dates[current_A_idx], current_A_price),
                                        "B": (dates[B_idx], B_price),
                                        "C": (dates[C_idx], C_price),
                                        "D": (dates[D_idx], D_price),
                                        "status": pattern_status,
                                        "pattern_type": "progressive",
                                        "retracement_pct": abs(C_price - B_price) / move_AB * 100,
                                        "initial_move_pct": move_pct * 100,
                                        "price_types": {
                                            "A": "high",
                                            "B": "low",
                                            "C": "high",
                                            "D": "high" if pattern_status == "failed" else "low"
                                        }
                                    }
                                    patterns.append(pattern)

        i += 1

    return patterns


def find_all_patterns_ohlc(ohlc_data, dates):
    """
    Find ALL ABCD patterns with DYNAMIC A point detection.
    When price makes a new extreme, that becomes the new A point.
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

    # Track unique patterns
    seen_patterns = set()

    if use_only_absolute_extremes:
        # For uptrend patterns (A is a low)
        patterns.extend(find_patterns_from_progressive_lows(
            ohlc_data, dates, min_change, retracement_target,
            retracement_tolerance, completion_extension, failure_level,
            min_points_separation, seen_patterns
        ))

        # For downtrend patterns (A is a high)
        patterns.extend(find_patterns_from_progressive_highs(
            ohlc_data, dates, min_change, retracement_target,
            retracement_tolerance, completion_extension, failure_level,
            min_points_separation, seen_patterns
        ))

    return patterns


def deduplicate_patterns(patterns, tolerance_seconds=60):
    """
    Additional deduplication function to remove patterns that are nearly identical.
    Patterns are considered duplicates if they have the same direction, status,
    and their C and D points are within tolerance_seconds of each other.
    """
    if not patterns:
        return patterns

    unique_patterns = []

    for pattern in patterns:
        is_duplicate = False

        for unique in unique_patterns:
            # Check if same direction and status
            if (pattern['direction'] == unique['direction'] and
                    pattern['status'] == unique['status']):

                # Get timestamps
                pattern_c_time = pd.to_datetime(pattern['C'][0])
                pattern_d_time = pd.to_datetime(pattern['D'][0])
                unique_c_time = pd.to_datetime(unique['C'][0])
                unique_d_time = pd.to_datetime(unique['D'][0])

                # Check if C and D points are within tolerance
                c_diff = abs((pattern_c_time - unique_c_time).total_seconds())
                d_diff = abs((pattern_d_time - unique_d_time).total_seconds())

                if c_diff <= tolerance_seconds and d_diff <= tolerance_seconds:
                    # Check price similarity (within 0.1%)
                    c_price_diff = abs(pattern['C'][1] - unique['C'][1]) / unique['C'][1]
                    d_price_diff = abs(pattern['D'][1] - unique['D'][1]) / unique['D'][1]

                    if c_price_diff < 0.001 and d_price_diff < 0.001:
                        is_duplicate = True
                        break

        if not is_duplicate:
            unique_patterns.append(pattern)

    return unique_patterns
def analyze_multiple_windows(prices, dates, ohlc_data=None):
    """
    Analyze with multiple window sizes using parameters from JSON.
    Now includes deduplication of patterns.
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

    # Track all raw patterns before deduplication
    raw_pattern_count = 0

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
                raw_pattern_count += len(window_patterns)

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

    # Deduplicate patterns across all windows
    if all_patterns:
        patterns_only = [p[0] for p in all_patterns]
        unique_patterns = deduplicate_patterns(patterns_only)

        # Rebuild the all_patterns list with only unique patterns
        unique_all_patterns = []
        for pattern in unique_patterns:
            # Find the corresponding analysis and window info
            for p, analysis, window_info in all_patterns:
                if (p['C'] == pattern['C'] and p['D'] == pattern['D'] and
                    p['direction'] == pattern['direction'] and p['status'] == pattern['status']):
                    unique_all_patterns.append((pattern, analysis, window_info))
                    break

        all_patterns = unique_all_patterns

        print(f"Deduplication: {raw_pattern_count} raw patterns → {len(all_patterns)} unique patterns")

    # Sort by significance using scoring weights from JSON
    all_patterns.sort(key=lambda x: calculate_pattern_score(x[0]), reverse=True)

    # Limit to top patterns if configured
    top_patterns = params.get('top_patterns', 50)
    if len(all_patterns) > top_patterns:
        all_patterns = all_patterns[:top_patterns]

    print(f"Found {len(all_patterns)} definitive patterns")
    return all_patterns


def validate_pattern_integrity(pattern, ohlc_data, dates):
    """
    Additional validation to ensure pattern integrity.
    Checks that price doesn't violate critical levels during pattern formation.
    """
    A_time, A_price = pattern['A']
    B_time, B_price = pattern['B']
    C_time, C_price = pattern['C']
    D_time, D_price = pattern['D']
    direction = pattern['direction']

    # Get indices
    A_idx = find_index_from_timestamp(dates, A_time)
    B_idx = find_index_from_timestamp(dates, B_time)
    C_idx = find_index_from_timestamp(dates, C_time)
    D_idx = find_index_from_timestamp(dates, D_time)

    highs = ohlc_data['high']
    lows = ohlc_data['low']

    # Check that price never violates A during the entire pattern
    for i in range(A_idx + 1, D_idx + 1):
        if direction == 'up':
            if lows[i] < A_price:
                return False, f"Price violated A level at index {i}"
        else:
            if highs[i] > A_price:
                return False, f"Price violated A level at index {i}"

    # Check that C doesn't go beyond B
    if direction == 'up':
        if C_price > B_price:
            return False, "C retracement went above B"
    else:
        if C_price < B_price:
            return False, "C retracement went below B"

    # Check proper sequence
    if not (A_idx < B_idx < C_idx < D_idx):
        return False, "Pattern points not in proper chronological order"

    return True, "Pattern valid"


def find_patterns_progressive(ohlc_data, dates):
    """
    Progressive ABCD pattern finder with dynamic A point updates.
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

    if use_only_absolute_extremes:
        # Use the new dynamic A detection
        return find_all_patterns_ohlc(ohlc_data, dates)
    else:
        # Use local extremes (your existing code for this case)
        patterns = []
        highs, lows = ohlc_data['high'], ohlc_data['low']

        for A_idx, A_price, A_type in find_local_extremes(ohlc_data):
            # ... existing local extremes code ...
            pass

        return patterns
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
