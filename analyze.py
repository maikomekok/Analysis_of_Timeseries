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


def find_local_extremes(ohlc_data, window_size=2):
    """
    CORRECTED VERSION - Properly detect local extremes without missing obvious peaks/troughs
    """
    highs = np.array(ohlc_data['high'])
    lows = np.array(ohlc_data['low'])
    extremes = []

    for i in range(window_size, len(highs) - window_size):
        # Local high: higher than ALL points in symmetric window
        left_highs = highs[i - window_size:i] if i >= window_size else []
        right_highs = highs[i + 1:i + window_size + 1] if i < len(highs) - window_size else []

        if (len(left_highs) == 0 or highs[i] > np.max(left_highs)) and \
                (len(right_highs) == 0 or highs[i] > np.max(right_highs)):
            extremes.append((i, highs[i], 'high'))

        # Local low: lower than ALL points in symmetric window
        left_lows = lows[i - window_size:i] if i >= window_size else []
        right_lows = lows[i + 1:i + window_size + 1] if i < len(lows) - window_size else []

        if (len(left_lows) == 0 or lows[i] < np.min(left_lows)) and \
                (len(right_lows) == 0 or lows[i] < np.min(right_lows)):
            extremes.append((i, lows[i], 'low'))

    return extremes


def find_all_patterns_ohlc(ohlc_data, dates):
    """
    Find ABCD patterns with your exact logic:
    A=lowest, B=highest, C=50% retrace, D=-23.6% extension
    """
    params = load_parameters()
    min_change = params['min_change']
    retracement_target = 0.5  # Exactly 50%
    retracement_tolerance = params['pattern_detection']['retracement_tolerance']
    completion_extension = params['pattern_detection']['completion_extension']  # Use configurable value
    failure_level = params['pattern_detection']['failure_level']
    min_points_separation = params['pattern_detection']['validation_rules']['min_points_separation']
    allow_incomplete = params['pattern_detection'].get('allow_incomplete_patterns', True)

    patterns = []
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    print(f"=== ABCD PATTERN DETECTION (YOUR EXACT LOGIC) ===")
    print(f"min_change: {min_change * 100:.3f}%")
    print(f"retracement_target: {retracement_target * 100:.0f}%")
    print(f"completion_extension: -{completion_extension * 100:.1f}%")
    print(f"allow_incomplete: {allow_incomplete}")

    # Find all local extremes
    extremes = find_local_extremes(ohlc_data, window_size=2)
    print(f"Found {len(extremes)} local extremes")

    # Look for UPTREND patterns: A(low) -> B(high) -> C(low, 50%) -> D(high, -23.6%)
    for i, (A_idx, A_price, A_type) in enumerate(extremes):
        if A_type != 'low':
            continue

        print(f"\n--- Testing A point #{i + 1}: {dates[A_idx]} ${A_price:.2f} (low) ---")

        # Find all potential B points (highs) after A
        potential_B_points = []
        for j, (B_idx, B_price, B_type) in enumerate(extremes[i + 1:], i + 1):
            if B_type == 'high' and B_idx > A_idx + min_points_separation:
                ab_move = B_price - A_price
                ab_move_pct = ab_move / A_price
                if ab_move_pct >= min_change:
                    potential_B_points.append((j, B_idx, B_price, ab_move, ab_move_pct))

        print(f"  Found {len(potential_B_points)} potential B points after A")

        # Test each B point (starting from highest)
        potential_B_points.sort(key=lambda x: x[2], reverse=True)  # Sort by price (highest first)

        for b_order, (b_ext_idx, B_idx, B_price, ab_move, ab_move_pct) in enumerate(potential_B_points):
            print(f"\n  Testing B #{b_order + 1}: {dates[B_idx]} ${B_price:.2f} (AB move: {ab_move_pct * 100:.3f}%)")

            # Calculate 50% retracement level
            target_C_price = B_price - (ab_move * retracement_target)  # 50% back toward A
            tolerance = ab_move * retracement_tolerance

            print(f"    Target C level: ${target_C_price:.2f} (±${tolerance:.2f})")

            # Look for C point (low) after B that hits 50% retracement
            C_found = False
            for k, (C_idx, C_price, C_type) in enumerate(extremes[b_ext_idx + 1:], b_ext_idx + 1):
                if C_type == 'low' and C_idx > B_idx + min_points_separation:

                    # Check if C is within tolerance of 50% retracement
                    if abs(C_price - target_C_price) <= tolerance:
                        retracement_pct = (B_price - C_price) / ab_move

                        print(
                            f"    ✓ Valid C found: {dates[C_idx]} ${C_price:.2f} (retrace: {retracement_pct * 100:.1f}%)")

                        # Look for D target (-23.6% extension beyond B)
                        target_D_price = B_price + (ab_move * completion_extension)
                        failure_price = B_price - (ab_move * failure_level)

                        print(f"    Searching for D - Target: ${target_D_price:.2f}, Failure: ${failure_price:.2f}")

                        # Search for D in remaining price data
                        D_found = False
                        for m in range(C_idx + min_points_separation, len(highs)):
                            # Check for failure first
                            if lows[m] <= failure_price:
                                print(f"    ✗ Pattern FAILED at {dates[m]} ${lows[m]:.2f}")
                                patterns.append({
                                    "direction": "up",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (dates[m], lows[m]),
                                    "status": "failed",
                                    "pattern_type": "abcd",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "price_types": {"A": "low", "B": "high", "C": "low", "D": "low"}
                                })
                                D_found = True
                                break

                            # Check for completion
                            elif highs[m] >= target_D_price:
                                print(f"    ✓ Pattern COMPLETED at {dates[m]} ${highs[m]:.2f}")
                                patterns.append({
                                    "direction": "up",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (dates[m], highs[m]),
                                    "status": "completed",
                                    "pattern_type": "abcd",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "price_types": {"A": "low", "B": "high", "C": "low", "D": "high"}
                                })
                                D_found = True
                                break

                        if D_found:
                            C_found = True
                            break  # Found valid ABCD pattern with this B
                        else:
                            print(f"    ~ Pattern pending (no D completion yet)")

                            # If incomplete patterns are allowed, add as pending
                            if allow_incomplete:
                                patterns.append({
                                    "direction": "down",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (None, None),  # No D yet
                                    "status": "pending",
                                    "pattern_type": "abcd_incomplete",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "target_D_price": target_D_price,
                                    "failure_price": failure_price,
                                    "price_types": {"A": "high", "B": "low", "C": "high", "D": "pending"}
                                })
                                print(f"    + Added as PENDING pattern (ABC complete, waiting for D)")
                                C_found = True
                                break

                            # If incomplete patterns are allowed, add as pending
                            if allow_incomplete:
                                patterns.append({
                                    "direction": "up",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (None, None),  # No D yet
                                    "status": "pending",
                                    "pattern_type": "abcd_incomplete",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "target_D_price": target_D_price,
                                    "failure_price": failure_price,
                                    "price_types": {"A": "low", "B": "high", "C": "low", "D": "pending"}
                                })
                                print(f"    + Added as PENDING pattern (ABC complete, waiting for D)")
                                C_found = True
                                break

            if C_found:
                break  # Found valid C with this B, don't try higher B points
            else:
                print(f"    ✗ No valid C found for this B, trying next highest B")

    # Look for DOWNTREND patterns: A(high) -> B(low) -> C(high, 50%) -> D(low, -23.6%)
    for i, (A_idx, A_price, A_type) in enumerate(extremes):
        if A_type != 'high':
            continue

        print(f"\n--- Testing A point #{i + 1}: {dates[A_idx]} ${A_price:.2f} (high) ---")

        # Find all potential B points (lows) after A
        potential_B_points = []
        for j, (B_idx, B_price, B_type) in enumerate(extremes[i + 1:], i + 1):
            if B_type == 'low' and B_idx > A_idx + min_points_separation:
                ab_move = A_price - B_price
                ab_move_pct = ab_move / A_price
                if ab_move_pct >= min_change:
                    potential_B_points.append((j, B_idx, B_price, ab_move, ab_move_pct))

        print(f"  Found {len(potential_B_points)} potential B points after A")

        # Test each B point (starting from lowest)
        potential_B_points.sort(key=lambda x: x[2])  # Sort by price (lowest first)

        for b_order, (b_ext_idx, B_idx, B_price, ab_move, ab_move_pct) in enumerate(potential_B_points):
            print(f"\n  Testing B #{b_order + 1}: {dates[B_idx]} ${B_price:.2f} (AB move: {ab_move_pct * 100:.3f}%)")

            # Calculate 50% retracement level
            target_C_price = B_price + (ab_move * retracement_target)  # 50% back toward A
            tolerance = ab_move * retracement_tolerance

            print(f"    Target C level: ${target_C_price:.2f} (±${tolerance:.2f})")

            # Look for C point (high) after B that hits 50% retracement
            C_found = False
            for k, (C_idx, C_price, C_type) in enumerate(extremes[b_ext_idx + 1:], b_ext_idx + 1):
                if C_type == 'high' and C_idx > B_idx + min_points_separation:

                    # Check if C is within tolerance of 50% retracement
                    if abs(C_price - target_C_price) <= tolerance:
                        retracement_pct = (C_price - B_price) / ab_move

                        print(
                            f"    ✓ Valid C found: {dates[C_idx]} ${C_price:.2f} (retrace: {retracement_pct * 100:.1f}%)")

                        # Look for D target (-23.6% extension beyond B)
                        target_D_price = B_price - (ab_move * completion_extension)
                        failure_price = B_price + (ab_move * failure_level)

                        print(f"    Searching for D - Target: ${target_D_price:.2f}, Failure: ${failure_price:.2f}")

                        # Search for D in remaining price data
                        D_found = False
                        for m in range(C_idx + min_points_separation, len(lows)):
                            # Check for failure first
                            if highs[m] >= failure_price:
                                print(f"    ✗ Pattern FAILED at {dates[m]} ${highs[m]:.2f}")
                                patterns.append({
                                    "direction": "down",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (dates[m], highs[m]),
                                    "status": "failed",
                                    "pattern_type": "abcd",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "price_types": {"A": "high", "B": "low", "C": "high", "D": "high"}
                                })
                                D_found = True
                                break

                            # Check for completion
                            elif lows[m] <= target_D_price:
                                print(f"    ✓ Pattern COMPLETED at {dates[m]} ${lows[m]:.2f}")
                                patterns.append({
                                    "direction": "down",
                                    "A": (dates[A_idx], A_price),
                                    "B": (dates[B_idx], B_price),
                                    "C": (dates[C_idx], C_price),
                                    "D": (dates[m], lows[m]),
                                    "status": "completed",
                                    "pattern_type": "abcd",
                                    "retracement_pct": retracement_pct * 100,
                                    "initial_move_pct": ab_move_pct * 100,
                                    "price_types": {"A": "high", "B": "low", "C": "high", "D": "low"}
                                })
                                D_found = True
                                break

                        if D_found:
                            C_found = True
                            break  # Found valid ABCD pattern with this B
                        else:
                            print(f"    ~ Pattern pending (no D completion yet)")

            if C_found:
                break  # Found valid C with this B, don't try lower B points
            else:
                print(f"    ✗ No valid C found for this B, trying next lowest B")

    print(f"\n=== SUMMARY ===")
    print(f"Total ABCD patterns found: {len(patterns)}")
    completed = [p for p in patterns if p['status'] == 'completed']
    failed = [p for p in patterns if p['status'] == 'failed']
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")

    return patterns


# Compatibility functions to maintain your existing imports
def find_correction_patterns(ohlc_data, dates):
    """Compatibility function"""
    return find_all_patterns_ohlc(ohlc_data, dates)


def find_patterns_from_local_extremes(ohlc_data, dates):
    """Compatibility function"""
    return find_all_patterns_ohlc(ohlc_data, dates)


def analyze_multiple_windows(prices, dates, ohlc_data=None):
    """Simplified version for compatibility"""
    if ohlc_data:
        patterns = find_all_patterns_ohlc(ohlc_data, dates)
        return [(p, "", {}) for p in patterns]  # Return in expected format
    return []


def load_and_prepare_data(csv_file, date_range=None, index_range=None):
    """Load and prepare OHLC data from CSV file"""
    df = pd.read_csv(csv_file)

    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        timestamp_col = df.columns[0]

    try:
        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='mixed')
    except:
        df['timestamp'] = pd.to_datetime(df[timestamp_col])

    if date_range:
        start_date, end_date = date_range
        subset = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    elif index_range:
        start_idx, end_idx = index_range
        subset = df.iloc[start_idx:end_idx]
    else:
        subset = df

    subset = subset.sort_values('timestamp')

    if all(col in subset.columns for col in ['open', 'high', 'low', 'close']):
        clean_subset = subset.dropna(subset=['open', 'high', 'low', 'close'])

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
    """Compatibility function"""
    return find_all_patterns_ohlc(ohlc_data, dates)


def test_your_specific_pattern(csv_file):
    """Test function to verify pattern detection works"""
    print("=== TESTING PATTERN DETECTION ===")

    prices, dates, df, ohlc_data = load_and_prepare_data(csv_file)

    if ohlc_data is None:
        print("ERROR: Could not load data")
        return []

    print(f"Data loaded: {len(prices)} points")

    # Test extremes detection
    extremes = find_local_extremes(ohlc_data, window_size=2)
    print(f"Extremes found: {len(extremes)}")

    # Test pattern detection
    patterns = find_all_patterns_ohlc(ohlc_data, dates)

    return patterns


