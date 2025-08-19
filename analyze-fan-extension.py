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
import matplotlib.patheffects as pe


def find_all_e_points_after_d(prices, dates, d_timestamp, d_price, min_change, direction, min_threshold_pct=1.0):
    """
    Find ALL potential E points after D (not just the first one)
    UPDATED: Ensures E points are at least min_threshold distance away from D

    Args:
        prices: Price array
        dates: Dates array
        d_timestamp: D point timestamp
        d_price: D point price
        min_change: Minimum change threshold for E point qualification
        direction: 'up' or 'down'
        min_threshold_pct: Minimum distance from D point (percentage)

    Returns:
        list: All E points found, sorted by timestamp
    """
    d_index = find_index_from_timestamp(dates, d_timestamp)
    if d_index >= len(prices) - 1:
        return []

    e_points = []
    min_distance_threshold = d_price * (min_threshold_pct / 100)

    print(f"  Searching for E points after D:")
    print(f"    D point: {d_timestamp}, ${d_price:.2f}")
    print(f"    Min change threshold: {min_change * 100:.1f}%")
    print(f"    Min distance from D: {min_threshold_pct}% = ${min_distance_threshold:.2f}")

    for i in range(d_index + 1, len(prices)):
        current_price = prices[i]
        current_timestamp = dates[i]

        if direction == 'up':
            # Look for highs
            price_change = (current_price - d_price) / d_price
            distance_from_d = current_price - d_price

            # Must meet BOTH criteria: min_change AND min_threshold distance
            if price_change >= min_change and distance_from_d >= min_distance_threshold:
                e_points.append((current_timestamp, current_price))
                print(f"    ✓ Valid E candidate: {current_timestamp}, ${current_price:.2f}")
                print(f"      Change: {price_change * 100:.2f}%, Distance: ${distance_from_d:.2f}")
            elif price_change >= min_change:
                print(f"    ✗ E candidate too close to D: {current_timestamp}, ${current_price:.2f}")
                print(
                    f"      Change: {price_change * 100:.2f}% ✓, Distance: ${distance_from_d:.2f} ✗ (need ${min_distance_threshold:.2f})")

        elif direction == 'down':
            # Look for lows
            price_change = (d_price - current_price) / d_price
            distance_from_d = d_price - current_price

            # Must meet BOTH criteria: min_change AND min_threshold distance
            if price_change >= min_change and distance_from_d >= min_distance_threshold:
                e_points.append((current_timestamp, current_price))
                print(f"    ✓ Valid E candidate: {current_timestamp}, ${current_price:.2f}")
                print(f"      Change: {price_change * 100:.2f}%, Distance: ${distance_from_d:.2f}")
            elif price_change >= min_change:
                print(f"    ✗ E candidate too close to D: {current_timestamp}, ${current_price:.2f}")
                print(
                    f"      Change: {price_change * 100:.2f}% ✓, Distance: ${distance_from_d:.2f} ✗ (need ${min_distance_threshold:.2f})")

    print(f"  Found {len(e_points)} potential E points after D (meeting both change and distance criteria)")
    return e_points


def find_valid_e_with_50_bounce_correct(prices, dates, d_timestamp, d_price, fib_levels, direction, min_change,
                                        min_threshold_pct):
    """
    Find the first E point that has a valid S bounce (50% of F-E move)
    UPDATED: Uses the new distance-checking E point finder
    """
    # Get all potential E points with minimum distance requirement
    all_e_points = find_all_e_points_after_d(prices, dates, d_timestamp, d_price, min_change, direction, min_threshold_pct)

    if not all_e_points:
        print(f"  No potential E points found meeting distance requirement")
        return None

    print(f"\n  Searching through {len(all_e_points)} potential E points for valid S bounce (50% of F-E)...")

    # Test each E point until we find one with valid S bounce
    for i, (e_timestamp, e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']
        fe_move = e_price - f_price
        s_level = f_price + (fe_move * 0.5)

        print(f"\n  Testing E candidate {i + 1}/{len(all_e_points)}: {e_timestamp}, ${e_price:.2f}")
        print(f"    Distance from D: ${abs(e_price - d_price):.2f}")
        print(f"    F-E move: ${fe_move:.2f}, S level: ${s_level:.2f}")

        # Check if this E point has valid S bounce
        validation = check_50_percent_bounce_after_e_correct(
            prices, dates, e_timestamp, e_price, fib_levels, direction, min_threshold_pct
        )

        if validation['valid_50_bounce']:
            print(f"    ✓ FOUND VALID E POINT with S bounce: {e_timestamp}, ${e_price:.2f}")
            print(f"      Distance from D: ${abs(e_price - d_price):.2f} (meets min requirement)")
            return {
                'e_point': (e_timestamp, e_price),
                'validation': validation,
                'candidate_number': i + 1,
                'total_candidates_tested': i + 1,
                'is_valid': True,
                'distance_from_d': abs(e_price - d_price)
            }
        else:
            print(f"    ✗ E candidate {i + 1} invalid: {validation['reason']}")

    # No valid E point found
    print(f"\n  ✗ NO VALID E POINT with S bounce found after testing {len(all_e_points)} candidates")

    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': validation if 'validation' in locals() else {'valid_50_bounce': False,
                                                                   'reason': 'No E points found meeting distance requirement'},
        'candidate_number': len(all_e_points),
        'total_candidates_tested': len(all_e_points),
        'is_valid': False,
        'distance_from_d': abs(all_e_points[-1][1] - d_price) if all_e_points else 0
    }
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


def calculate_fibonacci_levels(pattern):
    """
    Calculate key Fibonacci levels for the pattern

    Returns:
        dict: Dictionary with 38.2%, 50%, 61.8% levels
    """
    a_price = pattern['A'][1]
    b_price = pattern['B'][1]
    direction = pattern.get('direction', 'unknown')

    if direction == 'up':
        # UPTREND: A is low, B is high
        move_size = b_price - a_price
        levels = {
            '38.2': b_price - (move_size * 0.382),
            '50.0': b_price - (move_size * 0.5),
            '61.8': b_price - (move_size * 0.618)
        }
        print(f"  UPTREND: A=${a_price:.2f}, B=${b_price:.2f}, Move=${move_size:.2f}")
    elif direction == 'down':
        # DOWNTREND: A is high, B is low
        move_size = a_price - b_price
        levels = {
            '38.2': b_price + (move_size * 0.382),
            '50.0': b_price + (move_size * 0.5),
            '61.8': b_price + (move_size * 0.618)
        }
        print(f"  DOWNTREND: A=${a_price:.2f}, B=${b_price:.2f}, Move=${move_size:.2f}")
    else:
        # Auto-detect direction
        if b_price > a_price:
            move_size = b_price - a_price
            levels = {
                '38.2': b_price - (move_size * 0.382),
                '50.0': b_price - (move_size * 0.5),
                '61.8': b_price - (move_size * 0.618)
            }
            print(f"  AUTO-DETECTED UPTREND: A=${a_price:.2f}, B=${b_price:.2f}")
        else:
            move_size = a_price - b_price
            levels = {
                '38.2': b_price + (move_size * 0.382),
                '50.0': b_price + (move_size * 0.5),
                '61.8': b_price + (move_size * 0.618)
            }
            print(f"  AUTO-DETECTED DOWNTREND: A=${a_price:.2f}, B=${b_price:.2f}")

    print(f"  Fibonacci levels: 38.2%=${levels['38.2']:.2f}, 50%=${levels['50.0']:.2f}, 61.8%=${levels['61.8']:.2f}")
    return levels


def calculate_382_retracement_level(pattern):
    """
    Calculate the 38.2% retracement level of the AB move
    """
    fib_levels = calculate_fibonacci_levels(pattern)
    return fib_levels['38.2']


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


def check_50_percent_bounce_after_e_correct(prices, dates, e_timestamp, e_price, fib_levels, direction,
                                            min_threshold_pct):
    """
    Check if price bounces on 50% level after E point
    UPDATED: Now verifies ACTUAL BOUNCE BEHAVIOR, not just price touch

    Args:
        prices: Price array
        dates: Dates array
        e_timestamp: Timestamp of point E (new high/low)
        e_price: Price at point E
        fib_levels: Fibonacci levels dictionary (contains F = 38.2% level)
        direction: 'up' or 'down'
        min_threshold_pct: Minimum threshold from parameters

    Returns:
        dict: Validation result
    """
    e_idx = find_index_from_timestamp(dates, e_timestamp)

    # Search all remaining data after E
    if e_idx >= len(prices) - 1:
        return {
            'valid_50_bounce': False,
            'reason': 'E point at end of data'
        }

    remaining_prices = prices[e_idx + 1:]
    remaining_dates = dates[e_idx + 1:]

    # F point = 38.2% level of AB
    f_price = fib_levels['38.2']

    # Calculate 50% of F-E move
    fe_move = e_price - f_price  # Move from F to E
    s_level = f_price + (fe_move * 0.5)  # 50% of F-E move

    min_threshold = e_price * (min_threshold_pct / 100)

    print(f"\n  Checking 50% bounce after E (F-E move calculation):")
    print(f"    F (38.2% level): ${f_price:.2f}")
    print(f"    E (new {('high' if direction == 'up' else 'low')}): ${e_price:.2f}")
    print(f"    F-E move: ${fe_move:.2f}")
    print(f"    S (50% of F-E): ${s_level:.2f}")
    print(f"    Min threshold: {min_threshold_pct}% = ${min_threshold:.2f}")

    # STEP 1: Check if minimum threshold is crossed from E
    threshold_crossed = False
    threshold_idx = None
    threshold_point = None

    if direction == 'up':
        # After E (high), look for drop by min_threshold_pct
        threshold_price = e_price - min_threshold
        for i, price in enumerate(remaining_prices):
            if price < threshold_price:
                threshold_crossed = True
                threshold_idx = i
                threshold_point = (remaining_dates[i], price)
                print(f"    ✓ Threshold crossed at {remaining_dates[i]}, ${price:.2f}")
                break
    elif direction == 'down':
        # After E (low), look for rise by min_threshold_pct
        threshold_price = e_price + min_threshold
        for i, price in enumerate(remaining_prices):
            if price > threshold_price:
                threshold_crossed = True
                threshold_idx = i
                threshold_point = (remaining_dates[i], price)
                print(f"    ✓ Threshold crossed at {remaining_dates[i]}, ${price:.2f}")
                break

    if not threshold_crossed:
        print(f"    ✗ Threshold not crossed")
        return {
            'valid_50_bounce': False,
            'reason': f'Price did not move {min_threshold_pct}% from E',
            'threshold_pct': min_threshold_pct,
            's_level': s_level,
            'f_price': f_price,
            'fe_move': fe_move
        }

    # STEP 2: Look for ACTUAL BOUNCE at S level (50% of F-E move)
    tolerance = s_level * 0.0002  # 0.02% tolerance for S level

    search_prices = remaining_prices[threshold_idx:]
    search_dates = remaining_dates[threshold_idx:]

    print(f"    🔍 Searching for ACTUAL BOUNCE at S level ${s_level:.2f} (tolerance: {tolerance:.2f})")

    for i, price in enumerate(search_prices):
        if abs(price - s_level) <= tolerance:
            print(f"    📍 Found price touch at S level: ${price:.2f} at {search_dates[i]}")

            # CRITICAL: Verify ACTUAL BOUNCE BEHAVIOR
            bounce_confirmed = False
            bounce_strength = 0

            # Need at least 4 more data points to confirm bounce
            if i + 4 < len(search_prices):
                s_touch_price = price
                next_prices = search_prices[i + 1:i + 5]  # Next 4 prices after S touch

                print(f"    🔄 Verifying bounce behavior...")
                print(f"       S touch price: ${s_touch_price:.2f}")
                print(f"       Next 4 prices: {[f'${p:.2f}' for p in next_prices]}")

                if direction == 'up':
                    # UPTREND: After E (high) drops to S level, it should bounce UP (away from F level)
                    # S level is between F and E, so bouncing UP means moving toward E direction
                    prices_above_s = [p for p in next_prices if p > s_touch_price]
                    bounce_strength = len(prices_above_s)

                    if bounce_strength >= 3:  # At least 3 out of 4 prices above S touch
                        bounce_confirmed = True
                        print(f"    ✅ UPTREND BOUNCE CONFIRMED: {bounce_strength}/4 prices above S touch")
                    else:
                        print(f"    ❌ UPTREND bounce failed: Only {bounce_strength}/4 prices above S touch")

                elif direction == 'down':
                    # DOWNTREND: After E (low) rises to S level, it should bounce DOWN (away from F level)
                    # S level is between E and F, so bouncing DOWN means moving toward E direction
                    prices_below_s = [p for p in next_prices if p < s_touch_price]
                    bounce_strength = len(prices_below_s)

                    if bounce_strength >= 3:  # At least 3 out of 4 prices below S touch
                        bounce_confirmed = True
                        print(f"    ✅ DOWNTREND BOUNCE CONFIRMED: {bounce_strength}/4 prices below S touch")
                    else:
                        print(f"    ❌ DOWNTREND bounce failed: Only {bounce_strength}/4 prices below S touch")
            else:
                print(f"    ⚠️  Not enough data points after S touch to verify bounce")

            if bounce_confirmed:
                bounce_timestamp = search_dates[i]
                bounce_point = (bounce_timestamp, price)
                print(f"    🎯 REAL 50% BOUNCE FOUND at S level!")
                print(f"       Bounce point: {bounce_timestamp}, ${price:.2f}")
                print(f"       S level target: ${s_level:.2f}")
                print(f"       Bounce strength: {bounce_strength}/4 confirming prices")

                return {
                    'valid_50_bounce': True,
                    'reason': f'Threshold crossed and ACTUAL bounce confirmed at S (50% of F-E move)',
                    'bounce_point': bounce_point,
                    'threshold_point': threshold_point,
                    'threshold_pct': min_threshold_pct,
                    's_level': s_level,
                    'f_price': f_price,
                    'fe_move': fe_move,
                    'bounce_strength': bounce_strength,
                    'direction': direction
                }
            else:
                print(f"    ⏭️  Price touched S but no bounce confirmed, continuing search...")
                # Continue searching for other potential bounce points
                continue

    print(f"    ❌ No ACTUAL bounce found at S level")
    print(f"       Searched {len(search_prices)} prices after threshold cross")
    print(f"       Looking for bounce near S = ${s_level:.2f} with tolerance ${tolerance:.2f}")

    return {
        'valid_50_bounce': False,
        'reason': 'Threshold crossed but no ACTUAL bounce confirmed at S (50% of F-E move)',
        'threshold_point': threshold_point,
        'threshold_pct': min_threshold_pct,
        's_level': s_level,
        'f_price': f_price,
        'fe_move': fe_move,
        'bounce_strength': 0,
        'direction': direction
    }


def find_valid_e_with_trailing_strategy(prices, dates, d_timestamp, d_price, fib_levels, direction, min_change,
                                        min_threshold_pct):
    """
    TRAILING E POINT STRATEGY: Find S bounce first, then use ABSOLUTE extreme before bounce as E
    """
    # Get all potential E points
    all_e_points = find_all_e_points_after_d(prices, dates, d_timestamp, d_price, min_change, direction,
                                             min_threshold_pct)

    if not all_e_points:
        return {
            'e_point': None,
            'validation': {'valid_50_bounce': False, 'reason': 'No E points found'},
            'candidate_number': 0,
            'total_candidates_tested': 0,
            'is_valid': False,
            'strategy': 'trailing',
            'trailing_attempts': []
        }

    trailing_attempts = []

    # Test each E point for S bounce
    for i, (initial_e_timestamp, initial_e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']
        fe_move = initial_e_price - f_price
        s_level = f_price + (fe_move * 0.5)

        print(f"\n--- TRAILING ATTEMPT {i + 1}/{len(all_e_points)} ---")
        print(f"Testing initial E: {initial_e_timestamp}, ${initial_e_price:.2f}")

        # Check for S bounce using initial E
        validation = check_50_percent_bounce_after_e_correct(
            prices, dates, initial_e_timestamp, initial_e_price, fib_levels, direction, min_threshold_pct
        )

        if validation['valid_50_bounce']:
            print(f"\n🎯 S BOUNCE FOUND! Now finding ABSOLUTE {direction.upper()} extreme before bounce...")

            # Get bounce point
            bounce_timestamp = validation['bounce_point'][0]
            bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
            d_idx = find_index_from_timestamp(dates, d_timestamp)

            # Find ABSOLUTE extreme between D and bounce point
            search_prices = prices[d_idx:bounce_idx + 1]
            search_dates = dates[d_idx:bounce_idx + 1]

            if direction == 'up':
                # Find ABSOLUTE HIGHEST before bounce
                max_price = max(search_prices)
                max_idx = search_prices.index(max_price)
                final_e_timestamp = search_dates[max_idx]
                final_e_price = max_price
                print(f"    ✨ ABSOLUTE HIGHEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            elif direction == 'down':
                # Find ABSOLUTE LOWEST before bounce
                min_price = min(search_prices)
                min_idx = search_prices.index(min_price)
                final_e_timestamp = search_dates[min_idx]
                final_e_price = min_price
                print(f"    ✨ ABSOLUTE LOWEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            # Recalculate everything with the ABSOLUTE extreme
            final_fe_move = final_e_price - f_price
            final_s_level = f_price + (final_fe_move * 0.5)

            print(f"    📊 FINAL CALCULATION:")
            print(f"       F: ${f_price:.2f}")
            print(f"       E (absolute extreme): ${final_e_price:.2f}")
            print(f"       F-E move: ${final_fe_move:.2f}")
            print(f"       S level: ${final_s_level:.2f}")

            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (final_e_timestamp, final_e_price),  # Use absolute extreme
                'distance_from_d': abs(final_e_price - d_price),
                'fe_move': final_fe_move,
                's_level': final_s_level,
                'validation': validation,
                'found_bounce': True
            }
            trailing_attempts.append(attempt_record)

            return {
                'e_point': (final_e_timestamp, final_e_price),
                'validation': validation,
                'candidate_number': i + 1,
                'total_candidates_tested': i + 1,
                'is_valid': True,
                'distance_from_d': abs(final_e_price - d_price),
                'strategy': 'trailing',
                'trailing_attempts': trailing_attempts,
                'successful_attempt': i + 1
            }
        else:
            print(f"    ❌ No S bounce found with this E candidate")
            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'distance_from_d': abs(initial_e_price - d_price),
                'fe_move': fe_move,
                's_level': s_level,
                'validation': validation,
                'found_bounce': False
            }
            trailing_attempts.append(attempt_record)

    # No valid pattern found
    # No valid pattern found
    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': trailing_attempts[-1]['validation'] if trailing_attempts else {'valid_50_bounce': False,
                                                                                     'reason': 'No E points found'},
        'candidate_number': len(all_e_points),
        'total_candidates_tested': len(all_e_points),
        'is_valid': False,
        'distance_from_d': abs(all_e_points[-1][1] - d_price) if all_e_points else 0,  # ADD THIS LINE
        'strategy': 'trailing',
        'trailing_attempts': trailing_attempts,
        'successful_attempt': None
    }


def check_pattern_completion_236_extension(prices, dates, s_bounce_timestamp, s_bounce_price, f_price, e_price,
                                           direction):
    """
    FIXED: Correct -23.6% extension calculation

    UPTREND Logic:
    - F = 38.2% level (LOWER than E)
    - E = New HIGH (ABOVE F)
    - Target = E + 23.6% of (E-F) distance = HIGHER than E

    DOWNTREND Logic:
    - F = 38.2% level (HIGHER than E)
    - E = New LOW (BELOW F)
    - Target = E - 23.6% of (F-E) distance = LOWER than E
    """
    s_idx = find_index_from_timestamp(dates, s_bounce_timestamp)

    if s_idx >= len(prices) - 1:
        return {
            'pattern_completed': False,
            'reason': 'S bounce at end of data'
        }

    # Calculate extension target based on ACTUAL pattern direction
    if direction == 'up':
        # UPTREND: E is HIGH, F is lower, target should be ABOVE E
        fe_move_distance = e_price - f_price  # Positive: E > F
        extension_236 = e_price + (fe_move_distance * 0.236)  # Target ABOVE E

        print(f"\n  🎯 UPTREND Completion Calculation:")
        print(f"    F (38.2% level): ${f_price:.2f}")
        print(f"    E (HIGH): ${e_price:.2f}")
        print(f"    E-F move: ${fe_move_distance:.2f}")
        print(f"    Target (E + 23.6%): ${extension_236:.2f} (ABOVE E)")

    elif direction == 'down':
        # DOWNTREND: E is LOW, F is higher, target should be BELOW E
        fe_move_distance = f_price - e_price  # Positive: F > E
        extension_236 = e_price - (fe_move_distance * 0.236)  # Target BELOW E

        print(f"\n  🎯 DOWNTREND Completion Calculation:")
        print(f"    F (38.2% level): ${f_price:.2f}")
        print(f"    E (LOW): ${e_price:.2f}")
        print(f"    F-E move: ${fe_move_distance:.2f}")
        print(f"    Target (E - 23.6%): ${extension_236:.2f} (BELOW E)")

    print(f"    S bounce: ${s_bounce_price:.2f} at {s_bounce_timestamp}")

    # Search for target after S bounce
    remaining_prices = prices[s_idx + 1:]
    remaining_dates = dates[s_idx + 1:]
    tolerance = abs(extension_236) * 0.005

    for i, price in enumerate(remaining_prices):
        target_reached = False

        if direction == 'up':
            # UPTREND: Look for price to exceed target ABOVE E
            target_reached = price >= extension_236 - tolerance
        elif direction == 'down':
            # DOWNTREND: Look for price to fall below target BELOW E
            target_reached = price <= extension_236 + tolerance

        if target_reached:
            completion_timestamp = remaining_dates[i]
            completion_point = (completion_timestamp, price)

            print(f"    ✅ PATTERN COMPLETED! Target reached at {completion_timestamp}")
            print(f"       Target: ${extension_236:.2f}, Actual: ${price:.2f}")
            print(f"       Accuracy: ${abs(price - extension_236):.2f}")

            return {
                'pattern_completed': True,
                'completion_point': completion_point,
                'target_price': extension_236,
                'actual_price': price,
                'accuracy': abs(price - extension_236),
                'direction': direction,
                'fe_move_distance': fe_move_distance,
                'reason': f'Price reached -23.6% extension target ({direction}trend)'
            }

    print(f"    ⏳ Pattern not yet completed - target not reached")
    print(f"       Searched {len(remaining_prices)} prices after S bounce")

    return {
        'pattern_completed': False,
        'target_price': extension_236,
        'direction': direction,
        'fe_move_distance': fe_move_distance,
        'reason': 'Target -23.6% extension not yet reached',
        'prices_searched': len(remaining_prices)
    }


def find_valid_e_with_trailing_strategy_complete(prices, dates, d_timestamp, d_price, fib_levels, direction, min_change,
                                                 min_threshold_pct):
    """
    ENHANCED trailing strategy with pattern completion detection
    """
    # Get all potential E points
    all_e_points = find_all_e_points_after_d(prices, dates, d_timestamp, d_price, min_change, direction,
                                             min_threshold_pct)

    if not all_e_points:
        return {
            'e_point': None,
            'validation': {'valid_50_bounce': False, 'reason': 'No E points found'},
            'candidate_number': 0,
            'total_candidates_tested': 0,
            'is_valid': False,
            'pattern_completed': False,
            'strategy': 'trailing_with_completion',
            'trailing_attempts': []
        }

    trailing_attempts = []

    # Test each E point for S bounce AND pattern completion
    for i, (initial_e_timestamp, initial_e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']
        fe_move = initial_e_price - f_price
        s_level = f_price + (fe_move * 0.5)

        print(f"\n--- TRAILING ATTEMPT {i + 1}/{len(all_e_points)} ---")
        print(f"Testing initial E: {initial_e_timestamp}, ${initial_e_price:.2f}")

        # Initialize completion_result with default values
        completion_result = {
            'pattern_completed': False,
            'reason': 'No S bounce found'
        }

        # Check for S bounce using initial E
        validation = check_50_percent_bounce_after_e_correct(
            prices, dates, initial_e_timestamp, initial_e_price, fib_levels, direction, min_threshold_pct
        )

        if validation['valid_50_bounce']:
            print(f"\n🎯 S BOUNCE FOUND! Now finding ABSOLUTE {direction.upper()} extreme before bounce...")

            # Get bounce point
            bounce_timestamp = validation['bounce_point'][0]
            bounce_price = validation['bounce_point'][1]
            bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
            d_idx = find_index_from_timestamp(dates, d_timestamp)

            # Find ABSOLUTE extreme between D and bounce point
            search_prices = prices[d_idx:bounce_idx + 1]
            search_dates = dates[d_idx:bounce_idx + 1]

            if direction == 'up':
                # Find ABSOLUTE HIGHEST before bounce
                max_price = max(search_prices)
                max_idx = search_prices.index(max_price)
                final_e_timestamp = search_dates[max_idx]
                final_e_price = max_price
                print(f"    ✨ ABSOLUTE HIGHEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            elif direction == 'down':
                # Find ABSOLUTE LOWEST before bounce
                min_price = min(search_prices)
                min_idx = search_prices.index(min_price)
                final_e_timestamp = search_dates[min_idx]
                final_e_price = min_price
                print(f"    ✨ ABSOLUTE LOWEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            # Recalculate everything with the ABSOLUTE extreme
            final_fe_move = final_e_price - f_price
            final_s_level = f_price + (final_fe_move * 0.5)

            # NOW CHECK FOR PATTERN COMPLETION using the bounce point
            completion_result = check_pattern_completion_236_extension(
                prices, dates, bounce_timestamp, bounce_price, f_price, final_e_price, direction
            )

            # Pattern is VALID regardless of completion status
            print(f"    📊 FINAL CALCULATION:")
            print(f"       F (100%): ${f_price:.2f}")
            print(f"       E (0%): ${final_e_price:.2f}")
            print(f"       F-E move: ${final_fe_move:.2f}")
            print(f"       S level: ${final_s_level:.2f}")

            if completion_result['pattern_completed']:
                print(f"    🏆 PATTERN STATUS: VALID + COMPLETED!")
            else:
                print(f"    ✅ PATTERN STATUS: VALID (uncompleted)")

            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (final_e_timestamp, final_e_price),
                'distance_from_d': abs(final_e_price - d_price),
                'fe_move': final_fe_move,
                's_level': final_s_level,
                'validation': validation,
                'completion': completion_result,  # ✅ Fixed
                'found_bounce': True,
                'pattern_completed': completion_result['pattern_completed']
            }
            trailing_attempts.append(attempt_record)

            # RETURN VALID PATTERN (completed OR uncompleted)
            return {
                'e_point': (final_e_timestamp, final_e_price),
                'validation': validation,
                'completion': completion_result,
                'candidate_number': i + 1,
                'total_candidates_tested': i + 1,
                'is_valid': True,  # Valid regardless of completion
                'pattern_completed': completion_result['pattern_completed'],
                'distance_from_d': abs(final_e_price - d_price),
                'strategy': 'trailing_with_completion',
                'trailing_attempts': trailing_attempts,
                'successful_attempt': i + 1
            }
        else:
            print(f"    ❌ No S bounce found with this E candidate")
            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'distance_from_d': abs(initial_e_price - d_price),
                'fe_move': fe_move,
                's_level': s_level,
                'validation': validation,
                'completion': {'pattern_completed': False, 'reason': 'No S bounce'},  # ✅ Fixed
                'found_bounce': False,
                'pattern_completed': False
            }
            trailing_attempts.append(attempt_record)

    # No valid pattern found
    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': trailing_attempts[-1]['validation'] if trailing_attempts else {'valid_50_bounce': False,
                                                                                     'reason': 'No E points found'},
        'completion': {'pattern_completed': False, 'reason': 'No valid S bounce found'},
        'candidate_number': len(all_e_points),
        'total_candidates_tested': len(all_e_points),
        'is_valid': False,
        'pattern_completed': False,
        'distance_from_d': abs(all_e_points[-1][1] - d_price) if all_e_points else 0,
        'strategy': 'trailing_with_completion',
        'trailing_attempts': trailing_attempts,
        'successful_attempt': None
    }


def analyze_fan_extension_with_completion(pattern, prices, dates, min_change=0.01, min_threshold_pct=1.0):
    """
    Analyze fan extension with completion detection (23.6% extension)
    """
    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(pattern)

    # Get D point details
    d_timestamp = pattern['D'][0]
    d_price = pattern['D'][1]
    direction = pattern.get('direction', 'unknown')

    print(f"\n{'=' * 70}")
    print(f"ANALYZING {direction.upper()} PATTERN - WITH COMPLETION DETECTION")
    print(f"{'=' * 70}")
    print(f"  D point: {d_timestamp}, ${d_price:.2f}")
    print(f"  F point (100% level): ${fib_levels['38.2']:.2f}")
    print(f"  Target: -23.6% extension for pattern completion")

    # Determine search direction
    analysis_key = None
    if direction == 'down':
        print(f"  🔍 Looking for LOWs (E) → S bounce → completion target")
        analysis_key = 'lowest_after_d'
    elif direction == 'up':
        print(f"  🔍 Looking for HIGHs (E) → S bounce → completion target")
        analysis_key = 'highest_after_d'
    else:
        print(f"  🔍 Unknown direction, defaulting to HIGH search")
        analysis_key = 'highest_after_d'
        direction = 'up'

    # Use ENHANCED TRAILING STRATEGY with completion detection
    e_result = find_valid_e_with_trailing_strategy_complete(
        prices, dates, d_timestamp, d_price, fib_levels, direction, min_change, min_threshold_pct
    )

    # Build analysis results
    analysis = {
        'pattern': pattern,
        'fibonacci_levels': fib_levels,
        'retracement_382_level': fib_levels['38.2'],
        'f_level': fib_levels['38.2'],
        'd_timestamp': d_timestamp,
        'd_price': d_price,
        'pattern_direction': direction,
        'min_threshold_pct': min_threshold_pct,
        'min_distance_from_d': d_price * (min_threshold_pct / 100),
        'is_valid': False,
        'pattern_completed': False,
        'calculation_method': 'Trailing E Point Strategy with 23.6% completion detection',
        'strategy': 'trailing_with_completion'
    }

    if e_result and e_result['is_valid']:
        # Found valid E point
        e_point = e_result['e_point']
        e_timestamp, e_price = e_point

        validation_info = e_result['validation']
        # FIXED: Safe access to completion info
        completion_info = e_result.get('completion', {
            'pattern_completed': False,
            'reason': 'No completion data available'
        })
        s_level = validation_info['s_level']
        f_price = validation_info['f_price']
        fe_move = validation_info['fe_move']

        analysis[analysis_key] = e_point
        analysis['validation_info'] = validation_info
        analysis['completion_info'] = completion_info
        analysis['is_valid'] = True
        analysis['pattern_completed'] = completion_info.get('pattern_completed', False)
        analysis['s_level'] = s_level
        analysis['f_price'] = f_price
        analysis['fe_move'] = fe_move
        analysis['distance_from_d'] = e_result['distance_from_d']
        analysis['trailing_attempts'] = e_result['trailing_attempts']

        #if e_result and e_result['is_valid']:
        # ... [keep everything the same until here] ...

        # FIX: Use the SAME calculation logic as the completion function
        if direction == 'up':
            # UPTREND: E is above F, target is further above E
            fe_move_distance = e_price - f_price  # Should be positive
            extension_236 = e_price + (fe_move_distance * 0.236)
            print(f"  DEBUG: UPTREND calc - E: ${e_price:.2f}, F: ${f_price:.2f}")
            print(f"  DEBUG: fe_move_distance: ${fe_move_distance:.2f}")
            print(f"  DEBUG: extension_236: ${extension_236:.2f}")

        elif direction == 'down':
            # DOWNTREND: E is below F, target is further below E
            fe_move_distance = f_price - e_price  # Should be positive
            extension_236 = e_price - (fe_move_distance * 0.236)
            print(f"  DEBUG: DOWNTREND calc - E: ${e_price:.2f}, F: ${f_price:.2f}")
            print(f"  DEBUG: fe_move_distance: ${fe_move_distance:.2f}")
            print(f"  DEBUG: extension_236: ${extension_236:.2f}")
        else:
            # Default uptrend
            fe_move_distance = abs(e_price - f_price)
            extension_236 = e_price + (fe_move_distance * 0.236)

        analysis['completion_target'] = extension_236


        print(f"\n🎯 PATTERN ANALYSIS COMPLETE!")
        print(f"  ✅ Valid E point: ${e_price:.2f}")
        print(f"  ✅ S bounce confirmed: ${validation_info['bounce_point'][1]:.2f}")
        print(f"  🎯 Completion target (-23.6%): ${extension_236:.2f}")

        if completion_info.get('pattern_completed', False):
            print(f"  🏆 STATUS: VALID + COMPLETED!")
            print(f"      Target: ${completion_info.get('target_price', 'N/A'):.2f}")
            print(f"      Actual: ${completion_info.get('actual_price', 'N/A'):.2f}")
            print(f"      Accuracy: ${completion_info.get('accuracy', 'N/A'):.2f}")
        else:
            print(f"  ⏳ STATUS: VALID (waiting for completion)")
            print(f"      Still monitoring for target: ${extension_236:.2f}")

    else:
        # Strategy failed
        analysis['validation_info'] = e_result.get('validation', {
            'valid_50_bounce': False,
            'reason': 'No validation data'
        })
        # FIXED: Safe access to completion info when pattern failed
        analysis['completion_info'] = e_result.get('completion', {
            'pattern_completed': False,
            'reason': 'No valid pattern found'
        })
        analysis['is_valid'] = False
        analysis['pattern_completed'] = False
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

        print(f"\n❌ NO VALID PATTERN FOUND")

    return analysis
def plot_pattern_with_completion_analysis(analysis, prices, dates, min_change=0.01):
    """
    Professional visualization showing F-E-S pattern with completion target
    Shows both completed and uncompleted valid patterns
    """
    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

    # Classic professional setup
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(22, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Get display range
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)

    min_pattern_idx = min(pattern_indices)
    max_pattern_idx = max(pattern_indices)

    # Include final E point in display range
    final_e_point = None
    if analysis.get('lowest_after_d'):
        final_e_point = analysis['lowest_after_d']
    elif analysis.get('highest_after_d'):
        final_e_point = analysis['highest_after_d']

    if final_e_point:
        e_idx = find_index_from_timestamp(dates, final_e_point[0])
        max_display_idx = max(max_pattern_idx, e_idx + 250)
    else:
        max_display_idx = max_pattern_idx

    # Calculate display range with padding
    pattern_range = max_pattern_idx - min_pattern_idx
    padding = max(100, int(pattern_range * 0.7))
    start_idx = max(0, min_pattern_idx - padding)
    end_idx = min(len(prices) - 1, max_display_idx + padding)

    # Plot price data
    subset_prices = prices[start_idx:end_idx + 1]
    subset_dates = dates[start_idx:end_idx + 1]
    ax.plot(subset_dates, subset_prices, color='#1f77b4', linewidth=2, label='Price', zorder=1)

    # Pattern points A, B, C, D
    points = ['A', 'B', 'C', 'D']
    point_colors = {'A': '#2F2F2F', 'B': '#D62728', 'C': '#FF7F0E', 'D': '#1f77b4'}

    for point in points:
        timestamp, price = pattern[point]
        color = point_colors[point]
        ax.plot(timestamp, price, 'o', color=color, markersize=12, zorder=10,
                markeredgecolor='white', markeredgewidth=2)
        ax.text(timestamp, price, point, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', zorder=15)

    # Draw pattern lines
    for i in range(len(points) - 1):
        timestamp1, price1 = pattern[points[i]]
        timestamp2, price2 = pattern[points[i + 1]]
        ax.plot([timestamp1, timestamp2], [price1, price2], color='#666666',
                linewidth=2, alpha=0.8, zorder=4)

    # F level and point
    f_price = analysis.get('f_price', analysis['retracement_382_level'])
    d_timestamp = pattern['D'][0]

    ax.axhline(y=f_price, color='#2CA02C', linestyle='--', linewidth=3, alpha=0.8, zorder=3)
    ax.plot(d_timestamp, f_price, 's', color='#2CA02C', markersize=16, zorder=15,
            markeredgecolor='white', markeredgewidth=3)
    ax.text(d_timestamp, f_price, 'F', ha='center', va='center', fontsize=14,
            fontweight='black', color='white', zorder=16,
            path_effects=[pe.withStroke(linewidth=3, foreground='#1B5E1F')])

    # S level
    if analysis.get('s_level'):
        s_level = analysis['s_level']
        ax.axhline(y=s_level, color='#9467BD', linewidth=4, alpha=0.9, zorder=3)

    # COMPLETION TARGET LINE
    if analysis.get('completion_target'):
        target_price = analysis['completion_target']
        ax.axhline(y=target_price, color='#FF6B35', linestyle=':', linewidth=3, alpha=0.9, zorder=3)

    # Plot E point
    if final_e_point:
        e_timestamp, e_price = final_e_point

        # E point - solid diamond
        ax.plot(e_timestamp, e_price, 'D', color='#228B22', markersize=18, zorder=15,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(e_timestamp, e_price, 'E', ha='center', va='center', fontsize=16,
                fontweight='black', color='white', zorder=16,
                path_effects=[pe.withStroke(linewidth=4, foreground='#006400')])

    # Validation points
    validation_info = analysis.get('validation_info', {})

    # Threshold cross
    if validation_info.get('threshold_point'):
        threshold_timestamp, threshold_price = validation_info['threshold_point']
        ax.plot(threshold_timestamp, threshold_price, '^', color='#FF7F0E', markersize=10, zorder=13,
                markeredgecolor='white', markeredgewidth=2)

    # S bounce point
    if validation_info.get('bounce_point'):
        bounce_timestamp, bounce_price = validation_info['bounce_point']
        ax.plot(bounce_timestamp, bounce_price, '*', color='#DC143C', markersize=20, zorder=20,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(bounce_timestamp, bounce_price, 'S', ha='center', va='center', fontsize=18,
                fontweight='black', color='white', zorder=21,
                path_effects=[pe.withStroke(linewidth=4, foreground='#8B0000')])

    # COMPLETION POINT (if exists)
    completion_info = analysis.get('completion_info', {})
    if completion_info.get('pattern_completed') and completion_info.get('completion_point'):
        comp_timestamp, comp_price = completion_info['completion_point']

        # Large gold star for completion
        ax.plot(comp_timestamp, comp_price, '*', color='#FFD700', markersize=24, zorder=22,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(comp_timestamp, comp_price, 'T', ha='center', va='center', fontsize=18,
                fontweight='black', color='white', zorder=23,
                path_effects=[pe.withStroke(linewidth=4, foreground='#B8860B')])

    # Draw F-E connection line
    if analysis.get('is_valid', False) and final_e_point:
        ax.plot([d_timestamp, final_e_point[0]], [f_price, final_e_point[1]],
                color='#17BECF', linewidth=3, alpha=0.8, linestyle='--', zorder=8)

    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2, label='Price'),
        plt.Line2D([0], [0], color='#2CA02C', linestyle='--', linewidth=3, label=f'F Level (100%): ${f_price:.0f}'),
    ]

    if analysis.get('s_level'):
        s_level = analysis['s_level']
        legend_elements.append(
            plt.Line2D([0], [0], color='#9467BD', linewidth=4, label=f'S Level (50%): ${s_level:.0f}')
        )

    if analysis.get('completion_target'):
        target_price = analysis['completion_target']
        legend_elements.append(
            plt.Line2D([0], [0], color='#FF6B35', linestyle=':', linewidth=3,
                       label=f'Target (-23.6%): ${target_price:.0f}')
        )

    if analysis.get('is_valid', False):
        fe_move = analysis.get('fe_move', 0)
        legend_elements.append(
            plt.Line2D([0], [0], color='#17BECF', linestyle='--', linewidth=3, label=f'F-E Move: ${fe_move:.0f}')
        )

    legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
                       fontsize=11, facecolor='white', edgecolor='gray')
    legend.get_frame().set_linewidth(1)

    # Format chart
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax.set_xlabel('Time', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color='#333333')

    # Enhanced title with completion status
    pattern_completed = analysis.get('pattern_completed', False)

    if analysis.get('is_valid', False):
        if pattern_completed:
            title = f'🏆 {direction.upper()} Pattern - COMPLETED\nF-E-S Pattern + Target Reached'
            title_color = '#FFD700'  # Gold for completed
        else:
            title = f'✅ {direction.upper()} Pattern - VALID (Uncompleted)\nF-E-S Pattern Confirmed, Awaiting Target'
            title_color = '#2CA02C'  # Green for valid
    else:
        title = f'❌ {direction.upper()} Pattern - INVALID\nNo Valid F-E-S Pattern Found'
        title_color = '#D62728'  # Red for invalid

    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='top', color=title_color)

    # Professional grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3, color='#CCCCCC')
    ax.grid(True, which='minor', linestyle=':', alpha=0.1, color='#DDDDDD')

    # Clean axes styling
    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(1)

    ax.tick_params(colors='#333333', which='both')

    plt.tight_layout()
    plt.show()

    # Enhanced console output
    print("=" * 70)
    if analysis.get('is_valid', False):
        if pattern_completed:
            print("🏆 PATTERN STATUS: VALID + COMPLETED")
            comp_info = analysis['completion_info']
            print(f"  ✅ E Point: ${final_e_point[1]:.2f}")
            print(f"  ✅ S Bounce: ${validation_info['bounce_point'][1]:.2f}")
            print(f"  🎯 Target: ${comp_info['target_price']:.2f}")
            print(f"  🏆 Completion: ${comp_info['actual_price']:.2f}")
            print(f"  📊 Accuracy: ${comp_info['accuracy']:.2f}")
        else:
            print("✅ PATTERN STATUS: VALID (UNCOMPLETED)")
            print(f"  ✅ E Point: ${final_e_point[1]:.2f}")
            print(f"  ✅ S Bounce: ${validation_info['bounce_point'][1]:.2f}")
            print(f"  ⏳ Target: ${analysis['completion_target']:.2f}")
            print(f"  📈 Status: Monitoring for completion")
    else:
        print("❌ PATTERN STATUS: INVALID")
        print(f"  🔍 No valid F-E-S pattern found")
    print("=" * 70)

def plot_pattern_with_extension(analysis, prices, dates, min_change=0.01):
    """
    Plot pattern with F-E-S visualization
    ENHANCED: Shows F point marker on 38.2% line and all key measurement points
    """
    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

    plt.figure(figsize=(20, 12))

    # Convert timestamps to indices for plotting calculations
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)

    min_pattern_idx = min(pattern_indices)
    max_pattern_idx = max(pattern_indices)

    # Include E point if exists
    e_point = None
    e_label = None
    e_idx = None

    if analysis.get('lowest_after_d'):
        e_timestamp, e_price = analysis['lowest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx + 100)
        e_point = (e_timestamp, e_price)
        e_label = "E (Low)"
    elif analysis.get('highest_after_d'):
        e_timestamp, e_price = analysis['highest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx + 100)
        e_point = (e_timestamp, e_price)
        e_label = "E (High)"
    else:
        max_display_idx = max_pattern_idx

    # Calculate display range
    pattern_range = max_pattern_idx - min_pattern_idx
    padding = max(100, int(pattern_range * 0.7))
    start_idx = max(0, min_pattern_idx - padding)
    end_idx = min(len(prices) - 1, max_display_idx + padding)

    # Plot price data
    subset_prices = prices[start_idx:end_idx + 1]
    subset_dates = dates[start_idx:end_idx + 1]
    plt.plot(subset_dates, subset_prices, 'b-', alpha=0.5, linewidth=0.8, label='Price')

    # Plot pattern points
    points = ['A', 'B', 'C', 'D']
    point_colors = {'A': 'black', 'B': 'red', 'C': 'orange', 'D': 'blue'}

    for point in points:
        timestamp, price = pattern[point]
        color = point_colors.get(point, 'gray')
        plt.plot(timestamp, price, 'o', color=color, markersize=10, zorder=5)
        plt.text(timestamp, price, f'{point}', ha='center', va='center', fontsize=12, fontweight='bold',
                 color='white', zorder=6)

    # Draw pattern lines
    for i in range(len(points) - 1):
        timestamp1, price1 = pattern[points[i]]
        timestamp2, price2 = pattern[points[i + 1]]
        plt.plot([timestamp1, timestamp2], [price1, price2], 'r-', linewidth=2, zorder=4)

    # Plot F level (38.2% level) - ENHANCED
    f_price = analysis.get('f_price', analysis['retracement_382_level'])
    plt.axhline(y=f_price, color='green', linestyle='--', alpha=0.7,
                label=f'F Level (38.2%): ${f_price:.2f}', zorder=3, linewidth=2)

    # ===== NEW: ADD F POINT MARKER ON 38.2% LINE =====
    # Find where to place F point marker - use D timestamp or middle of pattern
    d_timestamp = pattern['D'][0]

    # Plot F point marker on the 38.2% line
    plt.plot(d_timestamp, f_price, 'D', color='green', markersize=12, zorder=7,
             markeredgecolor='darkgreen', markeredgewidth=2)
    plt.text(d_timestamp, f_price, 'F', ha='center', va='center', fontsize=12, fontweight='bold',
             color='white', zorder=8)

    # Plot S level (50% of F-E move) if available
    if analysis.get('s_level'):
        s_level = analysis['s_level']
        plt.axhline(y=s_level, color='purple', linestyle='-', alpha=0.9, linewidth=3,
                    label=f'S Level (50% of F-E): ${s_level:.2f}', zorder=3)

        # ===== NEW: ADD S POINT MARKERS =====
        # Add S point marker at key locations
        if e_point:
            e_timestamp = e_point[0]
            # Add S marker near E point
            plt.plot(e_timestamp, s_level, 's', color='purple', markersize=14, zorder=7,
                     markeredgecolor='darkmagenta', markeredgewidth=2)
            plt.text(e_timestamp, s_level, f'S (50% F-E)\n${s_level:.2f}\nBOUNCE TARGET',
                     ha='left', va='bottom', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.8), zorder=8)

    # Plot validation points if available
    validation_info = analysis.get('validation_info', {})

    # ===== ENHANCED: Plot threshold cross point =====
    if validation_info.get('threshold_point'):
        threshold_timestamp, threshold_price = validation_info['threshold_point']
        threshold_pct = validation_info.get('threshold_pct', 1.0)
        plt.plot(threshold_timestamp, threshold_price, '^', color='orange', markersize=12, zorder=6,
                 markeredgecolor='darkorange', markeredgewidth=2,
                 label=f'{threshold_pct}% Threshold Cross')

    # ===== ENHANCED: Plot S bounce point =====
    if validation_info.get('bounce_point'):
        bounce_timestamp, bounce_price = validation_info['bounce_point']
        plt.plot(bounce_timestamp, bounce_price, '*', color='purple', markersize=16, zorder=6,
                 markeredgecolor='darkmagenta', markeredgewidth=2,
                 label='S Bounce (50% of F-E)')

    # ===== ENHANCED: Plot E point with detailed validation status =====
    if e_point:
        e_timestamp, e_price = e_point

        # Get search statistics
        candidate_num = analysis.get('e_candidate_number', 1)
        total_tested = analysis.get('total_e_candidates_tested', 1)

        # Color based on validation
        if analysis.get('is_valid', False):
            e_color = 'lime'
            e_marker = 'o'
            validation_text = f"✓ VALID E#{candidate_num}\n(of {total_tested} tested)\nS BOUNCE FOUND"
        else:
            e_color = 'red'
            e_marker = 'x'
            validation_text = f"✗ INVALID E\n({total_tested} tested)\nNO S BOUNCE"

        plt.plot(e_timestamp, e_price, e_marker, color=e_color, markersize=14, zorder=5,
                 markeredgecolor='darkgreen' if analysis.get('is_valid', False) else 'darkred',
                 markeredgewidth=2)
        plt.text(e_timestamp, e_price, 'E', ha='center', va='center', fontsize=12, fontweight='bold',
                 color='white', zorder=6)

        # ===== ENHANCED: Draw F-E line with measurements =====
        if analysis.get('is_valid', False):
            # Draw line from F to E with enhanced styling
            plt.plot([d_timestamp, e_timestamp], [f_price, e_price],
                     color='cyan', linewidth=4, alpha=0.8, linestyle='--',
                     label=f'F-E Move: ${analysis.get("fe_move", 0):.2f}', zorder=4)

            # Add midpoint marker on F-E line
            mid_timestamp = pd.to_datetime(d_timestamp) + (
                        pd.to_datetime(e_timestamp) - pd.to_datetime(d_timestamp)) / 2
            mid_price = f_price + (e_price - f_price) / 2
            plt.plot(mid_timestamp, mid_price, 'D', color='cyan', markersize=8, zorder=6,
                     markeredgecolor='darkcyan', markeredgewidth=1)

    # ===== NEW: Add measurement annotations =====
    if analysis.get('is_valid', False) and e_point:
        # Add measurement arrows and annotations
        fe_move = analysis.get('fe_move', 0)

        # Arrow from F to 50% level
        if analysis.get('s_level'):
            s_level = analysis['s_level']
            plt.annotate('', xy=(d_timestamp, s_level), xytext=(d_timestamp, f_price),
                         arrowprops=dict(arrowstyle='<->', color='purple', lw=2, alpha=0.8))

    # ===== ENHANCED: Format and labels =====
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)

    validation_status = "VALID ✓" if analysis.get('is_valid', False) else "INVALID ✗"
    threshold_pct = analysis.get('min_threshold_pct', 1.0)
    total_tested = analysis.get('total_e_candidates_tested', 0)

    title = f'{direction.upper()}TREND - {validation_status} F-E-S Method\n'
    title += f'({total_tested} E candidates tested, Threshold: {threshold_pct}%)'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, which='both', linestyle='-', alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_all_key_points_detail(analysis, prices, dates):
    """
    Alternative detailed plotting function that shows ALL measurement points clearly
    """
    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), height_ratios=[3, 1])

    # Main price chart
    ax = ax1

    # Get display range (same logic as before)
    pattern_indices = []
    for point in ['A', 'B', 'C', 'D']:
        timestamp, price = pattern[point]
        idx = find_index_from_timestamp(dates, timestamp)
        pattern_indices.append(idx)

    min_pattern_idx = min(pattern_indices)
    max_pattern_idx = max(pattern_indices)

    e_point = None
    if analysis.get('lowest_after_d'):
        e_timestamp, e_price = analysis['lowest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx + 100)
        e_point = (e_timestamp, e_price)
    elif analysis.get('highest_after_d'):
        e_timestamp, e_price = analysis['highest_after_d']
        e_idx = find_index_from_timestamp(dates, e_timestamp)
        max_display_idx = max(max_pattern_idx, e_idx + 100)
        e_point = (e_timestamp, e_price)
    else:
        max_display_idx = max_pattern_idx

    pattern_range = max_pattern_idx - min_pattern_idx
    padding = max(100, int(pattern_range * 0.7))
    start_idx = max(0, min_pattern_idx - padding)
    end_idx = min(len(prices) - 1, max_display_idx + padding)

    subset_prices = prices[start_idx:end_idx + 1]
    subset_dates = dates[start_idx:end_idx + 1]

    # Plot price
    ax.plot(subset_dates, subset_prices, 'b-', alpha=0.6, linewidth=1, label='Price')

    # Plot all pattern points with large markers
    points = ['A', 'B', 'C', 'D']
    colors = {'A': 'black', 'B': 'red', 'C': 'orange', 'D': 'blue'}

    for point in points:
        timestamp, price = pattern[point]
        ax.plot(timestamp, price, 'o', color=colors[point], markersize=12, zorder=10)
        ax.text(timestamp, price, f'{point}\n${price:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # Plot F point prominently on 38.2% line
    f_price = analysis.get('f_price', analysis['retracement_382_level'])
    d_timestamp = pattern['D'][0]

    # F level line
    ax.axhline(y=f_price, color='green', linestyle='--', alpha=0.8, linewidth=3,
               label=f'F Level (38.2%): ${f_price:.2f}', zorder=5)

    # F point marker - LARGE and prominent
    ax.plot(d_timestamp, f_price, 'D', color='green', markersize=16, zorder=15,
            markeredgecolor='darkgreen', markeredgewidth=3)
    ax.text(d_timestamp, f_price, f'F POINT\n(38.2% Level)\n${f_price:.2f}\nMEASUREMENT START',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))

    # S level and point
    if analysis.get('s_level'):
        s_level = analysis['s_level']
        ax.axhline(y=s_level, color='purple', linestyle='-', alpha=0.9, linewidth=3,
                   label=f'S Level (50% F-E): ${s_level:.2f}', zorder=5)

        # S point markers at multiple locations
        if e_point:
            e_timestamp = e_point[0]
            ax.plot(e_timestamp, s_level, 's', color='purple', markersize=16, zorder=15,
                    markeredgecolor='darkmagenta', markeredgewidth=3)
            ax.text(e_timestamp, s_level, f'S POINT\n(50% of F-E Move)\n${s_level:.2f}\nBOUNCE TARGET',
                    ha='left', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="plum", alpha=0.9))

    # E point
    if e_point:
        e_timestamp, e_price = e_point
        is_valid = analysis.get('is_valid', False)

        color = 'lime' if is_valid else 'red'
        marker = 'o' if is_valid else 'x'

        ax.plot(e_timestamp, e_price, marker, color=color, markersize=16, zorder=15,
                markeredgecolor='darkgreen' if is_valid else 'darkred', markeredgewidth=3)

        status = "VALID E POINT" if is_valid else "INVALID E POINT"
        ax.text(e_timestamp, e_price, f'{status}\n${e_price:.2f}\nMEASUREMENT END',
                ha='center', va='top' if direction == 'down' else 'bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.3))

        # F-E line with measurement
        if is_valid:
            ax.plot([d_timestamp, e_timestamp], [f_price, e_price],
                    color='cyan', linewidth=5, alpha=0.8, linestyle='--',
                    label=f'F-E Move: ${analysis.get("fe_move", 0):.2f}', zorder=8)

    # Validation points
    validation_info = analysis.get('validation_info', {})

    if validation_info.get('threshold_point'):
        threshold_timestamp, threshold_price = validation_info['threshold_point']
        threshold_pct = validation_info.get('threshold_pct', 1.0)
        ax.plot(threshold_timestamp, threshold_price, '^', color='orange', markersize=14, zorder=12,
                markeredgecolor='darkorange', markeredgewidth=2)
        ax.text(threshold_timestamp, threshold_price, f'THRESHOLD CROSS\n{threshold_pct}%\n${threshold_price:.2f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8))

    if validation_info.get('bounce_point'):
        bounce_timestamp, bounce_price = validation_info['bounce_point']
        ax.plot(bounce_timestamp, bounce_price, '*', color='purple', markersize=18, zorder=12,
                markeredgecolor='darkmagenta', markeredgewidth=2)
        ax.text(bounce_timestamp, bounce_price, f'S BOUNCE CONFIRMED\n${bounce_price:.2f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mediumorchid", alpha=0.8))

    # Format main chart
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{direction.upper()}TREND Pattern - F-E-S Analysis with Key Measurement Points',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Detailed measurement panel
    ax2.text(0.1, 0.8, "KEY MEASUREMENTS:", fontsize=12, fontweight='bold')

    measurements = []
    measurements.append(f"F Point (38.2% Level): ${f_price:.2f}")
    if analysis.get('s_level'):
        measurements.append(f"S Point (50% of F-E): ${analysis['s_level']:.2f}")
    if e_point:
        measurements.append(f"E Point: ${e_point[1]:.2f}")
    if analysis.get('fe_move'):
        measurements.append(f"F-E Move Distance: ${analysis['fe_move']:.2f}")
        measurements.append(f"F-E Move %: {analysis.get('fe_move', 0) / f_price * 100:.2f}%")

    y_pos = 0.6
    for measurement in measurements:
        ax2.text(0.1, y_pos, f"• {measurement}", fontsize=11)
        y_pos -= 0.15

    # Validation status
    status = "✓ VALID PATTERN" if analysis.get('is_valid', False) else "✗ INVALID PATTERN"
    ax2.text(0.6, 0.8, f"STATUS: {status}", fontsize=12, fontweight='bold',
             color='green' if analysis.get('is_valid', False) else 'red')

    if validation_info.get('reason'):
        ax2.text(0.6, 0.6, f"Reason: {validation_info['reason']}", fontsize=10)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def get_completed_patterns_for_date(date_str, min_change=0.01, min_threshold_pct=1.0):
    """
    Load data for a specific date and get all completed patterns with TRAILING STRATEGY

    Args:
        date_str (str): Date in YYYY-MM-DD format
        min_change: Minimum percentage change from D to consider a new point
        min_threshold_pct: Minimum threshold percentage for validation AND minimum distance from D

    Returns:
        dict: All completed patterns and data for that date with trailing E point analysis
    """
    print(f"\n{'=' * 70}")
    print(f"FAN EXTENSION ANALYSIS WITH TRAILING E POINT STRATEGY")
    print(f"{'=' * 70}")
    print(f"Date: {date_str}")
    print(f"Change threshold: {min_change * 100:.1f}%")
    print(f"Distance threshold: {min_threshold_pct}% (E points must be this far from D)")
    print(f"Strategy: Trail through E candidates until 50% retracement found")

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

    # Analyze fan extensions for each completed pattern using TRAILING STRATEGY
    extension_analyses = []
    trailing_summary = {
        'total_patterns': len(completed_patterns),
        'successful_patterns': 0,
        'failed_patterns': 0,
        'total_attempts_across_all': 0,
        'average_attempts_per_pattern': 0
    }

    for i, pattern in enumerate(completed_patterns):
        print(f"\n{'=' * 50}")
        print(f"COMPLETED PATTERN {i + 1}/{len(completed_patterns)}")
        print(f"{'=' * 50}")
        print(f"  Direction: {pattern.get('direction', 'unknown')}")
        print(f"  A: {pattern['A'][0]}, Price ${pattern['A'][1]:.2f}")
        print(f"  B: {pattern['B'][0]}, Price ${pattern['B'][1]:.2f}")
        print(f"  C: {pattern['C'][0]}, Price ${pattern['C'][1]:.2f}")
        print(f"  D: {pattern['D'][0]}, Price ${pattern['D'][1]:.2f}")

        # Calculate required distance from D
        d_price = pattern['D'][1]
        required_distance = d_price * (min_threshold_pct / 100)
        print(f"  Required E distance from D: ${required_distance:.2f}")

        # Analyze fan extension with TRAILING STRATEGY
        extension_analysis = analyze_fan_extension_with_completion(pattern, prices, dates, min_change,
                                                                   min_threshold_pct)
        extension_analyses.append(extension_analysis)

        # Update trailing summary
        trailing_attempts = extension_analysis.get('trailing_attempts', [])
        trailing_summary['total_attempts_across_all'] += len(trailing_attempts)

        if extension_analysis['is_valid']:
            trailing_summary['successful_patterns'] += 1
            successful_attempt = extension_analysis.get('successful_attempt', 1)
            print(f"\n🎯 PATTERN {i + 1} SUCCESS - Found valid extension after {successful_attempt} attempts")
        else:
            trailing_summary['failed_patterns'] += 1
            print(f"\n❌ PATTERN {i + 1} FAILED - No valid extension found after {len(trailing_attempts)} attempts")

        print(f"\n📊 Pattern {i + 1} Results:")
        print(f"    38.2% Level: ${extension_analysis['retracement_382_level']:.2f}")
        if 's_level' in extension_analysis:
            print(f"    S Level (50% retracement): ${extension_analysis['s_level']:.2f}")
        print(f"    Min Distance from D: ${extension_analysis.get('min_distance_from_d', 0):.2f}")
        print(f"    Valid: {extension_analysis['is_valid']}")
        print(f"    E candidates tested: {len(trailing_attempts)}")

        if extension_analysis['is_valid']:
            if extension_analysis.get('lowest_after_d'):
                e_timestamp, e_price = extension_analysis['lowest_after_d']
                distance = extension_analysis.get('distance_from_d', 0)
                successful_attempt = extension_analysis.get('successful_attempt', 1)
                print(f"    ✓ VALID E (Lowest): {e_timestamp} → ${e_price:.2f}")
                print(f"      Distance from D: ${distance:.2f}")
                print(f"      Success on attempt: #{successful_attempt}")
            elif extension_analysis.get('highest_after_d'):
                e_timestamp, e_price = extension_analysis['highest_after_d']
                distance = extension_analysis.get('distance_from_d', 0)
                successful_attempt = extension_analysis.get('successful_attempt', 1)
                print(f"    ✓ VALID E (Highest): {e_timestamp} → ${e_price:.2f}")
                print(f"      Distance from D: ${distance:.2f}")
                print(f"      Success on attempt: #{successful_attempt}")
        else:
            print(f"    ❌ Pattern failed trailing strategy")
            if 'validation_info' in extension_analysis:
                print(f"    Final reason: {extension_analysis['validation_info']['reason']}")

        # Show trailing attempts summary for this pattern
        if trailing_attempts:
            print(f"\n    📋 Trailing Attempts Summary for Pattern {i + 1}:")
            for attempt in trailing_attempts:
                status = "✓ BOUNCE" if attempt['found_bounce'] else "❌ NO BOUNCE"
                print(
                    f"      E#{attempt['candidate_number']}: ${attempt['e_point'][1]:.2f} → S: ${attempt['s_level']:.2f} → {status}")

    # Calculate final statistics
    if trailing_summary['total_patterns'] > 0:
        trailing_summary['average_attempts_per_pattern'] = trailing_summary['total_attempts_across_all'] / \
                                                           trailing_summary['total_patterns']

    print(f"\n{'=' * 70}")
    print(f"TRAILING STRATEGY FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"📊 Overall Statistics:")
    print(f"  Total patterns analyzed: {trailing_summary['total_patterns']}")
    print(f"  Successful patterns: {trailing_summary['successful_patterns']}")
    print(f"  Failed patterns: {trailing_summary['failed_patterns']}")
    print(
        f"  Success rate: {(trailing_summary['successful_patterns'] / trailing_summary['total_patterns'] * 100):.1f}%")
    print(f"  Total E candidates tested across all patterns: {trailing_summary['total_attempts_across_all']}")
    print(f"  Average E candidates per pattern: {trailing_summary['average_attempts_per_pattern']:.1f}")

    return {
        "status": "success",
        "date": date_str,
        "completed_patterns": completed_patterns,
        "extension_analyses": extension_analyses,
        "prices": prices,
        "dates": dates,
        "total_patterns": len(all_patterns),
        "completed_count": len(completed_patterns),
        "valid_extensions": trailing_summary['successful_patterns'],
        "failed_extensions": trailing_summary['failed_patterns'],
        "threshold_used": min_threshold_pct,
        "strategy": "trailing",
        "trailing_summary": trailing_summary,
        "ready_for_fan_extension": True
    }

if __name__ == "__main__":
    # Example usage with trailing strategy
    date_to_analyze = "2025-07-22"
    params = load_parameters()
    min_change = params.get('min_change', 0.001)
    min_threshold_pct = params.get('min_threshold_pct', 0.001)  # Default 1%

    # Set to True if you want to see plots
    show_plots = True

    print(f"🚀 Starting trailing E point analysis for {date_to_analyze}")
    results = get_completed_patterns_for_date(date_to_analyze, min_change, min_threshold_pct)

    if results["status"] == "success":
        print(f"\n🎯 TRAILING STRATEGY COMPLETED!")
        print(f"Found {results['completed_count']} completed patterns")
        print(f"Valid extensions: {results['valid_extensions']}")
        print(f"Failed extensions: {results['failed_extensions']}")
        print(f"Success rate: {(results['valid_extensions']/results['completed_count']*100):.1f}%")
        print(f"Strategy: {results['strategy']}")

        if show_plots and results['extension_analyses']:
            print("\n📈 Generating plots for patterns with trailing analysis...")
            for i, analysis in enumerate(results['extension_analyses']):
                if analysis.get('lowest_after_d') or analysis.get('highest_after_d'):
                    print(f"\nPlotting pattern {i + 1} (trailing attempts: {len(analysis.get('trailing_attempts', []))})...")
                    plot_pattern_with_completion_analysis(analysis, results['prices'], results['dates'], min_change)
    else:
        print(f"❌ Error: {results['status']}")