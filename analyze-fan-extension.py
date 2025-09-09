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


def calculate_80_percent_retracement(f_price, e_price, direction, failure_percentage=0.80):
    """
    Calculate the failure retracement level between F and E points.
    This represents the failure threshold for excessive retracement beyond 50%.

    failure_percentage retracement = failure_percentage% back from E toward F (deep retracement failure level)
    50% retracement = 50% back from E toward F (normal S bounce level)

    In uptrend: F < failure_level < S level < E
    In downtrend: E < S level < failure_level < F

    Args:
        f_price: F point price (38.2% level)
        e_price: E point price (0% level)
        direction: 'up' or 'down'
        failure_percentage: Percentage for failure threshold (default 0.80 = 80%)
    """
    if direction == 'up':
        # UPTREND: F is below E, failure_percentage% retracement moves DOWN from E toward F
        fe_move = e_price - f_price  # Positive (E above F)
        retracement_80 = e_price - (fe_move * failure_percentage)  # Move failure_percentage% back toward F
        print(
            f"    UPTREND {failure_percentage * 100:.0f}% calc: E=${e_price:.2f} - {failure_percentage * 100:.0f}% of ${fe_move:.2f} = ${retracement_80:.2f}")
    elif direction == 'down':
        # DOWNTREND: F is above E, failure_percentage% retracement moves UP from E toward F
        fe_move = f_price - e_price  # Positive (F above E)
        retracement_80 = e_price + (fe_move * failure_percentage)  # Move failure_percentage% back toward F
        print(
            f"    DOWNTREND {failure_percentage * 100:.0f}% calc: E=${e_price:.2f} + {failure_percentage * 100:.0f}% of ${fe_move:.2f} = ${retracement_80:.2f}")
    else:
        # Auto-detect direction
        if e_price > f_price:  # Uptrend
            fe_move = e_price - f_price
            retracement_80 = e_price - (fe_move * failure_percentage)
        else:  # Downtrend
            fe_move = f_price - e_price
            retracement_80 = e_price + (fe_move * failure_percentage)

    return retracement_80


def check_80_percent_failure(prices, dates, e_timestamp, e_price, f_price, direction, start_from_timestamp=None,
                             failure_percentage=0.80):
    """
    Check if price crosses retracement failure level after E point.
    ENHANCED: Better logging and more comprehensive failure detection.
    - UPTREND: Failure if price goes BELOW failure level (too much retracement down)
    - DOWNTREND: Failure if price goes ABOVE failure level (too much retracement up)

    Args:
        failure_percentage: Percentage for failure threshold (default 0.80 = 80%)
    """
    if start_from_timestamp:
        start_idx = find_index_from_timestamp(dates, start_from_timestamp)
    else:
        start_idx = find_index_from_timestamp(dates, e_timestamp)

    if start_idx >= len(prices) - 1:
        return {
            'failed_80_percent': False,
            'reason': 'At end of data'
        }

    retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction, failure_percentage)

    print(f"  🔍 ENHANCED {failure_percentage * 100:.0f}% RETRACEMENT FAILURE CHECK:")
    print(f"    F (100% level): ${f_price:.2f}")
    print(f"    E (0% level): ${e_price:.2f}")
    print(f"    {failure_percentage * 100:.0f}% retracement level: ${retracement_80:.2f}")
    print(f"    Checking prices from index {start_idx + 1} to {len(prices) - 1}")

    if direction == 'up':
        print(f"    UPTREND: Failure if price goes BELOW ${retracement_80:.2f} (excessive retracement)")
    else:
        print(f"    DOWNTREND: Failure if price goes ABOVE ${retracement_80:.2f} (excessive retracement)")

    # Track the closest approach to failure level for debugging
    closest_approach = float('inf')
    closest_timestamp = None
    closest_price = None

    for i in range(start_idx + 1, len(prices)):
        current_price = prices[i]
        current_timestamp = dates[i]

        # Track closest approach for debugging
        if direction == 'up':
            distance_to_failure = current_price - retracement_80
            if distance_to_failure < closest_approach:
                closest_approach = distance_to_failure
                closest_timestamp = current_timestamp
                closest_price = current_price
        else:
            distance_to_failure = retracement_80 - current_price
            if distance_to_failure < closest_approach:
                closest_approach = distance_to_failure
                closest_timestamp = current_timestamp
                closest_price = current_price

        if direction == 'up':
            if current_price <= retracement_80:
                print(f"    ❌ UPTREND FAILURE DETECTED!")
                print(
                    f"       Price ${current_price:.2f} <= {failure_percentage * 100:.0f}% level ${retracement_80:.2f}")
                print(f"       At timestamp: {current_timestamp}")
                return {
                    'failed_80_percent': True,
                    'failure_point': (current_timestamp, current_price),
                    'failure_level': retracement_80,
                    'reason': f'UPTREND: Excessive retracement - price went below {failure_percentage * 100:.0f}% level',
                    'closest_approach': closest_approach,
                    'closest_point': (closest_timestamp, closest_price)
                }
        elif direction == 'down':
            if current_price >= retracement_80:
                print(f"    ❌ DOWNTREND FAILURE DETECTED!")
                print(
                    f"       Price ${current_price:.2f} >= {failure_percentage * 100:.0f}% level ${retracement_80:.2f}")
                print(f"       At timestamp: {current_timestamp}")
                return {
                    'failed_80_percent': True,
                    'failure_point': (current_timestamp, current_price),
                    'failure_level': retracement_80,
                    'reason': f'DOWNTREND: Excessive retracement - price went above {failure_percentage * 100:.0f}% level',
                    'closest_approach': closest_approach,
                    'closest_point': (closest_timestamp, closest_price)
                }

    print(f"    ✅ No {failure_percentage * 100:.0f}% retracement failure detected")
    print(f"    📊 Closest approach to failure level: ${closest_approach:.2f}")
    if closest_timestamp:
        print(f"    📍 Closest point: {closest_timestamp} at ${closest_price:.2f}")

    return {
        'failed_80_percent': False,
        'failure_level': retracement_80,
        'reason': f'No excessive retracement beyond {failure_percentage * 100:.0f}% level',
        'closest_approach': closest_approach,
        'closest_point': (closest_timestamp, closest_price)
    }


def find_valid_e_with_trailing_strategy_complete(prices, dates, d_timestamp, d_price, fib_levels, direction, min_change,
                                                 min_threshold_pct, failure_percentage=0.80):
    """
    ENHANCED trailing strategy with configurable retracement failure criteria

    Args:
        failure_percentage: Percentage for failure threshold (default 0.80 = 80%)
    """
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
            'failed_80_percent': False,
            'strategy': 'trailing_with_completion',
            'trailing_attempts': []
        }

    trailing_attempts = []

    for i, (initial_e_timestamp, initial_e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']
        fe_move = initial_e_price - f_price
        s_level = f_price + (fe_move * 0.5)

        print(
            f"\n--- TRAILING ATTEMPT {i + 1}/{len(all_e_points)} (WITH {failure_percentage * 100:.0f}% FAILURE CHECK) ---")
        print(f"Testing initial E: {initial_e_timestamp}, ${initial_e_price:.2f}")

        validation = check_50_percent_bounce_after_e_correct(
            prices, dates, initial_e_timestamp, initial_e_price, fib_levels, direction, min_threshold_pct,
            failure_percentage
        )

        # Check if pattern failed due to 80% retracement
        if validation.get('failed_80_percent', False):
            print(f"    ❌ PATTERN FAILED: 80% retracement violation")
            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'distance_from_d': abs(initial_e_price - d_price),
                'fe_move': fe_move,
                's_level': s_level,
                'validation': validation,
                'completion': {'pattern_completed': False, 'reason': '80% retracement failure'},
                'found_bounce': validation['valid_50_bounce'],
                'pattern_completed': False,
                'failed_80_percent': True,
                'failure_type': 'retracement_violation',
                'final_valid': False
            }
            trailing_attempts.append(attempt_record)

            # Return immediately on 80% failure
            return {
                'e_point': (initial_e_timestamp, initial_e_price),
                'validation': validation,
                'completion': attempt_record['completion'],
                'candidate_number': i + 1,
                'total_candidates_tested': i + 1,
                'is_valid': False,
                'pattern_completed': False,
                'failed_80_percent': True,
                'failure_info': validation.get('failure_info', {}),
                'distance_from_d': abs(initial_e_price - d_price),
                'strategy': 'trailing_with_completion',
                'trailing_attempts': trailing_attempts,
                'successful_attempt': None
            }

        if validation['valid_50_bounce']:
            print(f"\n🎯 Initial S BOUNCE FOUND! Finding {direction.upper()} extreme BEFORE bounce...")

            bounce_timestamp = validation['bounce_point'][0]
            bounce_price = validation['bounce_point'][1]
            bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
            d_idx = find_index_from_timestamp(dates, d_timestamp)

            # Search ONLY from D to the bounce point (not including bounce point)
            search_prices = prices[d_idx:bounce_idx]
            search_dates = dates[d_idx:bounce_idx]

            if direction == 'up':
                max_price = max(search_prices)
                max_idx = search_prices.index(max_price)
                final_e_timestamp = search_dates[max_idx]
                final_e_price = max_price
                print(f"    ✨ HIGHEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")
            elif direction == 'down':
                min_price = min(search_prices)
                min_idx = search_prices.index(min_price)
                final_e_timestamp = search_dates[min_idx]
                final_e_price = min_price
                print(f"    ✨ LOWEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            # Re-validate S using the final E with configurable failure check
            final_validation = check_50_percent_bounce_after_e_correct(
                prices, dates, final_e_timestamp, final_e_price, fib_levels, direction, min_threshold_pct,
                failure_percentage
            )

            # Check if final validation failed due to 80%
            if final_validation.get('failed_80_percent', False):
                print(f"    ❌ FINAL E FAILED: 80% retracement violation")
                attempt_record = {
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'distance_from_d': abs(final_e_price - d_price),
                    'fe_move': abs(final_e_price - f_price),
                    's_level': final_validation.get('s_level', s_level),
                    'validation': final_validation,
                    'completion': {'pattern_completed': False, 'reason': '80% retracement failure'},
                    'found_bounce': final_validation['valid_50_bounce'],
                    'pattern_completed': False,
                    'failed_80_percent': True,
                    'final_valid': False
                }
                trailing_attempts.append(attempt_record)

                return {
                    'e_point': (final_e_timestamp, final_e_price),
                    'validation': final_validation,
                    'completion': attempt_record['completion'],
                    'candidate_number': i + 1,
                    'total_candidates_tested': i + 1,
                    'is_valid': False,
                    'pattern_completed': False,
                    'failed_80_percent': True,
                    'failure_info': final_validation.get('failure_info', {}),
                    'distance_from_d': abs(final_e_price - d_price),
                    'strategy': 'trailing_with_completion',
                    'trailing_attempts': trailing_attempts,
                    'successful_attempt': None
                }

            if not final_validation['valid_50_bounce']:
                print(f"    ❌ No valid S bounce after final E")
                attempt_record = {
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'distance_from_d': abs(final_e_price - d_price),
                    'fe_move': abs(final_e_price - f_price),
                    's_level': final_validation.get('s_level', s_level),
                    'validation': final_validation,
                    'completion': {'pattern_completed': False, 'reason': 'No S bounce after final E'},
                    'found_bounce': False,
                    'pattern_completed': False,
                    'failed_80_percent': False,
                    'final_valid': False
                }
                trailing_attempts.append(attempt_record)
                continue

            final_bounce_timestamp = final_validation['bounce_point'][0]
            final_bounce_price = final_validation['bounce_point'][1]
            final_s_level = final_validation['s_level']
            final_fe_move = abs(final_e_price - f_price)

            tolerance = final_s_level * 0.0005
            if abs(final_bounce_price - final_s_level) > tolerance:
                print(f"    ❌ Bounce at ${final_bounce_price:.2f} not at final S: ${final_s_level:.2f}")
                attempt_record = {
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'distance_from_d': abs(final_e_price - d_price),
                    'fe_move': final_fe_move,
                    's_level': final_s_level,
                    'validation': final_validation,
                    'completion': {'pattern_completed': False, 'reason': 'Bounce not at final S level'},
                    'found_bounce': True,
                    'pattern_completed': False,
                    'failed_80_percent': False,
                    'final_valid': False
                }
                trailing_attempts.append(attempt_record)
                continue

            print(f"    ✅ Fixed S bounce at final S: ${final_s_level:.2f} at {final_bounce_timestamp}")

            # Check completion with 80% failure monitoring
            completion_result = check_pattern_completion_236_extension(
                prices, dates, final_bounce_timestamp, final_e_price, f_price, direction
            )

            print(f"    📊 FINAL CALCULATION:")
            print(f"       F: ${f_price:.2f}")
            print(f"       E: ${final_e_price:.2f}")
            print(f"       S: ${final_s_level:.2f}")

            # Check if completion failed due to 80%
            if completion_result.get('failed_80_percent', False):
                print(f"    ❌ PATTERN FAILED: 80% retracement during target approach")
                attempt_record = {
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'distance_from_d': abs(final_e_price - d_price),
                    'fe_move': final_fe_move,
                    's_level': final_s_level,
                    'validation': final_validation,
                    'completion': completion_result,
                    'found_bounce': True,
                    'pattern_completed': False,
                    'failed_80_percent': True,
                    'final_valid': False
                }
                trailing_attempts.append(attempt_record)

                return {
                    'e_point': (final_e_timestamp, final_e_price),
                    'validation': final_validation,
                    'completion': completion_result,
                    'candidate_number': i + 1,
                    'total_candidates_tested': i + 1,
                    'is_valid': False,
                    'pattern_completed': False,
                    'failed_80_percent': True,
                    'failure_info': completion_result,
                    'distance_from_d': abs(final_e_price - d_price),
                    'strategy': 'trailing_with_completion',
                    'trailing_attempts': trailing_attempts,
                    'successful_attempt': None
                }

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
                'validation': final_validation,
                'completion': completion_result,
                'found_bounce': True,
                'pattern_completed': completion_result['pattern_completed'],
                'failed_80_percent': False,
                'final_valid': True
            }
            trailing_attempts.append(attempt_record)

            return {
                'e_point': (final_e_timestamp, final_e_price),
                'validation': final_validation,
                'completion': completion_result,
                'candidate_number': i + 1,
                'total_candidates_tested': i + 1,
                'is_valid': True,
                'pattern_completed': completion_result['pattern_completed'],
                'failed_80_percent': False,
                'distance_from_d': abs(final_e_price - d_price),
                'strategy': 'trailing_with_completion',
                'trailing_attempts': trailing_attempts,
                'successful_attempt': i + 1
            }
        else:
            print(f"    ❌ No initial S bounce found with this E candidate")
            attempt_record = {
                'candidate_number': i + 1,
                'e_point': (initial_e_timestamp, initial_e_price),
                'distance_from_d': abs(initial_e_price - d_price),
                'fe_move': fe_move,
                's_level': s_level,
                'validation': validation,
                'completion': {'pattern_completed': False, 'reason': 'No S bounce'},
                'found_bounce': False,
                'pattern_completed': False,
                'failed_80_percent': validation.get('failed_80_percent', False),
                'final_valid': False
            }
            trailing_attempts.append(attempt_record)

    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': trailing_attempts[-1]['validation'] if trailing_attempts else {
            'valid_50_bounce': False, 'reason': 'No E points found', 'failed_80_percent': False
        },
        'completion': {'pattern_completed': False, 'reason': 'No valid S bounce found'},
        'candidate_number': len(all_e_points),
        'total_candidates_tested': len(all_e_points),
        'is_valid': False,
        'pattern_completed': False,
        'failed_80_percent': False,
        'distance_from_d': abs(all_e_points[-1][1] - d_price) if all_e_points else 0,
        'strategy': 'trailing_with_completion',
        'trailing_attempts': trailing_attempts,
        'successful_attempt': None
    }


def analyze_fan_extension_with_completion(pattern, prices, dates, min_change=0.001, min_threshold_pct=1.0,
                                          failure_percentage=0.80):
    fib_levels = calculate_fibonacci_levels(pattern)

    # Get D point details
    d_timestamp = pattern['D'][0]
    d_price = pattern['D'][1]
    direction = pattern.get('direction', 'unknown')

    print(f"\n{'=' * 70}")
    print(f"ANALYZING {direction.upper()} PATTERN - WITH COMPLETION & 80% FAILURE DETECTION")
    print(f"{'=' * 70}")
    print(f"  D point: {d_timestamp}, ${d_price:.2f}")
    print(f"  F point (38.2% level): ${fib_levels['38.2']:.2f}")
    print(f"  Target: -23.6% extension for pattern completion")
    print(f"  Failure: {failure_percentage * 100:.0f}% retracement violation = pattern invalidation")

    # Determine search direction
    analysis_key = None
    if direction == 'down':
        print(f"  Looking for LOWs (E) → S bounce → completion target")
        analysis_key = 'lowest_after_d'
    elif direction == 'up':
        print(f"  Looking for HIGHs (E) → S bounce → completion target")
        analysis_key = 'highest_after_d'
    else:
        print(f"  Unknown direction, defaulting to HIGH search")
        analysis_key = 'highest_after_d'
        direction = 'up'

    # Use ENHANCED TRAILING STRATEGY with configurable failure criteria
    e_result = find_valid_e_with_trailing_strategy_complete(
        prices, dates, d_timestamp, d_price, fib_levels, direction, min_change, min_threshold_pct, failure_percentage
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
        'failed_80_percent': False,
        'recovery_type': None,
        'calculation_method': 'Trailing E Point Strategy with 50% retracement, 23.6% completion, 80% failure detection, and 100% retracement recovery',
        'strategy': 'trailing_with_completion_and_recovery'
    }

    if e_result and e_result.get('failed_80_percent', False):
        # Pattern failed due to 80% retracement
        analysis['failed_80_percent'] = True
        analysis['failure_info'] = e_result.get('failure_info', {})
        analysis['is_valid'] = False
        analysis['pattern_completed'] = False

        # Still record the E point for visualization
        if e_result.get('e_point'):
            e_point = e_result['e_point']
            analysis[analysis_key] = e_point

        analysis['validation_info'] = e_result.get('validation', {})
        analysis['completion_info'] = e_result.get('completion', {})
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

        # Calculate 80% level for visualization
        if e_result.get('e_point'):
            e_timestamp, e_price = e_result['e_point']
            f_price = fib_levels['38.2']
            retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction)
            analysis['retracement_80_level'] = retracement_80

        print(f"\n🚨 PATTERN ANALYSIS COMPLETE - FAILED DUE TO 80% RETRACEMENT!")

        # Show S bounce information if available (even for failed patterns)
        validation_info = e_result.get('validation', {})
        if validation_info.get('bounce_point'):
            bounce_timestamp, bounce_price = validation_info['bounce_point']
            print(f"  📍 S bounce found: {bounce_timestamp} at ${bounce_price:.2f}")
            print(f"  📊 S level (50% retracement): ${validation_info.get('s_level', 0):.2f}")
            print(f"  ⚠️  Pattern had valid S bounce but failed due to excessive retracement")

        failure_info = analysis.get('failure_info', {})
        if failure_info.get('failure_point'):
            failure_timestamp, failure_price = failure_info['failure_point']
            print(f"  📍 Failure point: {failure_timestamp} at ${failure_price:.2f}")
            print(f"  📊 80% retracement level: ${failure_info.get('failure_level', 0):.2f}")
            print(f"  📈 Closest approach: ${failure_info.get('closest_approach', 0):.2f}")
        print(f"  ❌ Pattern invalidated due to excessive retracement")

        # Show recovery information if available
        if analysis.get('recovery_type') == '100_percent_retracement':
            print(f"\n🔄 RECOVERY STATUS: 100% retracement recovery attempted")
            if analysis.get('secondary_attempt'):
                e2_timestamp, e2_price = analysis['E2']
                retracement_timestamp, retracement_price = analysis['retracement_point']
                print(f"  ✅ Recovery successful - New weaker fan pattern created")
                print(f"  📍 100% retracement: {retracement_timestamp} at ${retracement_price:.2f}")
                print(f"  📍 New E2: {e2_timestamp} at ${e2_price:.2f}")
                print(f"  🏷️ Pattern strength: {analysis['strength']}")
        elif analysis.get('recovery_type') == 'failed':
            print(f"\n🔄 RECOVERY STATUS: 100% retracement recovery failed")
            print(f"  ❌ No valid E2 found after 100% retracement")

        # ♻️ Try to recycle with 100% retracement recovery (NEW LOGIC)
        print(f"\n🔄 Attempting 100% retracement recovery...")
        recycle_result = recycle_failed_pattern_with_100_percent_retracement(
            prices, dates,
            fib_levels['38.2'],  # F price
            e_point[1],  # failed E price
            e_point[0],  # failed E timestamp
            direction,
            min_threshold_pct
        )
        if recycle_result:
            analysis['secondary_attempt'] = recycle_result
            analysis['recovery_type'] = '100_percent_retracement'
            # For plotting convenience
            analysis['E2'] = recycle_result['E2']
            analysis['strength'] = recycle_result['strength']
            analysis['retracement_point'] = recycle_result['retracement_point']
            analysis['retracement_type'] = recycle_result['retracement_type']

            e2_timestamp, e2_price = recycle_result['E2']
            retracement_timestamp, retracement_price = recycle_result['retracement_point']

            print(f"♻️ 100% RETRACEMENT RECOVERY SUCCESSFUL!")
            print(f"   📍 100% retracement: {retracement_timestamp} at ${retracement_price:.2f}")
            print(f"   📍 New E2 (weaker): {e2_timestamp} at ${e2_price:.2f}")
            print(f"   📊 F-E2 distance: ${recycle_result['fe2_distance']:.2f}")
            print(f"   🏷️ Status: {recycle_result['strength']} fan pattern")
        else:
            print(f"❌ 100% retracement recovery failed - no valid E2 found")
            analysis['recovery_type'] = 'failed'



    elif e_result and e_result['is_valid']:
        # Found valid E point (no 80% failure)
        e_point = e_result['e_point']
        e_timestamp, e_price = e_point

        validation_info = e_result['validation']
        completion_info = e_result.get('completion', {
            'pattern_completed': False,
            'reason': 'No completion data available'
        })

        # Get values from validation_info with safe defaults
        s_level = validation_info.get('s_level', (e_price + fib_levels['38.2']) / 2)
        f_price = validation_info.get('f_price', fib_levels['38.2'])
        fe_move = validation_info.get('fe_move', abs(e_price - f_price))

        analysis[analysis_key] = e_point
        analysis['validation_info'] = validation_info
        analysis['completion_info'] = completion_info
        analysis['is_valid'] = True
        analysis['pattern_completed'] = completion_info.get('pattern_completed', False)
        analysis['failed_80_percent'] = False
        analysis['s_level'] = s_level
        analysis['f_price'] = f_price
        analysis['fe_move'] = fe_move
        analysis['distance_from_d'] = e_result['distance_from_d']
        analysis['trailing_attempts'] = e_result['trailing_attempts']

        # Calculate 80% level for reference
        retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction)
        analysis['retracement_80_level'] = retracement_80

        # Calculate completion target
        if direction == 'up':
            fe_move_distance = e_price - f_price
            extension_236 = e_price + (fe_move_distance * 0.236)
        elif direction == 'down':
            fe_move_distance = f_price - e_price
            extension_236 = e_price - (fe_move_distance * 0.236)
        else:
            fe_move_distance = abs(e_price - f_price)
            extension_236 = e_price + (fe_move_distance * 0.236)

        analysis['completion_target'] = extension_236

        print(f"\n PATTERN ANALYSIS COMPLETE - VALID!")
        print(f"  Valid E point: ${e_price:.2f}")
        print(f"  S bounce confirmed: ${validation_info['bounce_point'][1]:.2f}")
        print(f"  80% safety level: ${retracement_80:.2f}")
        print(f"  Completion target (-23.6%): ${extension_236:.2f}")

        if completion_info.get('pattern_completed', False):
            print(f"  STATUS: VALID + COMPLETED!")
        else:
            print(f"  STATUS: VALID (waiting for completion)")

    else:
        # Strategy failed (no valid pattern found, no 80% failure)
        analysis['validation_info'] = e_result.get('validation', {
            'valid_50_bounce': False,
            'reason': 'No validation data',
            'failed_80_percent': False
        })
        analysis['completion_info'] = e_result.get('completion', {
            'pattern_completed': False,
            'reason': 'No valid pattern found'
        })
        analysis['is_valid'] = False
        analysis['pattern_completed'] = False
        analysis['failed_80_percent'] = False
        analysis['trailing_attempts'] = e_result.get('trailing_attempts', [])

        print(f"\n NO VALID PATTERN FOUND")

    return analysis


def find_valid_e_with_trailing_strategy(prices, dates, d_timestamp, d_price, fib_levels, direction, min_change,
                                        min_threshold_pct):
    """
    TRAILING E POINT STRATEGY: Find S bounce first, then use ABSOLUTE extreme before bounce as E
    """
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

    for i, (initial_e_timestamp, initial_e_price) in enumerate(all_e_points):
        f_price = fib_levels['38.2']
        fe_move = initial_e_price - f_price
        s_level = f_price + (fe_move * 0.5)

        print(f"\n--- TRAILING ATTEMPT {i + 1}/{len(all_e_points)} ---")
        print(f"Testing initial E: {initial_e_timestamp}, ${initial_e_price:.2f}")

        validation = check_50_percent_bounce_after_e_correct(
            prices, dates, initial_e_timestamp, initial_e_price, fib_levels, direction, min_threshold_pct
        )

        if validation['valid_50_bounce']:
            print(f"\n🎯 S BOUNCE FOUND! Now finding {direction.upper()} extreme BEFORE bounce...")

            bounce_timestamp = validation['bounce_point'][0]
            bounce_price = validation['bounce_point'][1]
            bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)
            d_idx = find_index_from_timestamp(dates, d_timestamp)

            # Search ONLY from D to the bounce point (not including bounce point)
            search_prices = prices[d_idx:bounce_idx]
            search_dates = dates[d_idx:bounce_idx]

            if direction == 'up':
                max_price = max(search_prices)
                max_idx = search_prices.index(max_price)
                final_e_timestamp = search_dates[max_idx]
                final_e_price = max_price
                print(f"    ✨ HIGHEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            elif direction == 'down':
                min_price = min(search_prices)
                min_idx = search_prices.index(min_price)
                final_e_timestamp = search_dates[min_idx]
                final_e_price = min_price
                print(f"    ✨ LOWEST before bounce: {final_e_timestamp}, ${final_e_price:.2f}")

            final_fe_move = final_e_price - f_price
            final_s_level = f_price + (final_fe_move * 0.5)

            # Verify the bounce price is close to the FINAL S level
            tolerance = final_s_level * 0.0002
            if abs(bounce_price - final_s_level) > tolerance:
                print(f"    ❌ Bounce at ${bounce_price:.2f} not at final S (50% of final F-E): ${final_s_level:.2f}")
                print(f"      Difference: ${abs(bounce_price - final_s_level):.2f} > tolerance ${tolerance:.2f}")
                attempt_record = {
                    'candidate_number': i + 1,
                    'e_point': (final_e_timestamp, final_e_price),
                    'distance_from_d': abs(final_e_price - d_price),
                    'fe_move': final_fe_move,
                    's_level': final_s_level,
                    'validation': validation,
                    'found_bounce': True,
                    'final_valid': False
                }
                trailing_attempts.append(attempt_record)
                continue

            print(f"    ✅ Bounce confirmed at final S (50% of final F-E move): ${final_s_level:.2f}")

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
                'found_bounce': True,
                'final_valid': True
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
                'found_bounce': False,
                'final_valid': False
            }
            trailing_attempts.append(attempt_record)

    # No valid pattern found
    return {
        'e_point': all_e_points[-1] if all_e_points else None,
        'validation': trailing_attempts[-1]['validation'] if trailing_attempts else {'valid_50_bounce': False,
                                                                                     'reason': 'No E points found'},
        'candidate_number': len(all_e_points),
        'total_candidates_tested': len(all_e_points),
        'is_valid': False,
        'distance_from_d': abs(all_e_points[-1][1] - d_price) if all_e_points else 0,
        'strategy': 'trailing',
        'trailing_attempts': trailing_attempts,
        'successful_attempt': None
    }


def check_50_percent_bounce_after_e_correct(prices, dates, e_timestamp, e_price, fib_levels, direction,
                                            min_threshold_pct, failure_percentage=0.80):
    """
    Check for the FIRST valid 50% retracement (S point) immediately after E.
    ENHANCED: Now includes configurable retracement failure criteria.

    Args:
        failure_percentage: Percentage for failure threshold (default 0.80 = 80%)
    """
    e_idx = find_index_from_timestamp(dates, e_timestamp)

    if e_idx >= len(prices) - 1:
        return {
            'valid_50_bounce': False,
            'reason': 'E point at end of data',
            'failed_80_percent': False
        }

    f_price = fib_levels['38.2']

    # S = 50% retracement from E back toward F
    if direction == 'up':
        s_level = e_price - ((e_price - f_price) * 0.5)
    elif direction == 'down':
        s_level = e_price + ((f_price - e_price) * 0.5)
    else:
        s_level = (e_price + f_price) / 2

    tolerance = s_level * 0.0002

    print(f"\n  Checking 50% retracement after E (with {failure_percentage * 100:.0f}% failure criteria):")
    print(f"    F (100% level): ${f_price:.2f}")
    print(f"    E (0% level): ${e_price:.2f}")
    print(f"    S (50% retracement): ${s_level:.2f}")

    # Look for the FIRST touch of S after E
    bounce_found = False
    bounce_point = None

    for i, price in enumerate(prices[e_idx + 1:], start=e_idx + 1):
        if abs(price - s_level) <= tolerance:
            bounce_timestamp = dates[i]
            bounce_point = (bounce_timestamp, price)
            bounce_found = True
            print(f"    FIRST VALID S bounce at {bounce_timestamp}, ${price:.2f}")
            break

    # Now check for retracement failure AFTER the S bounce (if found)
    if bounce_found:
        # Check for failure from the bounce point onwards
        failure_check = check_80_percent_failure(prices, dates, bounce_point[0], e_price, f_price, direction,
                                                 failure_percentage=failure_percentage)
    else:
        # Check for failure from E point onwards (no bounce found)
        failure_check = check_80_percent_failure(prices, dates, e_timestamp, e_price, f_price, direction,
                                                 failure_percentage=failure_percentage)

    if failure_check['failed_80_percent']:
        print(f"    🚨 PATTERN FAILED: Excessive retracement beyond 80% level")
        print(f"    📍 Failure point: {failure_check['failure_point'][0]} at ${failure_check['failure_point'][1]:.2f}")
        print(f"    📊 80% level: ${failure_check['failure_level']:.2f}")

        # Return with bounce information if found
        result = {
            'valid_50_bounce': bounce_found,
            'reason': '80% retracement failure - excessive retracement',
            'failed_80_percent': True,
            'failure_info': failure_check,
            's_level': s_level,
            'f_price': f_price,
            'e_price': e_price,
            'retracement_80_level': failure_check['failure_level']
        }

        if bounce_found:
            result['bounce_point'] = bounce_point
            print(f"    ⚠️  S bounce was found but pattern failed due to excessive retracement")

        return result

    if bounce_found:
        return {
            'valid_50_bounce': True,
            'reason': 'First S touch found after E',
            'bounce_point': bounce_point,
            's_level': s_level,
            'f_price': f_price,
            'e_price': e_price,
            'fe_move': abs(e_price - f_price),
            'direction': direction,
            'failed_80_percent': False,
            'retracement_80_level': failure_check['failure_level']
        }

    print(f"    No bounce found at S level")
    return {
        'valid_50_bounce': False,
        'reason': 'No bounce found at 50% retracement',
        's_level': s_level,
        'f_price': f_price,
        'e_price': e_price,
        'fe_move': abs(e_price - f_price),
        'direction': direction,
        'failed_80_percent': False,
        'retracement_80_level': failure_check['failure_level']
    }


def check_pattern_completion_236_extension(prices, dates, bounce_timestamp, e_price, f_price, direction):
    """
    Check for pattern completion at -23.6% extension from E.
    ENHANCED: Monitors for 80% retracement failure during target approach.
    """
    bounce_idx = find_index_from_timestamp(dates, bounce_timestamp)

    if bounce_idx >= len(prices) - 1:
        return {
            'pattern_completed': False,
            'reason': 'S bounce at end of data',
            'failed_80_percent': False
        }

    # Calculate extension target
    if direction == 'up':
        fe_move_distance = e_price - f_price
        extension_236 = e_price + (fe_move_distance * 0.236)
        print(f"\n  🎯 UPTREND Completion Calculation:")
        print(f"    F (100%): ${f_price:.2f}")
        print(f"    E (0%): ${e_price:.2f}")
        print(f"    Target (-23.6%): ${extension_236:.2f} (above E)")
    elif direction == 'down':
        fe_move_distance = f_price - e_price
        extension_236 = e_price - (fe_move_distance * 0.236)
        print(f"\n  🎯 DOWNTREND Completion Calculation:")
        print(f"    F (100%): ${f_price:.2f}")
        print(f"    E (0%): ${e_price:.2f}")
        print(f"    Target (-23.6%): ${extension_236:.2f} (below E)")
    else:
        return {
            'pattern_completed': False,
            'reason': 'Unknown direction',
            'failed_80_percent': False
        }

    # Get 80% level for monitoring
    retracement_80 = calculate_80_percent_retracement(f_price, e_price, direction)
    print(f"  80% failure level: ${retracement_80:.2f}")

    # Search for first target hit after S, while monitoring for 80% failure
    remaining_prices = prices[bounce_idx + 1:]
    remaining_dates = dates[bounce_idx + 1:]

    if not remaining_prices:
        return {
            'pattern_completed': False,
            'target_price': extension_236,
            'direction': direction,
            'reason': 'No data after S bounce',
            'failed_80_percent': False
        }

    for i, price in enumerate(remaining_prices):
        current_timestamp = remaining_dates[i]

        # First check for 80% failure
        if direction == 'up' and price <= retracement_80:
            print(
                f"    ❌ 80% FAILURE during target approach: ${price:.2f} <= ${retracement_80:.2f} at {current_timestamp}")
            return {
                'pattern_completed': False,
                'target_price': extension_236,
                'direction': direction,
                'reason': '80% retracement failure during target approach',
                'failed_80_percent': True,
                'failure_point': (current_timestamp, price),
                'failure_level': retracement_80
            }
        elif direction == 'down' and price >= retracement_80:
            print(
                f"    ❌ 80% FAILURE during target approach: ${price:.2f} >= ${retracement_80:.2f} at {current_timestamp}")
            return {
                'pattern_completed': False,
                'target_price': extension_236,
                'direction': direction,
                'reason': '80% retracement failure during target approach',
                'failed_80_percent': True,
                'failure_point': (current_timestamp, price),
                'failure_level': retracement_80
            }

        # Then check for target completion
        if direction == 'up' and price >= extension_236:
            t_timestamp = current_timestamp
            t_price = price
            print(f"    ✅ TARGET HIT at {t_timestamp}, ${t_price:.2f}")
            return {
                'pattern_completed': True,
                'completion_point': (t_timestamp, t_price),
                'target_price': extension_236,
                'actual_price': t_price,
                'accuracy': abs(t_price - extension_236),
                'direction': direction,
                'fe_move_distance': fe_move_distance,
                'reason': 'Target reached (uptrend)',
                'failed_80_percent': False
            }
        elif direction == 'down' and price <= extension_236:
            t_timestamp = current_timestamp
            t_price = price
            print(f"    ✅ TARGET HIT at {t_timestamp}, ${t_price:.2f}")
            return {
                'pattern_completed': True,
                'completion_point': (t_timestamp, t_price),
                'target_price': extension_236,
                'actual_price': t_price,
                'accuracy': abs(t_price - extension_236),
                'direction': direction,
                'fe_move_distance': fe_move_distance,
                'reason': 'Target reached (downtrend)',
                'failed_80_percent': False
            }

    print(f"    ⏳ Target not yet reached, no 80% failure detected")
    return {
        'pattern_completed': False,
        'target_price': extension_236,
        'direction': direction,
        'fe_move_distance': fe_move_distance,
        'reason': 'Target -23.6% not reached yet, pattern still valid',
        'failed_80_percent': False
    }


def plot_pattern_with_completion_analysis(analysis, prices, dates, min_change=0.01):
    """
    Professional visualization showing F-E-S pattern with completion target and CORRECTED 80% failure level
    """
    pattern = analysis['pattern']
    direction = pattern.get('direction', 'unknown')

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
    ax.text(d_timestamp, f_price, 'F (100%)', ha='center', va='bottom', fontsize=12,
            fontweight='bold', color='#2CA02C', zorder=16)

    # E point marking as 0% - E should never be marked as failed
    if final_e_point:
        e_timestamp, e_price = final_e_point

        # E point is always valid - it's the reference point
        e_color = '#228B22'  # Green for E point
        e_marker = 'D'
        e_size = 18

        ax.plot(e_timestamp, e_price, e_marker, color=e_color, markersize=e_size, zorder=15,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(e_timestamp, e_price, 'E (0%)', ha='center', va='top', fontsize=12,
                fontweight='bold', color=e_color, zorder=16)

    # S level (50% retracement) - Show even for failed patterns
    if analysis.get('s_level'):
        s_level = analysis['s_level']
        # Different styling for failed vs valid patterns
        if analysis.get('failed_80_percent', False):
            s_color = '#FF6B35'  # Orange-red for failed patterns
            s_alpha = 0.7
        else:
            s_color = '#9467BD'  # Purple for valid patterns
            s_alpha = 0.9
        ax.axhline(y=s_level, color=s_color, linewidth=4, alpha=s_alpha, zorder=3)

    # CORRECTED 80% RETRACEMENT LEVEL - Between F and E, closer to F
    retracement_80 = analysis.get('retracement_80_level') or analysis.get('validation_info', {}).get(
        'retracement_80_level')
    if retracement_80 and final_e_point:
        ax.axhline(y=retracement_80, color='#FF4444', linestyle='-.', linewidth=4, alpha=0.9, zorder=3)
        ax.text(d_timestamp, retracement_80, '80% Retracement', ha='left', va='bottom', fontsize=10,
                fontweight='bold', color='#FF4444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCCCC', alpha=0.8))

    # COMPLETION TARGET LINE
    if analysis.get('completion_target'):
        target_price = analysis['completion_target']
        ax.axhline(y=target_price, color='#FF6B35', linestyle=':', linewidth=3, alpha=0.9, zorder=3)

    # S bounce point - Show even for failed patterns
    validation_info = analysis.get('validation_info', {})
    if validation_info.get('bounce_point'):
        bounce_timestamp, bounce_price = validation_info['bounce_point']

        # Different styling for failed vs valid patterns
        if analysis.get('failed_80_percent', False):
            # Failed pattern - show S bounce in red/orange
            bounce_color = '#FF6B35'  # Orange-red for failed patterns
            bounce_text = 'S (50%) - FAILED'
            bounce_size = 18
        else:
            # Valid pattern - show S bounce in purple
            bounce_color = '#DC143C'  # Purple for valid patterns
            bounce_text = 'S (50%)'
            bounce_size = 20

        ax.plot(bounce_timestamp, bounce_price, '*', color=bounce_color, markersize=bounce_size, zorder=20,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(bounce_timestamp, bounce_price, bounce_text, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', zorder=21,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bounce_color, alpha=0.8))

    # 80% FAILURE POINT - Show where pattern actually failed
    if analysis.get('failed_80_percent', False):
        failure_info = analysis.get('failure_info', {}) or analysis.get('validation_info', {}).get('failure_info', {})
        if failure_info.get('failure_point'):
            fail_timestamp, fail_price = failure_info['failure_point']
            fail_timestamp = pd.to_datetime(fail_timestamp)
            ax.plot(fail_timestamp, fail_price, 'X', color='#FF0000', markersize=28, zorder=22,
                    markeredgecolor='white', markeredgewidth=4)
            ax.text(fail_timestamp, fail_price, 'EXCESSIVE\nRETRACEMENT', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='#FF0000', zorder=23,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFAAAA', alpha=0.8))

    # T point (completion extreme)
    completion_info = analysis.get('completion_info', {})
    if completion_info.get('pattern_completed') and completion_info.get('completion_point'):
        comp_timestamp, comp_price = completion_info['completion_point']
        comp_timestamp = pd.to_datetime(comp_timestamp)
        ax.plot(comp_timestamp, comp_price, '*', color='#FFD700', markersize=24, zorder=22,
                markeredgecolor='white', markeredgewidth=3)
        ax.text(comp_timestamp, comp_price, 'T (-23.6%)', ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', zorder=23)

    # Draw F-E connection line
    if final_e_point:
        line_color = '#FF6666' if analysis.get('failed_80_percent', False) else '#17BECF'
        line_style = ':' if analysis.get('failed_80_percent', False) else '--'
        ax.plot([d_timestamp, final_e_point[0]], [f_price, final_e_point[1]],
                color=line_color, linewidth=3, alpha=0.8, linestyle=line_style, zorder=8)

    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2, label='Price'),
        plt.Line2D([0], [0], color='#2CA02C', linestyle='--', linewidth=3, label=f'F Level (100%): ${f_price:.0f}'),
    ]

    if analysis.get('s_level'):
        s_level = analysis['s_level']
        if analysis.get('failed_80_percent', False):
            legend_elements.append(
                plt.Line2D([0], [0], color='#9467BD', linewidth=4, label=f'S Level (50%) - FAILED: ${s_level:.0f}')
            )
        else:
            legend_elements.append(
                plt.Line2D([0], [0], color='#9467BD', linewidth=4, label=f'S Level (50%): ${s_level:.0f}')
            )

    if retracement_80:
        legend_elements.append(
            plt.Line2D([0], [0], color='#FF4444', linestyle='-.', linewidth=4,
                       label=f'80% Retracement: ${retracement_80:.0f}')
        )

    if analysis.get('completion_target'):
        target_price = analysis['completion_target']
        legend_elements.append(
            plt.Line2D([0], [0], color='#FF6B35', linestyle=':', linewidth=3,
                       label=f'Target (-23.6%): ${target_price:.0f}')
        )

    if 'E2' in analysis:
        e2_timestamp, e2_price = analysis['E2']
        ax.scatter(e2_timestamp, e2_price, c='orange', marker='x', s=80, label='E2 (weaker)')
        ax.text(e2_timestamp, e2_price, "E2", color="orange", fontsize=10, weight="bold")

        # Draw F–E2 dashed line
        f_price = analysis.get('f_level')
        if f_price:
            ax.plot([analysis['d_timestamp'], e2_timestamp],
                    [f_price, e2_price],
                    linestyle="--", color="orange", alpha=0.8, label="F–E2 (weaker)")

    legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
                       fontsize=11, facecolor='white', edgecolor='gray')

    # Format chart
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')

    # Enhanced title
    if analysis.get('failed_80_percent', False):
        title = f'FAILED {direction.upper()} Pattern - Excessive Retracement\nPrice exceeded 80% retracement level'
        title_color = '#FF0000'
    elif analysis.get('is_valid', False):
        if analysis.get('pattern_completed', False):
            title = f'COMPLETED {direction.upper()} Pattern\nF-E-S-T Pattern (T at -23.6% extension)'
            title_color = '#FFD700'
        else:
            title = f'VALID {direction.upper()} Pattern (Uncompleted)\nF-E-S Pattern Confirmed, Awaiting T'
            title_color = '#2CA02C'
    else:
        title = f'INVALID {direction.upper()} Pattern\nNo Valid F-E-S Pattern Found'
        title_color = '#D62728'

    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='top', color=title_color)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Console output
    print("=" * 70)
    if analysis.get('failed_80_percent', False):
        print("PATTERN STATUS: FAILED - EXCESSIVE RETRACEMENT")
        failure_info = analysis.get('failure_info', {}) or analysis.get('validation_info', {}).get('failure_info', {})
        print(f"  80% Retracement Level: ${retracement_80:.2f}")

        # Show S bounce information for failed patterns
        validation_info = analysis.get('validation_info', {})
        if validation_info.get('bounce_point'):
            bounce_timestamp, bounce_price = validation_info['bounce_point']
            print(f"  S Bounce Found: {bounce_timestamp} at ${bounce_price:.2f}")
            print(f"  S Level (50%): ${validation_info.get('s_level', 0):.2f}")
            print(f"  ⚠️  Pattern had valid S bounce but failed due to excessive retracement")

        print(f"  Pattern failed due to excessive retracement beyond 80% level")
        if failure_info.get('failure_point'):
            fail_timestamp, fail_price = failure_info['failure_point']
            print(f"  Failure Point: {fail_timestamp} at ${fail_price:.2f}")
    elif analysis.get('is_valid', False):
        if analysis.get('pattern_completed', False):
            print("PATTERN STATUS: VALID + COMPLETED")
            comp_info = analysis['completion_info']
            print(f"  E Point (0%): ${final_e_point[1]:.2f}")
            print(f"  S Bounce (50%): ${analysis.get('s_level', 0):.2f}")
            print(f"  80% Safety Level: ${retracement_80:.2f}")
            print(f"  T Target (-23.6%): ${comp_info['target_price']:.2f}")
            print(f"  T Point: ${comp_info['actual_price']:.2f}")
        else:
            print("PATTERN STATUS: VALID (UNCOMPLETED)")
            print(f"  E Point (0%): ${final_e_point[1]:.2f}")
            print(f"  S Bounce (50%): ${analysis.get('s_level', 0):.2f}")
            print(f"  80% Safety Level: ${retracement_80:.2f}")
            print(f"  T Target (-23.6%): ${analysis['completion_target']:.2f}")
    else:
        print("PATTERN STATUS: INVALID")
    print("=" * 70)


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

    ax.set_xlim(min(subset_dates), max(subset_dates))
    ax.set_ylim(min(subset_prices) * 0.95, max(subset_prices) * 1.05)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def recycle_failed_pattern_with_same_F(prices, dates, f_price, failed_e_price, failed_e_timestamp,
                                       direction, min_threshold_pct=1.0, failure_percentage=0.80):
    """
    After failure, keep F fixed and try to find a new E2 when price pushes far enough from F again.
    Mark this new attempt as 'weaker'.
    """
    f_idx = find_index_from_timestamp(dates, failed_e_timestamp)  # continue search after failed E
    fe2_threshold = abs(failed_e_price - f_price) * (min_threshold_pct / 100)

    for i in range(f_idx + 1, len(prices)):
        current_price = prices[i]

        if direction == 'up' and current_price - f_price >= fe2_threshold:
            return {
                'E2': (dates[i], current_price),
                'status': 'secondary',
                'strength': 'weaker'
            }
        elif direction == 'down' and f_price - current_price >= fe2_threshold:
            return {
                'E2': (dates[i], current_price),
                'status': 'secondary',
                'strength': 'weaker'
            }

    return {
        'E2': (dates[i], current_price),
        'status': 'secondary',
        'strength': 'weaker'
    }


def recycle_failed_pattern_with_100_percent_retracement(prices, dates, f_price, failed_e_price, failed_e_timestamp,
                                                        direction, min_threshold_pct=1.0):
    """
    NEW: After 80% retracement failure, wait for 100% retracement and start over from same F to the highest.
    This creates a 'weaker' fan pattern.
    """
    f_idx = find_index_from_timestamp(dates, failed_e_timestamp)

    # Calculate 100% retracement level (price returns to F level)
    if direction == 'up':
        # For up pattern, 100% retracement means price falls back to F level
        target_retracement = f_price
        tolerance = f_price * 0.001  # 0.1% tolerance

        # Look for price to return to F level (100% retracement)
        for i in range(f_idx + 1, len(prices)):
            current_price = prices[i]
            if current_price <= target_retracement + tolerance:
                # Found 100% retracement, now look for new E2 (highest after retracement)
                retracement_idx = i
                print(f"    ♻️ Found 100% retracement at {dates[i]}, ${current_price:.2f}")

                # Search for new E2 (highest point after retracement)
                search_prices = prices[retracement_idx:]
                search_dates = dates[retracement_idx:]

                if search_prices:
                    max_price = max(search_prices)
                    max_idx = search_prices.index(max_price)
                    new_e2_timestamp = search_dates[max_idx]
                    new_e2_price = max_price

                    # Check if new E2 is far enough from F
                    fe2_distance = new_e2_price - f_price
                    min_distance = f_price * (min_threshold_pct / 100)

                    if fe2_distance >= min_distance:
                        return {
                            'E2': (new_e2_timestamp, new_e2_price),
                            'status': 'recovered_100_percent',
                            'strength': 'weaker',
                            'retracement_point': (dates[retracement_idx], current_price),
                            'retracement_type': '100_percent',
                            'fe2_distance': fe2_distance,
                            'min_distance_required': min_distance
                        }
                    else:
                        print(f"    ❌ New E2 at ${new_e2_price:.2f} too close to F (${f_price:.2f})")
                        return None
                break

    elif direction == 'down':
        # For down pattern, 100% retracement means price rises back to F level
        target_retracement = f_price
        tolerance = f_price * 0.001  # 0.1% tolerance

        # Look for price to return to F level (100% retracement)
        for i in range(f_idx + 1, len(prices)):
            current_price = prices[i]
            if current_price >= target_retracement - tolerance:
                # Found 100% retracement, now look for new E2 (lowest after retracement)
                retracement_idx = i
                print(f"    ♻️ Found 100% retracement at {dates[i]}, ${current_price:.2f}")

                # Search for new E2 (lowest point after retracement)
                search_prices = prices[retracement_idx:]
                search_dates = dates[retracement_idx:]

                if search_prices:
                    min_price = min(search_prices)
                    min_idx = search_prices.index(min_price)
                    new_e2_timestamp = search_dates[min_idx]
                    new_e2_price = min_price

                    # Check if new E2 is far enough from F
                    fe2_distance = f_price - new_e2_price
                    min_distance = f_price * (min_threshold_pct / 100)

                    if fe2_distance >= min_distance:
                        return {
                            'E2': (new_e2_timestamp, new_e2_price),
                            'status': 'recovered_100_percent',
                            'strength': 'weaker',
                            'retracement_point': (dates[retracement_idx], current_price),
                            'retracement_type': '100_percent',
                            'fe2_distance': fe2_distance,
                            'min_distance_required': min_distance
                        }
                    else:
                        print(f"    ❌ New E2 at ${new_e2_price:.2f} too close to F (${f_price:.2f})")
                        return None
                break

    return None


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
        'recovered_patterns': 0,
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
            if extension_analysis.get('recovery_type') == '100_percent_retracement':
                trailing_summary['recovered_patterns'] += 1
                print(
                    f"\n🔄 PATTERN {i + 1} RECOVERED - 100% retracement recovery successful after {len(trailing_attempts)} attempts")
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

        # Show recovery information
        if extension_analysis.get('recovery_type') == '100_percent_retracement':
            print(f"    Recovery: 100% retracement recovery successful")
            if extension_analysis.get('E2'):
                e2_timestamp, e2_price = extension_analysis['E2']
                print(f"    New E2: {e2_timestamp} at ${e2_price:.2f}")
                print(f"    Pattern strength: {extension_analysis.get('strength', 'unknown')}")
        elif extension_analysis.get('recovery_type') == 'failed':
            print(f"    Recovery: 100% retracement recovery failed")

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
    print(f"  Recovered patterns (100% retracement): {trailing_summary['recovered_patterns']}")
    print(f"  Failed patterns: {trailing_summary['failed_patterns']}")
    total_successful = trailing_summary['successful_patterns'] + trailing_summary['recovered_patterns']
    print(f"  Total successful + recovered: {total_successful}")
    print(
        f"  Success rate: {(trailing_summary['successful_patterns'] / trailing_summary['total_patterns'] * 100):.1f}%")
    print(
        f"  Success + Recovery rate: {(total_successful / trailing_summary['total_patterns'] * 100):.1f}%")
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
        "recovered_extensions": trailing_summary['recovered_patterns'],
        "failed_extensions": trailing_summary['failed_patterns'],
        "threshold_used": min_threshold_pct,
        "strategy": "trailing_with_recovery",
        "trailing_summary": trailing_summary,
        "ready_for_fan_extension": True
    }


if __name__ == "__main__":
    # Example usage with trailing strategy
    date_to_analyze = "2025-08-23"
    params = load_parameters()
    min_change = params.get('min_change', 0.001)
    min_threshold_pct = params.get('min_threshold_pct', 0.001)  # Default 1%

    # Set to True if you want to see plots
    show_plots = True

    print(f"🚀 Starting trailing E point analysis for {date_to_analyze}")
    results = get_completed_patterns_for_date(date_to_analyze, min_change, min_threshold_pct)

    if results["status"] == "success":
        print(f"\n🎯 TRAILING STRATEGY WITH RECOVERY COMPLETED!")
        print(f"Found {results['completed_count']} completed patterns")
        print(f"Valid extensions: {results['valid_extensions']}")
        print(f"Recovered extensions (100% retracement): {results['recovered_extensions']}")
        print(f"Failed extensions: {results['failed_extensions']}")
        total_successful = results['valid_extensions'] + results['recovered_extensions']
        print(f"Total successful + recovered: {total_successful}")
        print(f"Success rate: {(results['valid_extensions'] / results['completed_count'] * 100):.1f}%")
        print(f"Success + Recovery rate: {(total_successful / results['completed_count'] * 100):.1f}%")
        print(f"Strategy: {results['strategy']}")

        if show_plots and results['extension_analyses']:
            print("\n📈 Generating plots for patterns with trailing analysis...")
            for i, analysis in enumerate(results['extension_analyses']):
                if analysis.get('lowest_after_d') or analysis.get('highest_after_d'):
                    print(
                        f"\nPlotting pattern {i + 1} (trailing attempts: {len(analysis.get('trailing_attempts', []))})...")
                    plot_pattern_with_completion_analysis(analysis, results['prices'], results['dates'], min_change)
    else:
        print(f"❌ Error: {results['status']}")