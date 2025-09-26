import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def load_parameters():
    """Load parameters from JSON"""
    try:
        with open('parameters.json', 'r') as f:
            return json.load(f)
    except:
        return {'min_change': 0.001}


def zigzag_extremes(ohlc_data, reversal_pct=0.001):
    """
    ZigZag implementation - finds all significant extremes
    """
    closes = np.array(ohlc_data['close'], dtype=float)
    highs = np.array(ohlc_data['high'], dtype=float)
    lows = np.array(ohlc_data['low'], dtype=float)
    extremes = []

    if len(closes) == 0:
        return extremes

    print(f"=== ZigZag Extremes Detection ===")
    print(f"Reversal threshold: {reversal_pct * 100:.3f}%")

    # Always start with the first point
    first_high = highs[0]
    first_low = lows[0]

    # Check if first point is a significant high or low
    look_ahead = min(10, len(closes) - 1)
    if look_ahead > 0:
        if first_high >= max(highs[1:look_ahead + 1]):
            extremes.append((0, first_high, 'high'))
            print(f"  First point HIGH: idx 0 = ${first_high:.2f}")
        elif first_low <= min(lows[1:look_ahead + 1]):
            extremes.append((0, first_low, 'low'))
            print(f"  First point LOW: idx 0 = ${first_low:.2f}")

    # Initialize tracking variables
    trend = None
    last_extreme_idx = 0
    last_extreme_price = extremes[0][1] if extremes else closes[0]

    # Track provisional peaks/troughs
    high_idx, high_price = 0, highs[0]
    low_idx, low_price = 0, lows[0]

    for i in range(1, len(closes)):
        # Update provisional extremes
        if highs[i] > high_price:
            high_price, high_idx = highs[i], i
        if lows[i] < low_price:
            low_price, low_idx = lows[i], i

        if trend is None:
            # Establish initial trend
            up_move = (high_price - last_extreme_price) / last_extreme_price if last_extreme_price > 0 else 0
            down_move = (last_extreme_price - low_price) / last_extreme_price if last_extreme_price > 0 else 0

            if up_move >= reversal_pct:
                trend = 'up'
                print(f"  Initial UP trend confirmed: {up_move * 100:.3f}%")
            elif down_move >= reversal_pct:
                trend = 'down'
                print(f"  Initial DOWN trend confirmed: {down_move * 100:.3f}%")

        elif trend == 'up':
            # Check for down reversal
            reversal = (high_price - lows[i]) / high_price if high_price > 0 else 0
            if reversal >= reversal_pct:
                extremes.append((high_idx, high_price, 'high'))
                print(f"  HIGH: idx {high_idx} = ${high_price:.2f}")
                trend = 'down'
                last_extreme_idx, last_extreme_price = high_idx, high_price
                low_idx, low_price = i, lows[i]

        elif trend == 'down':
            # Check for up reversal
            if low_price > 0:
                reversal = (highs[i] - low_price) / low_price
                if reversal >= reversal_pct:
                    extremes.append((low_idx, low_price, 'low'))
                    print(f"  LOW: idx {low_idx} = ${low_price:.2f}")
                    trend = 'up'
                    last_extreme_idx, last_extreme_price = low_idx, low_price
                    high_idx, high_price = i, highs[i]

    # Add final extreme
    if trend == 'up' and high_idx > last_extreme_idx:
        extremes.append((high_idx, high_price, 'high'))
        print(f"  Final HIGH: idx {high_idx} = ${high_price:.2f}")
    elif trend == 'down' and low_idx > last_extreme_idx:
        extremes.append((low_idx, low_price, 'low'))
        print(f"  Final LOW: idx {low_idx} = ${low_price:.2f}")

    print(f"Total ZigZag extremes: {len(extremes)}")
    return extremes


def find_absolute_global_extremes(ohlc_data, extremes):
    """
    Find THE single absolute global high and low from the entire dataset
    A point must be the absolute extreme, not just locally significant
    """
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    # Find absolute highest high in entire dataset
    absolute_high_price = max(highs)
    absolute_high_idx = highs.index(absolute_high_price)

    # Find absolute lowest low in entire dataset
    absolute_low_price = min(lows)
    absolute_low_idx = lows.index(absolute_low_price)

    print(f"\n=== Finding ABSOLUTE Global Extremes ===")
    print(f"Absolute HIGHEST HIGH in dataset: idx {absolute_high_idx} = ${absolute_high_price:.2f}")
    print(f"Absolute LOWEST LOW in dataset: idx {absolute_low_idx} = ${absolute_low_price:.2f}")

    # Verify these are also in our ZigZag extremes (they should be)
    global_high_confirmed = False
    global_low_confirmed = False

    for idx, price, type_ in extremes:
        if idx == absolute_high_idx and abs(price - absolute_high_price) < 0.01:
            global_high_confirmed = True
            print(f"  ✓ Absolute high CONFIRMED in ZigZag extremes")
        if idx == absolute_low_idx and abs(price - absolute_low_price) < 0.01:
            global_low_confirmed = True
            print(f"  ✓ Absolute low CONFIRMED in ZigZag extremes")

    if not global_high_confirmed:
        print(f"  ⚠ Absolute high NOT in ZigZag extremes - adding it")
    if not global_low_confirmed:
        print(f"  ⚠ Absolute low NOT in ZigZag extremes - adding it")

    return (absolute_high_idx, absolute_high_price), (absolute_low_idx, absolute_low_price)

def find_all_50_percent_c_points(ohlc_data, dates, A_idx, A_price, B_idx, B_price, direction):
    """
    Find ONLY the FIRST valid 50% retracement C point for a given AB move
    Returns list with only the first valid C point (to maintain compatibility)
    """
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    ab_move = abs(B_price - A_price)
    if ab_move == 0:
        return []

    # Calculate exact 50% retracement level
    if direction == 'up':  # A=low, B=high, looking for pullback
        retracement_50_level = B_price - (ab_move * 0.5)
        tolerance = ab_move * 0.002  # 2% tolerance
        print(
            f"      Looking for FIRST 50% pullback from ${B_price:.2f} to ${retracement_50_level:.2f} (±${tolerance:.2f})")

        # Search for FIRST price that touches 50% level
        for k in range(B_idx + 1, len(dates)):
            current_low = lows[k]
            current_high = highs[k]

            # Check if price actually touched the 50% level
            if (retracement_50_level - tolerance) <= current_low <= (retracement_50_level + tolerance) or \
                    (retracement_50_level - tolerance) <= current_high <= (retracement_50_level + tolerance):

                # Use the price closest to 50% level
                low_diff = abs(current_low - retracement_50_level)
                high_diff = abs(current_high - retracement_50_level)

                if low_diff <= high_diff:
                    C_idx, C_price = k, current_low
                else:
                    C_idx, C_price = k, current_high

                actual_retracement = ((B_price - C_price) / ab_move) * 100
                print(f"        ✓ FIRST C found: {dates[C_idx]} ${C_price:.2f} ({actual_retracement:.1f}% retracement)")
                return [(C_idx, C_price, actual_retracement)]  # Return list with single item

            # Stop if price moved too far beyond 88.6% (pattern failure)
            if current_low < B_price - (ab_move * 0.886):
                print(f"        Pattern search stopped - exceeded 88.6% at {dates[k]} ${current_low:.2f}")
                break

    else:  # direction == 'down': A=high, B=low, looking for bounce
        retracement_50_level = B_price + (ab_move * 0.5)
        tolerance = ab_move * 0.02
        print(
            f"      Looking for FIRST 50% bounce from ${B_price:.2f} to ${retracement_50_level:.2f} (±${tolerance:.2f})")

        # Search for FIRST price that touches 50% level
        for k in range(B_idx + 1, len(dates)):
            current_low = lows[k]
            current_high = highs[k]

            # Check if price actually touched the 50% level
            if (retracement_50_level - tolerance) <= current_low <= (retracement_50_level + tolerance) or \
                    (retracement_50_level - tolerance) <= current_high <= (retracement_50_level + tolerance):

                # Use the price closest to 50% level
                low_diff = abs(current_low - retracement_50_level)
                high_diff = abs(current_high - retracement_50_level)

                if high_diff <= low_diff:
                    C_idx, C_price = k, current_high
                else:
                    C_idx, C_price = k, current_low

                actual_retracement = ((C_price - B_price) / ab_move) * 100
                print(f"        ✓ FIRST C found: {dates[C_idx]} ${C_price:.2f} ({actual_retracement:.1f}% retracement)")
                return [(C_idx, C_price, actual_retracement)]  # Return list with single item

            # Stop if price moved too far beyond 88.6% (pattern failure)
            if current_high > B_price + (ab_move * 0.886):
                print(f"        Pattern search stopped - exceeded 88.6% at {dates[k]} ${current_high:.2f}")
                break

    print(f"      No valid 50% C point found")
    return []


def find_d_completion(ohlc_data, dates, C_idx, C_price, B_price, ab_move, direction):
    """
    Find D point using proper ABCD theory: 23.6% extension from B (beyond the AB move)
    This matches standard Fibonacci retracement placement: A-B with -23.6% extension
    """
    highs = ohlc_data['high']
    lows = ohlc_data['low']

    if direction == 'up':  # A=low, B=high, C=pullback, D=target high
        # D = B + 23.6% extension of AB move (beyond B)
        D_target = B_price + (ab_move * 0.236)
        print(f"    AB move: ${ab_move:.2f}")
        print(f"    D target: ${D_target:.2f} (B + 23.6% extension of AB move)")
        print(f"    B=${B_price:.2f} + 23.6% of ${ab_move:.2f} = ${D_target:.2f}")

        for m in range(C_idx + 1, len(dates)):
            # Pattern invalidation if goes back below C significantly
            if lows[m] < C_price * 0.98:
                print(f"    Pattern invalidated at {dates[m]} (broke below C)")
                return "failed"

            # D target reached - use FIRST touch
            if highs[m] >= D_target:
                print(f"    ✓ D target reached: {dates[m]} ${highs[m]:.2f}")
                return (m, highs[m])

    else:  # direction == 'down': A=high, B=low, C=bounce, D=target low
        # D = B - 23.6% extension of AB move (beyond B, further down)
        D_target = B_price - (ab_move * 0.236)
        print(f"    AB move: ${ab_move:.2f}")
        print(f"    D target: ${D_target:.2f} (B - 23.6% extension of AB move)")
        print(f"    B=${B_price:.2f} - 23.6% of ${ab_move:.2f} = ${D_target:.2f}")

        for m in range(C_idx + 1, len(dates)):
            # Pattern invalidation if goes back above C significantly
            if highs[m] > C_price * 1.02:
                print(f"    Pattern invalidated at {dates[m]} (broke above C)")
                return "failed"

            # D target reached - use FIRST touch
            if lows[m] <= D_target:
                print(f"    ✓ D target reached: {dates[m]} ${lows[m]:.2f}")
                return (m, lows[m])

    # Pattern still pending if we reach here
    return "pending"


def find_abcd_patterns_iterative(ohlc_data, dates):
    """
    CLEAN ITERATIVE ABCD Pattern Detection - Smart B Selection:
    1. Test FIRST B, if successful pattern found, look for NEXT logical B
    """
    params = load_parameters()
    min_change = params.get('min_change', 0.001)

    print(f"\n=== SMART B SELECTION ABCD PATTERN DETECTION ===")
    print(
        f"A = FIXED absolute global extreme, B = FIRST + NEXT logical only, C = FIRST 50% retracement, D = 127.2% targets")

    # Step 1: Get all ZigZag extremes
    extremes = zigzag_extremes(ohlc_data, reversal_pct=min_change)

    if len(extremes) < 2:
        print("Not enough ZigZag extremes")
        return []

    # Step 2: Find THE single absolute global extremes for A points
    global_high, global_low = find_absolute_global_extremes(ohlc_data, extremes)

    patterns = []

    # UPTREND PATTERNS: A(FIXED absolute low) -> smart B selection
    print(f"\n=== UPTREND PATTERNS (A = FIXED absolute lowest low) ===")

    A_idx, A_price = global_low
    print(f"\n✓ FIXED A (absolute lowest low): {dates[A_idx]} ${A_price:.2f}")

    # Get ALL highs after A for B candidates (from ZigZag extremes)
    all_b_candidates = [(idx, price) for idx, price, type_ in extremes
                        if type_ == 'high' and idx > A_idx]

    if not all_b_candidates:
        print("  No potential B candidates found for uptrend")
    else:
        print(f"\n  DEBUG: Found {len(all_b_candidates)} potential B candidates (all highs after A):")
        for i, (idx, price) in enumerate(all_b_candidates, 1):
            ab_move_pct = ((price - A_price) / A_price) * 100 if A_price > 0 else 0
            print(f"    Candidate {i}: {dates[idx]} ${price:.2f} (AB move: {ab_move_pct:.2f}%)")

        # SMART B SELECTION:
        # 1. FIRST B: First local high after A (chronologically)
        first_b = min(all_b_candidates, key=lambda x: x[0])

        selected_b_candidates = []
        selected_b_candidates.append(("FIRST", first_b[0], first_b[1]))
        print(f"  → Selected FIRST B: {dates[first_b[0]]} ${first_b[1]:.2f}")

        # Test first B to see if it has valid C and D
        ab_move = first_b[1] - A_price
        ab_move_pct = (ab_move / A_price) * 100 if A_price > 0 else 0

        if ab_move_pct >= min_change * 100:
            c_results = find_all_50_percent_c_points(ohlc_data, dates, A_idx, A_price, first_b[0], first_b[1], 'up')
            if c_results:
                C_idx, C_price, actual_retracement = c_results[0]
                d_result = find_d_completion(ohlc_data, dates, C_idx, C_price, first_b[1], ab_move, 'up')

                if d_result != "failed":
                    print(f"     ✓ FIRST B has valid pattern, looking for NEXT higher B")

                    # 2. NEXT HIGHER B: Among highs > first B, take the smallest one
                    higher_candidates = [(idx, price) for idx, price in all_b_candidates
                                         if price > first_b[1]]

                    if higher_candidates:
                        higher_candidates.sort(key=lambda x: x[1])  # Sort by price ascending
                        next_higher_b = higher_candidates[0]  # Take smallest among higher

                        selected_b_candidates.append(("NEXT_HIGHER", next_higher_b[0], next_higher_b[1]))
                        print(f"  → Selected NEXT HIGHER B: {dates[next_higher_b[0]]} ${next_higher_b[1]:.2f}")
                    else:
                        print(f"  → No higher B candidates found")
                else:
                    print(f"     ❌ FIRST B pattern failed, not looking for additional B")

        # Test each selected B candidate
        for b_num, (selection_type, B_idx, B_price) in enumerate(selected_b_candidates, 1):
            ab_move = B_price - A_price
            ab_move_pct = (ab_move / A_price) * 100 if A_price > 0 else 0

            print(f"\n  → Testing B#{b_num} ({selection_type}): {dates[B_idx]} ${B_price:.2f}")

            if ab_move_pct < min_change * 100:
                print(f"     ❌ SKIPPED: Move too small")
                continue

            c_results = find_all_50_percent_c_points(ohlc_data, dates, A_idx, A_price, B_idx, B_price, 'up')

            if not c_results:
                print(f"     ❌ No valid C point found")
                continue

            C_idx, C_price, actual_retracement = c_results[0]
            print(f"     ✓ Found C: {dates[C_idx]} ${C_price:.2f}")

            d_result = find_d_completion(ohlc_data, dates, C_idx, C_price, B_price, ab_move, 'up')

            pattern_data = {
                "direction": "up",
                "A": (dates[A_idx], A_price),
                "B": (dates[B_idx], B_price),
                "C": (dates[C_idx], C_price),
                "A_idx": A_idx, "B_idx": B_idx, "C_idx": C_idx,
                "ab_move_pct": ab_move_pct,
                "retracement_pct": actual_retracement,
                "b_selection_type": selection_type,
                "b_candidate": b_num
            }

            if d_result == "failed":
                pattern_data.update({"status": "failed", "D": None})
                print(f"     ❌ D search failed")
            elif d_result == "pending":
                pending_d_price = B_price + (ab_move * 0.236)
                pattern_data.update({
                    "status": "pending",
                    "D": (dates[-1], pending_d_price),
                    "D_idx": len(dates) - 1
                })
                print(f"     ⏳ Pattern pending")
            elif d_result is not None:
                D_idx, D_price = d_result
                pattern_data.update({
                    "status": "completed",
                    "D": (dates[D_idx], D_price),
                    "D_idx": D_idx
                })
                print(f"     ✅ Pattern completed!")

            patterns.append(pattern_data)

    # DOWNTREND PATTERNS: A(FIXED absolute high) -> smart B selection
    print(f"\n=== DOWNTREND PATTERNS (A = FIXED absolute highest high) ===")

    A_idx, A_price = global_high
    print(f"\n✓ FIXED A (absolute highest high): {dates[A_idx]} ${A_price:.2f}")

    # Get ALL lows after A for B candidates
    all_b_candidates = [(idx, price) for idx, price, type_ in extremes
                        if type_ == 'low' and idx > A_idx]

    if not all_b_candidates:
        print("  No potential B candidates found for downtrend")
    else:
        print(f"\n  DEBUG: Found {len(all_b_candidates)} potential B candidates (all lows after A):")
        for i, (idx, price) in enumerate(all_b_candidates, 1):
            ab_move_pct = ((A_price - price) / A_price) * 100 if A_price > 0 else 0
            print(f"    Candidate {i}: {dates[idx]} ${price:.2f} (AB move: {ab_move_pct:.2f}%)")

        # SMART B SELECTION:
        # 1. FIRST B: First local low after A (chronologically)
        first_b = min(all_b_candidates, key=lambda x: x[0])

        selected_b_candidates = []
        selected_b_candidates.append(("FIRST", first_b[0], first_b[1]))
        print(f"  → Selected FIRST B: {dates[first_b[0]]} ${first_b[1]:.2f}")

        # Test first B to see if it has valid C and D
        ab_move = A_price - first_b[1]
        ab_move_pct = (ab_move / A_price) * 100 if A_price > 0 else 0

        if ab_move_pct >= min_change * 100:
            c_results = find_all_50_percent_c_points(ohlc_data, dates, A_idx, A_price, first_b[0], first_b[1], 'down')
            if c_results:
                C_idx, C_price, actual_retracement = c_results[0]
                d_result = find_d_completion(ohlc_data, dates, C_idx, C_price, first_b[1], ab_move, 'down')

                if d_result != "failed":
                    print(f"     ✓ FIRST B has valid pattern, looking for NEXT lower B")

                    # 2. NEXT LOWER B: Among lows < first B, take the highest one (closest to first B)
                    lower_candidates = [(idx, price) for idx, price in all_b_candidates
                                        if price < first_b[1]]

                    if lower_candidates:
                        lower_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by price descending
                        next_lower_b = lower_candidates[0]  # Take highest among lower

                        selected_b_candidates.append(("NEXT_LOWER", next_lower_b[0], next_lower_b[1]))
                        print(f"  → Selected NEXT LOWER B: {dates[next_lower_b[0]]} ${next_lower_b[1]:.2f}")
                    else:
                        print(f"  → No lower B candidates found")
                else:
                    print(f"     ❌ FIRST B pattern failed, not looking for additional B")

        # Test each selected B candidate
        for b_num, (selection_type, B_idx, B_price) in enumerate(selected_b_candidates, 1):
            ab_move = A_price - B_price
            ab_move_pct = (ab_move / A_price) * 100 if A_price > 0 else 0

            print(f"\n  → Testing B#{b_num} ({selection_type}): {dates[B_idx]} ${B_price:.2f}")

            if ab_move_pct < min_change * 100:
                print(f"     ❌ SKIPPED: Move too small")
                continue

            c_results = find_all_50_percent_c_points(ohlc_data, dates, A_idx, A_price, B_idx, B_price, 'down')

            if not c_results:
                print(f"     ❌ No valid C point found")
                continue

            C_idx, C_price, actual_retracement = c_results[0]
            print(f"     ✓ Found C: {dates[C_idx]} ${C_price:.2f}")

            d_result = find_d_completion(ohlc_data, dates, C_idx, C_price, B_price, ab_move, 'down')

            pattern_data = {
                "direction": "down",
                "A": (dates[A_idx], A_price),
                "B": (dates[B_idx], B_price),
                "C": (dates[C_idx], C_price),
                "A_idx": A_idx, "B_idx": B_idx, "C_idx": C_idx,
                "ab_move_pct": ab_move_pct,
                "retracement_pct": actual_retracement,
                "b_selection_type": selection_type,
                "b_candidate": b_num
            }

            if d_result == "failed":
                pattern_data.update({"status": "failed", "D": None})
                print(f"     ❌ D search failed")
            elif d_result == "pending":
                pending_d_price = B_price - (ab_move * 0.236)
                pattern_data.update({
                    "status": "pending",
                    "D": (dates[-1], pending_d_price),
                    "D_idx": len(dates) - 1
                })
                print(f"     ⏳ Pattern pending")
            elif d_result is not None:
                D_idx, D_price = d_result
                pattern_data.update({
                    "status": "completed",
                    "D": (dates[D_idx], D_price),
                    "D_idx": D_idx
                })
                print(f"     ✅ Pattern completed!")

            patterns.append(pattern_data)

    print(f"\n=== FINAL SMART B SELECTION RESULTS ===")
    print(f"Total patterns found: {len(patterns)}")

    completed = len([p for p in patterns if p['status'] == 'completed'])
    pending = len([p for p in patterns if p['status'] == 'pending'])
    failed = len([p for p in patterns if p['status'] == 'failed'])

    print(f"Completed: {completed}, Pending: {pending}, Failed: {failed}")

    for i, p in enumerate(patterns, 1):
        status_symbol = "✅" if p['status'] == 'completed' else "⏳" if p['status'] == 'pending' else "❌"
        print(
            f"{i}. {status_symbol} {p['direction'].upper()} {p['status'].upper()} (B#{p.get('b_candidate', '?')} - {p.get('b_selection_type', 'UNKNOWN')})")
        print(f"   A: {p['A'][0]} ${p['A'][1]:.2f} (FIXED)")
        print(f"   B: {p['B'][0]} ${p['B'][1]:.2f} ({p.get('b_selection_type', 'UNKNOWN')})")
        if p.get('C'):
            print(f"   C: {p['C'][0]} ${p['C'][1]:.2f} ({p['retracement_pct']:.1f}%)")
        if p.get('D'):
            print(f"   D: {p['D'][0]} ${p['D'][1]:.2f}")
        print()

    return patterns
def plot_abcd_patterns(ohlc_data, dates, patterns, extremes, max_patterns=10):
    """Plot ABCD patterns with proper point placement and improved timestamp visibility"""
    if not patterns:
        print("No patterns to plot")
        return

    # Convert dates to datetime objects for plotting
    plot_dates = []
    if isinstance(dates[0], str):
        try:
            # Try multiple date formats
            if 'T' in dates[0] or ' ' in dates[0]:

                plot_dates = pd.to_datetime(dates).tolist()
            else:
                plot_dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
        except:
            try:
                # Fallback to pandas
                plot_dates = pd.to_datetime(dates).tolist()
            except:
                # Last resort - use indices
                plot_dates = list(range(len(dates)))
                print("Warning: Could not parse dates, using indices instead")
    else:
        plot_dates = dates

    patterns_to_plot = patterns[:max_patterns]

    # Create larger figure for better visibility
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot price data
    highs = ohlc_data['high']
    lows = ohlc_data['low']
    closes = ohlc_data['close']

    ax.plot(plot_dates, closes, color='black', linewidth=1, alpha=0.7, label='Close Price')

    # Plot high-low range
    for i in range(len(plot_dates)):
        ax.plot([plot_dates[i], plot_dates[i]], [lows[i], highs[i]], color='gray', linewidth=0.5, alpha=0.3)

    # Plot ZigZag extremes
    extreme_dates = []
    extreme_prices = []
    extreme_colors = []

    for idx, price, type_ in extremes:
        if idx < len(plot_dates):
            extreme_dates.append(plot_dates[idx])
            extreme_prices.append(price)
            extreme_colors.append('red' if type_ == 'high' else 'green')

    if len(extreme_dates) > 1:
        ax.plot(extreme_dates, extreme_prices, color='purple', linewidth=1, alpha=0.6, linestyle='--',
                label='ZigZag Extremes')

    # Plot extremes as points
    for date, price, color in zip(extreme_dates, extreme_prices, extreme_colors):
        ax.scatter(date, price, color=color, s=30, alpha=0.8, zorder=5)

    # Plot ABCD patterns
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, pattern in enumerate(patterns_to_plot):
        color = colors[i % len(colors)]

        if pattern['status'] == 'completed':
            alpha = 0.8
            linewidth = 3
        elif pattern['status'] == 'pending':
            alpha = 0.6
            linewidth = 2
        else:  # failed
            alpha = 0.4
            linewidth = 1

        # Get pattern indices to use correct dates
        A_idx = pattern['A_idx']
        B_idx = pattern['B_idx']
        C_idx = pattern.get('C_idx')
        D_idx = pattern.get('D_idx')

        # Get prices
        A_price = pattern['A'][1]
        B_price = pattern['B'][1]
        C_price = pattern['C'][1] if pattern.get('C') else None
        D_price = pattern['D'][1] if pattern.get('D') else None

        # Use indices to get correct plot dates
        A_date = plot_dates[A_idx]
        B_date = plot_dates[B_idx]
        C_date = plot_dates[C_idx] if C_idx is not None else None
        D_date = plot_dates[D_idx] if D_idx is not None else None

        # Plot points with different markers for each type
        point_size = 100
        ax.scatter(A_date, A_price, color=color, s=point_size, alpha=0.9, zorder=10, marker='o',
                   edgecolors='white', linewidths=2, label=f'Pattern {i + 1} - {pattern["status"]}')
        ax.scatter(B_date, B_price, color=color, s=point_size, alpha=0.9, zorder=10, marker='s',
                   edgecolors='white', linewidths=2)

        if C_date and C_price:
            ax.scatter(C_date, C_price, color=color, s=point_size, alpha=0.9, zorder=10, marker='^',
                       edgecolors='white', linewidths=2)

        if D_date and D_price:
            marker = 'D' if pattern['status'] == 'completed' else 'x'
            ax.scatter(D_date, D_price, color=color, s=point_size, alpha=0.9, zorder=10, marker=marker,
                       edgecolors='white', linewidths=2)

        # Draw lines between points
        if pattern['status'] != 'failed' or pattern.get('C'):
            # AB line
            ax.plot([A_date, B_date], [A_price, B_price], color=color, linewidth=linewidth, alpha=alpha)

            # BC line (if C exists)
            if C_date and C_price:
                ax.plot([B_date, C_date], [B_price, C_price], color=color, linewidth=linewidth,
                        alpha=alpha, linestyle=':')

                # CD line (if D exists)
                if D_date and D_price:
                    linestyle = '-' if pattern['status'] == 'completed' else '--'
                    ax.plot([C_date, D_date], [C_price, D_price], color=color, linewidth=linewidth,
                            alpha=alpha, linestyle=linestyle)

        # Add labels with better positioning
        label_offset = 25

        # A label
        if pattern['direction'] == 'up':  # A is low point
            ax.annotate(f'A{i + 1}', (A_date, A_price), xytext=(0, -label_offset),
                        textcoords='offset points', ha='center', va='top', fontsize=12,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        else:  # A is high point
            ax.annotate(f'A{i + 1}', (A_date, A_price), xytext=(0, label_offset),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

        # B label
        if pattern['direction'] == 'up':  # B is high point
            ax.annotate(f'B{i + 1}', (B_date, B_price), xytext=(0, label_offset),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        else:  # B is low point
            ax.annotate(f'B{i + 1}', (B_date, B_price), xytext=(0, -label_offset),
                        textcoords='offset points', ha='center', va='top', fontsize=12,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

        # C label (if exists)
        if C_date and C_price:
            if pattern['direction'] == 'up':  # C is pullback (lower than B)
                ax.annotate(f'C{i + 1}', (C_date, C_price), xytext=(0, -label_offset),
                            textcoords='offset points', ha='center', va='top', fontsize=12,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
            else:  # C is bounce (higher than B)
                ax.annotate(f'C{i + 1}', (C_date, C_price), xytext=(0, label_offset),
                            textcoords='offset points', ha='center', va='bottom', fontsize=12,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

        # D label (if exists)
        if D_date and D_price:
            if pattern['direction'] == 'up':  # D is target high
                ax.annotate(f'D{i + 1}', (D_date, D_price), xytext=(0, label_offset),
                            textcoords='offset points', ha='center', va='bottom', fontsize=12,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
            else:  # D is target low
                ax.annotate(f'D{i + 1}', (D_date, D_price), xytext=(0, -label_offset),
                            textcoords='offset points', ha='center', va='top', fontsize=12,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

    ax.set_title(
        f'Iterative ABCD Patterns - A(Global Fixed), B(Iterate Locals), C(50% First Touch), D(127.2% Target)\n{len(patterns_to_plot)} patterns shown',
        fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # IMPROVED TIMESTAMP FORMATTING
    if isinstance(plot_dates[0], datetime):
        # Determine the time range to choose appropriate formatting
        time_span = plot_dates[-1] - plot_dates[0]

        if time_span.days > 7:
            # More than a week - show dates
            date_format = '%m-%d'
            interval = max(1, len(plot_dates) // 20)  # Show ~20 labels max
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
        elif time_span.days > 1:
            # More than a day - show date and hour
            date_format = '%m-%d %H:%M'
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span.total_seconds() / 3600 / 15))))
        else:
            # Less than a day - show hour:minute
            date_format = '%H:%M'
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span.total_seconds() / 3600 / 12))))

        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

        # Add minor ticks for better granularity
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    else:
        # If we're using indices, show fewer labels
        ax.set_xlabel('Time Index', fontsize=12)
        step = max(1, len(plot_dates) // 20)
        ax.set_xticks(range(0, len(plot_dates), step))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    ax.legend(loc='upper left', fontsize=8, ncol=2)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Extra space for rotated labels

    plt.show()

    # Print some debug info about timestamps
    print(f"\nTimestamp Debug Info:")
    print(f"Original date format: {type(dates[0])}")
    print(f"First few dates: {dates[:3]}")
    print(f"Plot date format: {type(plot_dates[0])}")
    if isinstance(plot_dates[0], datetime):
        print(f"Date range: {plot_dates[0]} to {plot_dates[-1]}")
        print(f"Time span: {plot_dates[-1] - plot_dates[0]}")

# Test function
def test_iterative_abcd(csv_file):
    """Test the iterative ABCD detection"""
    df = pd.read_csv(csv_file)
    ohlc_data = {
        'open': df['open'].tolist(),
        'high': df['high'].tolist(),
        'low': df['low'].tolist(),
        'close': df['close'].tolist()
    }
    dates = df['timestamp'].tolist()

    print(f"Testing iterative approach on {len(dates)} data points")
    print(f"Price range: ${min(ohlc_data['low']):.2f} - ${max(ohlc_data['high']):.2f}")

    patterns = find_abcd_patterns_iterative(ohlc_data, dates)

    if patterns:
        extremes = zigzag_extremes(ohlc_data, reversal_pct=0.001)
        plot_abcd_patterns(ohlc_data, dates, patterns, extremes, max_patterns=10)

    return patterns


if __name__ == "__main__":
    patterns = test_iterative_abcd("btc_1minute_data_1minute_2025-09-01.csv")
    print(f"\nFinal result: {len(patterns)} patterns found")

    completed = len([p for p in patterns if p['status'] == 'completed'])
    pending = len([p for p in patterns if p['status'] == 'pending'])
    failed = len([p for p in patterns if p['status'] == 'failed'])
    print(f"Completed: {completed}, Pending: {pending}, Failed: {failed}")