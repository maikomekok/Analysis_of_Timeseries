import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def detect_trend(prices):
    """Automatically detects if the trend is upward or downward."""
    if prices[-1] > prices[0]:
        return "up"
    else:
        return "down"


def find_best_starting_point(prices, window=50):
    """
    Find the best starting point for Fibonacci retracement analysis.
    Uses a rolling window to find significant local minima or maxima.

    Args:
    prices (list): Price series
    window (int): Size of the rolling window to search for local extrema

    Returns:
    int: Index of the best starting point
    """
    direction = detect_trend(prices)

    # Use numpy for efficient rolling window calculation
    prices_array = np.array(prices)

    if direction == "up":
        # For uptrend, look for a local minimum
        local_mins = []
        for i in range(window, len(prices) - window):
            local_window = prices_array[i - window:i + window]
            if prices_array[i] == np.min(local_window):
                local_mins.append((i, prices_array[i]))

        # Select the local minimum with the lowest price
        if local_mins:
            return min(local_mins, key=lambda x: x[1])[0]
    else:
        # For downtrend, look for a local maximum
        local_maxs = []
        for i in range(window, len(prices) - window):
            local_window = prices_array[i - window:i + window]
            if prices_array[i] == np.max(local_window):
                local_maxs.append((i, prices_array[i]))

        # Select the local maximum with the highest price
        if local_maxs:
            return max(local_maxs, key=lambda x: x[1])[0]

    # Fallback to a point near the start if no suitable point found
    return len(prices) // 4


def find_retracement_extension(prices, threshold=0.3, nested_threshold=0.2):
    """
    Find Fibonacci retracement points in price data dynamically.
    Improved version with better starting point selection.
    """
    # Find a better starting point
    A = find_best_starting_point(prices)

    direction = detect_trend(prices[A:])

    # Adjust prices to start from the new A point
    prices_subset = prices[A:]

    B, C, D = None, None, None

    # Find B (Swing High/Low)
    if direction == "up":
        B = max(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])  # Find highest point
    else:
        B = min(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])  # Find lowest point

    if B is None:
        return None

    # Find C based on retracement threshold
    if direction == "up":
        move = prices_subset[B] - prices_subset[0]
        retracement_threshold = prices_subset[B] - (move * threshold)
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] <= retracement_threshold:
                C = j
                break
    else:
        move = prices_subset[0] - prices_subset[B]
        retracement_threshold = prices_subset[B] + (move * threshold)
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] >= retracement_threshold:
                C = j
                break

    if C is None:
        return None

    # Find D (Final Move Completion)
    if direction == "up":
        D = max(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])  # Find new high
    else:
        D = min(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])  # Find new low

    # Adjust indices to original price series
    return {
        "A": (A + 0, prices[A + 0]),
        "B": (A + B, prices[A + B]),
        "C": (A + C, prices[A + C]),
        "D": (A + D, prices[A + D]),
        "direction": direction
    }


def plot_fib_retracement(prices, result):
    if not result:
        print("No valid retracement found.")
        return

    direction = result.get("direction", "up")
    plt.figure(figsize=(14, 8))
    plt.plot(prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series')
    move_type = "Upward" if direction == "up" else "Downward"
    plt.title(f'Fibonacci Retracement Analysis - {move_type} Move', fontsize=16)

    # Mark key points
    points = {'A': ('black', result['A']), 'B': ('red', result['B']), 'C': ('green', result['C']),
              'D': ('blue', result['D'])}
    for label, (color, (idx, price)) in points.items():
        plt.scatter(idx, price, color=color, s=100, zorder=3)
        plt.text(idx, price, label, fontsize=14, fontweight='bold', ha='right', va='bottom', color=color)

    # Plot Fibonacci retracement levels
    A, B, C = result['A'][1], result['B'][1], result['C'][1]
    main_move = B - A if direction == "up" else A - B
    fib_levels = [0, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
    for level in fib_levels:
        fib_price = B - (main_move * level) if direction == "up" else B + (main_move * level)
        plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8, verticalalignment='center', color='gray')

    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def analyze_move(result):
    """Analyze the retracement and extension ratios."""
    if result is None:
        return "No valid pattern found."

    A, B, C, D = result['A'][1], result['B'][1], result['C'][1], result['D'][1]
    direction = result['direction']

    if direction == "up":
        move_distance = B - A
        retracement_ratio = (B - C) / move_distance if move_distance != 0 else 0
        extension_ratio = (D - C) / move_distance if move_distance != 0 else None
    else:
        move_distance = A - B
        retracement_ratio = (C - B) / move_distance if move_distance != 0 else 0
        extension_ratio = (C - D) / move_distance if move_distance != 0 else None

    # Classify move
    if retracement_ratio < 0.382:
        status = "Progressive move (shallow retracement)"
    elif retracement_ratio >= 0.618:
        status = "Failed move (deep retracement)"
    else:
        status = "Successful move (moderate retracement)"

    if extension_ratio is not None:
        if extension_ratio > 1.0:
            status += " with strong extension"
        elif extension_ratio < 0.5:
            status += " with weak extension"

    return status


# Example usage
def main():
    # Load BTC data
    df = pd.read_csv("BTCUSDT-hourly-historical-price.csv")
    Prices = df["close"].dropna().iloc[200:800]
    prices = Prices.tolist()

    # Run Analysis on BTC Data
    result = find_retracement_extension(prices)

    if result:
        plot_fib_retracement(prices, result)
        analysis = analyze_move(result)

        print(f"Detected trend: {result['direction']}")
        print(f"Identified points:")
        for point, (idx, price) in result.items():
            if point != 'direction':
                print(f"{point}: Index {idx}, Price {price}")
        print(f"Analysis: {analysis}")

        plt.show()
    else:
        print("Could not find a valid Fibonacci retracement pattern.")


if __name__ == "__main__":
    main()