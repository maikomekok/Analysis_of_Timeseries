import matplotlib.pyplot as plt
import numpy as np


def find_retracement_extension(prices, threshold=0.3, nested_threshold=0.2, direction="up"):
    """
    Find Fibonacci retracement points in price data.

    Parameters:
    - prices: list of price values
    - threshold: retracement threshold (default 0.3 or 30%)
    - nested_threshold: threshold for nested patterns (default 0.2 or 20%)
    - direction: "up" for upward move or "down" for downward move

    Returns:
    - Dictionary with identified points
    """
    A = 0
    B, C, D = None, None, None

    if direction == "up":
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                B = i
                break
    else:
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                B = i
                break

    if B is None:
        return None

    if direction == "up":
        move = prices[B] - prices[A]
        retracement_threshold = prices[B] - (move * threshold)
        for j in range(B + 1, len(prices) - 1):
            if prices[j] <= retracement_threshold and prices[j] < prices[j - 1] and prices[j] < prices[j + 1]:
                C = j
                break
    else:
        move = prices[A] - prices[B]
        retracement_threshold = prices[B] + (move * threshold)
        for j in range(B + 1, len(prices) - 1):
            if prices[j] >= retracement_threshold and prices[j] > prices[j - 1] and prices[j] > prices[j + 1]:
                C = j
                break

    if C is None:
        return None

    if direction == "up":
        for k in range(C + 1, len(prices) - 1):
            if prices[k] > prices[k - 1] and prices[k] > prices[k + 1]:
                D = k
                break
    else:
        for k in range(C + 1, len(prices) - 1):
            if prices[k] < prices[k - 1] and prices[k] < prices[k + 1]:
                D = k
                break

    result = {
        "A": (A, prices[A]),
        "B": (B, prices[B]),
        "C": (C, prices[C]),
        "threshold": threshold,
        "nested_threshold": nested_threshold,
        "direction": direction
    }

    if D is not None:
        result["D"] = (D, prices[D])

    return result


def plot_fib_retracement(prices, result):
    if not result:
        print("No valid retracement found.")
        return

    direction = result.get("direction", "up")
    plt.figure(figsize=(14, 8))
    plt.plot(prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series')
    move_type = "Upward" if direction == "up" else "Downward"
    plt.title(f'Fibonacci Retracement Analysis - {move_type} Move', fontsize=16)

    points = {'A': ('black', result['A']), 'B': ('red', result['B']), 'C': ('green', result['C'])}
    if 'D' in result:
        points['D'] = ('blue', result['D'])

    for label, (color, (idx, price)) in points.items():
        plt.scatter(idx, price, color=color, s=100, zorder=3)
        va_pos = 'bottom' if label in ['A', 'C'] else 'top'
        plt.text(idx, price, label, fontsize=14, fontweight='bold', ha='right', va=va_pos, color=color)

    # Plot Fibonacci retracement lines
    if direction == "up":
        main_move = result['B'][1] - result['A'][1]
        # Use custom levels: 76.4% instead of 78.6%, and add -23.6% and -61.8%
        fib_levels = [0, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
        for level in fib_levels:
            fib_price = result['B'][1] - (main_move * level)
            plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8, verticalalignment='center', color='gray')
    else:
        main_move = result['A'][1] - result['B'][1]
        fib_levels = [0, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
        for level in fib_levels:
            fib_price = result['B'][1] + (main_move * level)
            plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8, verticalalignment='center', color='gray')

    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def analyze_move(result, prices):
    """
    Analyze the move based on retracement and extension ratios.
    For an upward move, we compute:
      - retracement_ratio = (B - C) / (B - A)
      - extension_ratio = (D - C) / (B - A) if D exists

    Adjust thresholds to classify the move:
      - If retracement_ratio < 0.382: Progressive move (shallow retracement)
      - If retracement_ratio >= 0.618: Failed move (deep retracement)
      - Otherwise: Successful move

    Optionally, strong extension (extension_ratio > 1.0) adds further confirmation.
    """
    if result is None:
        return "No valid pattern found."

    direction = result.get("direction", "up")
    if direction == "up":
        A = result['A'][1]
        B = result['B'][1]
        C = result['C'][1]
        move_distance = B - A
        retracement_ratio = (B - C) / move_distance if move_distance != 0 else 0

        extension_ratio = None
        if "D" in result:
            D = result["D"][1]
            extension_ratio = (D - C) / move_distance

        # Define thresholds (these are examples; adjust as needed)
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
    else:
        # For a downward move, flip the ratios
        A = result['A'][1]
        B = result['B'][1]
        C = result['C'][1]
        move_distance = A - B
        retracement_ratio = (C - B) / move_distance if move_distance != 0 else 0

        extension_ratio = None
        if "D" in result:
            D = result["D"][1]
            extension_ratio = (C - D) / move_distance

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


# Sample price data for upward and downward moves
up_prices = [10, 12, 15, 20, 25, 30, 28, 25, 20, 22, 28, 25, 30, 35, 40]
## Example of price data for an upward move that results in a failed move:
up_prices = [10, 15, 22, 30, 15, 18, 20, 25, 28, 27, 30]

down_prices = [40, 38, 35, 30, 25, 20, 22, 25, 30, 28, 22, 25, 20, 15, 10]

thresh = 0.3
nested_thresh = 0.2

# Analyze upward move
up_result = find_retracement_extension(up_prices, threshold=thresh, nested_threshold=nested_thresh, direction="up")
plot_fib_retracement(up_prices, up_result)
up_analysis = analyze_move(up_result, up_prices)
print("Upward Move - Identified points:", up_result)
print("Upward Move Analysis:", up_analysis)

# Analyze downward move
down_result = find_retracement_extension(down_prices, threshold=thresh, nested_threshold=nested_thresh,
                                         direction="down")
plot_fib_retracement(down_prices, down_result)
down_analysis = analyze_move(down_result, down_prices)
print("Downward Move - Identified points:", down_result)
print("Downward Move Analysis:", down_analysis)

plt.show()
