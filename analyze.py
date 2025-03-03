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

    # For upward move: Look for peak (B higher than A)
    # For downward move: Look for trough (B lower than A)
    if direction == "up":
        # Find point B (first peak)
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                B = i
                break
    else:  # direction == "down"
        # Find point B (first trough)
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                B = i
                break

    if B is None:
        return None

    # Find point C (retracement after B)
    if direction == "up":
        move = prices[B] - prices[A]
        retracement_threshold = prices[B] - (move * threshold)

        for j in range(B + 1, len(prices) - 1):
            if prices[j] <= retracement_threshold and prices[j] < prices[j - 1] and prices[j] < prices[j + 1]:
                C = j
                break
    else:  # direction == "down"
        move = prices[A] - prices[B]  # Note: A - B for downward move
        retracement_threshold = prices[B] + (move * threshold)

        for j in range(B + 1, len(prices) - 1):
            if prices[j] >= retracement_threshold and prices[j] > prices[j - 1] and prices[j] > prices[j + 1]:
                C = j
                break

    if C is None:
        return None

    # Find point D (next peak/trough after C)
    if direction == "up":
        for k in range(C + 1, len(prices) - 1):
            if prices[k] > prices[k - 1] and prices[k] > prices[k + 1]:
                D = k
                break
    else:  # direction == "down"
        for k in range(C + 1, len(prices) - 1):
            if prices[k] < prices[k - 1] and prices[k] < prices[k + 1]:
                D = k
                break

    # Find nested pattern (new B and new C) between C and D
    new_B, new_C = None, None

    if C is not None and D is not None:
        # Slice the prices between C and D to look for nested pattern
        sub_prices = prices[C:D + 1]

        if len(sub_prices) >= 3:  # Need at least 3 points for a pattern
            if direction == "up":
                # Find new B (highest price between C and D)
                new_B_idx = C + np.argmax(sub_prices)

                # Confirm new B is a peak
                if new_B_idx > C and new_B_idx < D:
                    if prices[new_B_idx] > prices[new_B_idx - 1] and prices[new_B_idx] > prices[new_B_idx + 1]:
                        new_B = new_B_idx

                        # Find new C (retracement after new B)
                        if new_B < D - 1:
                            new_move = prices[new_B] - prices[C]
                            new_retracement_threshold = prices[new_B] - (new_move * nested_threshold)

                            for j in range(new_B + 1, D):
                                if (prices[j] <= new_retracement_threshold and
                                        prices[j] < prices[j - 1] and
                                        j + 1 < len(prices) and prices[j] < prices[j + 1]):
                                    new_C = j
                                    break
            else:  # direction == "down"
                # Find new B (lowest price between C and D)
                new_B_idx = C + np.argmin(sub_prices)

                # Confirm new B is a trough
                if new_B_idx > C and new_B_idx < D:
                    if prices[new_B_idx] < prices[new_B_idx - 1] and prices[new_B_idx] < prices[new_B_idx + 1]:
                        new_B = new_B_idx

                        # Find new C (retracement after new B)
                        if new_B < D - 1:
                            new_move = prices[C] - prices[new_B]  # Note: C - new_B for downward
                            new_retracement_threshold = prices[new_B] + (new_move * nested_threshold)

                            for j in range(new_B + 1, D):
                                if (prices[j] >= new_retracement_threshold and
                                        prices[j] > prices[j - 1] and
                                        j + 1 < len(prices) and prices[j] > prices[j + 1]):
                                    new_C = j
                                    break

    # Calculate momentum extension
    momentum_extension = None
    extension_targets = None

    if D is not None:
        # Calculate momentum extensions beyond point D
        if direction == "up":
            # Calculate AB move
            AB_move = prices[B] - prices[A]

            # Calculate extension levels based on BC move
            BC_move = prices[B] - prices[C]

            # Set target extension levels
            target_levels = [1.0, 1.272, 1.618, 2.0, 2.618]
            extension_targets = {}

            for level in target_levels:
                target_price = prices[C] + (BC_move * level)
                extension_targets[level] = target_price

            # Determine if we already have a momentum extension
            if D < len(prices) - 1:
                post_D_max = max(prices[D + 1:])
                post_D_max_idx = D + 1 + np.argmax(prices[D + 1:])

                # Check if price continued higher after D
                if post_D_max > prices[D]:
                    momentum_extension = {
                        "idx": post_D_max_idx,
                        "price": post_D_max,
                        "percent_beyond_D": ((post_D_max - prices[D]) / prices[D]) * 100
                    }

                # Calculate which extension target it reached
                if momentum_extension:
                    for level, target in extension_targets.items():
                        if momentum_extension["price"] >= target:
                            momentum_extension["target_reached"] = level

        else:  # direction == "down"
            # Calculate AB move for downward direction
            AB_move = prices[A] - prices[B]

            # Calculate extension levels based on BC move
            BC_move = prices[C] - prices[B]

            # Set target extension levels
            target_levels = [1.0, 1.272, 1.618, 2.0, 2.618]
            extension_targets = {}

            for level in target_levels:
                target_price = prices[C] - (BC_move * level)
                extension_targets[level] = target_price

            # Determine if we already have a momentum extension
            if D < len(prices) - 1:
                post_D_min = min(prices[D + 1:])
                post_D_min_idx = D + 1 + np.argmin(prices[D + 1:])

                # Check if price continued lower after D
                if post_D_min < prices[D]:
                    momentum_extension = {
                        "idx": post_D_min_idx,
                        "price": post_D_min,
                        "percent_beyond_D": ((prices[D] - post_D_min) / prices[D]) * 100
                    }

                # Calculate which extension target it reached
                if momentum_extension:
                    for level, target in extension_targets.items():
                        if momentum_extension["price"] <= target:
                            momentum_extension["target_reached"] = level

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

    if new_B is not None:
        result["new_B"] = (new_B, prices[new_B])

    if new_C is not None:
        result["new_C"] = (new_C, prices[new_C])

    if extension_targets is not None:
        result["extension_targets"] = extension_targets

    if momentum_extension is not None:
        result["momentum_extension"] = momentum_extension

    return result


def plot_fib_retracement(prices, result):
    if not result:
        print("No valid retracement found.")
        return

    direction = result.get("direction", "up")

    plt.figure(figsize=(14, 8))

    # Create price chart
    plt.plot(prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series')

    # Add pattern labels
    move_type = "Upward" if direction == "up" else "Downward"
    plt.title(f'Fibonacci Retracement with Momentum Extension - {move_type} Move', fontsize=16)

    # Plot main ABCD points
    points = {'A': ('black', result['A']), 'B': ('red', result['B']), 'C': ('green', result['C'])}
    if 'D' in result:
        points['D'] = ('blue', result['D'])

    for label, (color, (idx, price)) in points.items():
        plt.scatter(idx, price, color=color, s=100, zorder=3)

        # Position labels based on direction
        if direction == "up":
            va_pos = 'bottom' if label in ['A', 'C'] else 'top'
        else:
            va_pos = 'top' if label in ['A', 'C'] else 'bottom'

        plt.text(idx, price, label, fontsize=14, fontweight='bold',
                 ha='right', va=va_pos, color=color)

    # Plot nested pattern points
    nested_points = {}
    if 'new_B' in result:
        nested_points['new B'] = ('darkred', result['new_B'])
    if 'new_C' in result:
        nested_points['new C'] = ('darkgreen', result['new_C'])

    for label, (color, (idx, price)) in nested_points.items():
        plt.scatter(idx, price, color=color, s=144, zorder=3, marker='s')
        plt.text(idx, price, label, fontsize=14, fontweight='bold',
                 ha='right', va='bottom', color=color)

    # Plot momentum extension point if it exists
    if 'momentum_extension' in result:
        momentum = result['momentum_extension']
        plt.scatter(momentum['idx'], momentum['price'], color='purple', s=144, zorder=3, marker='*')

        # Add label with target level reached
        target_text = f"Target: {momentum.get('target_reached', '?')}x"
        plt.text(momentum['idx'], momentum['price'],
                 f"Extension\n{target_text}",
                 fontsize=12, fontweight='bold', ha='right', va='top', color='purple')

    # Calculate and plot Fibonacci levels
    if direction == "up":
        main_move = result['B'][1] - result['A'][1]
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        for level in fib_levels:
            fib_price = result['B'][1] - (main_move * level)
            plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8,
                     verticalalignment='center', color='gray')
    else:
        main_move = result['A'][1] - result['B'][1]  # Note: A - B for downward
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        for level in fib_levels:
            fib_price = result['B'][1] + (main_move * level)
            plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8,
                     verticalalignment='center', color='gray')

    # Plot extension targets
    if 'extension_targets' in result:
        ext_levels = sorted(result['extension_targets'].keys())
        extension_colors = ['purple', 'indigo', 'darkviolet', 'mediumorchid', 'darkorchid']

        for level, color in zip(ext_levels, extension_colors):
            target_price = result['extension_targets'][level]
            plt.axhline(y=target_price, color=color, linestyle='-.', alpha=0.6, linewidth=1.5)
            plt.text(len(prices) - 1, target_price, f'{level}x', fontsize=8,
                     verticalalignment='center', horizontalalignment='right', color=color)

    # Draw lines connecting main points
    plt.plot([result['A'][0], result['B'][0], result['C'][0]],
             [result['A'][1], result['B'][1], result['C'][1]], 'k--', alpha=0.5)

    if 'D' in result:
        plt.plot([result['C'][0], result['D'][0]], [result['C'][1], result['D'][1]], 'k--', alpha=0.5)

        # Draw momentum extension line if it exists
        if 'momentum_extension' in result:
            momentum = result['momentum_extension']
            plt.plot([result['D'][0], momentum['idx']],
                     [result['D'][1], momentum['price']],
                     'purple', linestyle='-', linewidth=2, alpha=0.7)

    # Draw lines for nested pattern if it exists
    if 'new_B' in result and 'new_C' in result:
        plt.plot([result['C'][0], result['new_B'][0], result['new_C'][0]],
                 [result['C'][1], result['new_B'][1], result['new_C'][1]], 'r-.', alpha=0.5)

    # Add momentum extension explanation box
    if 'momentum_extension' in result:
        momentum = result['momentum_extension']
        box_text = (f"Momentum Extension:\n"
                    f"• Beyond point D: {momentum['percent_beyond_D']:.1f}%\n"
                    f"• Target reached: {momentum.get('target_reached', '?')}x\n"
                    f"• Signals strong trend continuation")

        plt.figtext(0.02, 0.02, box_text, fontsize=10,
                    bbox=dict(facecolor='lavender', alpha=0.5))

    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# Sample price data with momentum extension for upward move
up_prices = [10, 12, 15, 20, 25, 30, 28, 25, 20, 22, 28, 25, 30, 35, 40, 45, 55, 65]
#              A             B        C     nB  nC        D     Extension →

# Sample price data with momentum extension for downward move
down_prices = [65, 60, 55, 50, 45, 35, 38, 42, 45, 40, 35, 38, 30, 25, 20, 15, 10, 5]
#                A             B        C     nB  nC        D     Extension →

thresh = 0.3
nested_thresh = 0.2

# Find and plot upward move with momentum extension
up_result = find_retracement_extension(up_prices, threshold=thresh,
                                       nested_threshold=nested_thresh,
                                       direction="up")
plot_fib_retracement(up_prices, up_result)
print("\nUpward Move - Identified points:")
for key, value in up_result.items():
    if key not in ["threshold", "nested_threshold", "direction", "extension_targets"]:
        print(f"{key}: {value}")
if "momentum_extension" in up_result:
    print(f"Momentum Extension: {up_result['momentum_extension']}")
if "extension_targets" in up_result:
    print("Extension Targets:")
    for level, price in up_result["extension_targets"].items():
        print(f"  {level}x: {price}")

# Find and plot downward move with momentum extension
down_result = find_retracement_extension(down_prices, threshold=thresh,
                                         nested_threshold=nested_thresh,
                                         direction="down")
plot_fib_retracement(down_prices, down_result)
print("\nDownward Move - Identified points:")
for key, value in down_result.items():
    if key not in ["threshold", "nested_threshold", "direction", "extension_targets"]:
        print(f"{key}: {value}")
if "momentum_extension" in down_result:
    print(f"Momentum Extension: {down_result['momentum_extension']}")
if "extension_targets" in down_result:
    print("Extension Targets:")
    for level, price in down_result["extension_targets"].items():
        print(f"  {level}x: {price}")

plt.show()