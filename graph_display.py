import os
import matplotlib.pyplot as plt
from analyze import load_and_prepare_data, display_results


def quick_display(csv_file):
    """
    Simple function to just display the price chart without pattern analysis.
    Useful for debugging visualization issues.
    """
    print(f"Loading data from: {csv_file}")

    # Load the data
    prices, dates, df = load_and_prepare_data(csv_file)

    # Create a simple plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(dates, prices, marker='o', linestyle='-', color='blue', alpha=0.7, markersize=3)

    # Basic formatting
    ax.set_title(f'Bitcoin Price Chart', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Directly specify the CSV file path
    csv_file = 'C:/Users/admin/Desktop/btc_minute_data/btc_minute_data_2025-04-15.csv'
    quick_display(csv_file)