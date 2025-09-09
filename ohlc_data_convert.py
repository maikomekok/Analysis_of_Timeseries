from OHLC_data_convertion import process_bitcoin_data
import os

# Process Bitcoin data with 1-minute intervals (60 seconds)
results = process_bitcoin_data(
    input_dir='C:\\Users\\admin\\Desktop\\btc_data',
    output_dir='C:\\Users\\admin\\Desktop\\btc_1minute_data',  # Output directory for 1-minute data
    temp_dir='C:\\Users\\admin\\Desktop\\temp_extract',       # Separate temp directory
    interval_seconds=60,                                       # 60 seconds = 1-minute intervals
    cleanup=True
)

print(f"Processing completed. Generated {len(results)} files.")
for date, filepath in results.items():
    print(f"Date {date}: {filepath}")

# Uncomment below to process 1-second data instead:
# results_1sec = process_bitcoin_data(
#     input_dir='C:\\Users\\admin\\Desktop\\btc_data',
#     output_dir='C:\\Users\\admin\\Desktop\\btc_1second_data',  # Different output for 1-second data
#     temp_dir='C:\\Users\\admin\\Desktop\\temp_extract',
#     interval_seconds=1,                                        # 1-second intervals
#     cleanup=True
# )

# ========================== DIRECTORY CONFIGURATION ==========================
# Edit these paths to match your setup:

INPUT_DIRECTORY = 'C:\\Users\\admin\\Desktop\\btc_data'
OUTPUT_DIRECTORY_1MIN = 'C:\\Users\\admin\\Desktop\\btc_1minute_data'
OUTPUT_DIRECTORY_1SEC = 'C:\\Users\\admin\\Desktop\\btc_1second_data'
TEMP_DIRECTORY = 'C:\\Users\\admin\\Desktop\\temp_extract'

# Examples for different operating systems:
#
# Windows:
# INPUT_DIRECTORY = 'C:\\crypto_data\\btc'
# OUTPUT_DIRECTORY_1MIN = 'C:\\results\\btc_1minute'
#
# Linux/Mac:
# INPUT_DIRECTORY = '/home/user/btc_data'
# OUTPUT_DIRECTORY_1MIN = '/home/user/results/btc_1minute'
#
# Relative paths:
# INPUT_DIRECTORY = './raw_data'
# OUTPUT_DIRECTORY_1MIN = './processed_1minute'

# ==========================================================================