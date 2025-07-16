import pandas as pd
from datetime import datetime, timedelta
import pytz

def save_trading_data(start_date, end_date):
    # Define the Singapore timezone
    singapore_tz = pytz.timezone('Asia/Singapore')
    # Initialize an empty list to store records
    records = []

    # Convert start_date and end_date to datetime objects
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    while current_date <= end_date:
        # Handle Monday to Saturday for 6:30 AM
        if current_date.weekday() in range(0, 6):  # Monday to Saturday
            # Set the time to 6:30 AM and localize to Singapore timezone
            trading_time = current_date.replace(hour=6, minute=30, second=0, microsecond=0)
            trading_time = singapore_tz.localize(trading_time)
            # Convert to nanosecond-level UTC timestamp
            timestamp_ns_utc = int(trading_time.astimezone(pytz.UTC).timestamp() * 1e9)
            # Append the record to the list
            records.append([timestamp_ns_utc, 'tradingStatus', 'trading'])

        # Handle Monday to Friday for 7:30 PM
        if current_date.weekday() in range(0, 5):  # Monday to Friday
            # Set the time to 7:30 PM and localize to Singapore timezone
            close_time = current_date.replace(hour=19, minute=30, second=0, microsecond=0)
            close_time = singapore_tz.localize(close_time)
            # Convert to nanosecond-level UTC timestamp
            timestamp_ns_utc = int(close_time.astimezone(pytz.UTC).timestamp() * 1e9)
            # Append the record to the list
            records.append([timestamp_ns_utc, 'tradingStatus', 'closePos'])

        # Increment the current date by one day
        current_date += timedelta(days=1)

    # Create a DataFrame from the records
    df = pd.DataFrame(records, columns=['time_stamp_ns_in_UTC', 'tradingStatus', 'action'])
    return df