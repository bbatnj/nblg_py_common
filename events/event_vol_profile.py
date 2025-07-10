import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import timedelta
from ..quants.stats import cohens_d

def calc_event_vol_profile(df_res, df_vol, time_window_start_hours=-2, time_window_end_hours=4, freq='15T', percentiles=[50, 90], save_dir=None):
    df_vol_non_weekends = df_vol[df_vol.index.dayofweek < 5]
    def try_convert_to_datetime(time_str):
        try:
            return pd.to_datetime(time_str)
        except ValueError:
            return None

    df_res['Time'] = df_res['Time'].apply(try_convert_to_datetime)
    df_res_cleaned = df_res.dropna(subset=['Time'])
    time_A = df_res_cleaned[['Event', 'Time']].drop_duplicates()
    time_A['Time'] = pd.to_datetime(time_A['Time'])
    time_A['Time'] = time_A['Time'].dt.tz_localize(None)
    df_vol_non_weekends['Date'] = df_vol_non_weekends.index.date
    df_vol_non_weekends['Time'] = df_vol_non_weekends.index.time
    time_noA = defaultdict(list)

    for event, event_time in time_A.iterrows():
        event_date = event_time['Time'].date()
        event_time_of_day = event_time['Time'].time()
        event_name = event_time['Event']
        matching_rows = df_vol_non_weekends[(df_vol_non_weekends['Time'] == event_time_of_day) &
                                            (df_vol_non_weekends['Date'] != event_date)].index.tolist()
        time_noA[event_name].extend(matching_rows)

    data = []
    for event, timestamps in time_noA.items():
        for timestamp in timestamps:
            data.append({'Event': event, 'Time': timestamp})
    df_time_noA = pd.DataFrame(data)

    df_vol_non_weekends.index = pd.to_datetime(df_vol_non_weekends.index)
    time_window_start = timedelta(hours=-2)
    time_window_end = timedelta(hours=4)

    df_A = []
    df_noA = []
    time_A.set_index('Time', inplace=True)
    df_time_noA.set_index('Time', inplace=True)

    def process_events(time_A, df_vol_non_weekends, time_window_start, time_window_end):
        selected_data_list = []
        for _, row in time_A.iterrows():
            event = row['Event']
            event_time = row.name
            start_time = event_time + time_window_start
            end_time = event_time + time_window_end
            print(f"Processing Event: {event}, Start: {start_time}, End: {end_time}")
            mask = (df_vol_non_weekends.index >= start_time) & (df_vol_non_weekends.index <= end_time)
            print(f"Matching records: {mask.sum()}")
            selected_data = df_vol_non_weekends.loc[mask].copy()
            if not selected_data.empty:
                selected_data['Event'] = event
                selected_data_list.append(selected_data)
        if selected_data_list:
            df_A = pd.concat(selected_data_list)
        else:
            df_A = pd.DataFrame()
        return df_A

    df_A = process_events(time_A, df_vol_non_weekends, time_window_start, time_window_end)
    df_noA = process_events(df_time_noA, df_vol_non_weekends, time_window_start, time_window_end)

    df_A['DateTime'] = pd.to_datetime(df_A['Date'].astype(str) + ' ' + df_A['Time'].astype(str))
    df_noA['DateTime'] = pd.to_datetime(df_noA['Date'].astype(str) + ' ' + df_noA['Time'].astype(str))

    df_A.set_index(['Event', 'DateTime'], inplace=True)
    df_noA.set_index(['Event', 'DateTime'], inplace=True)
    df_A.drop(columns=['Date', 'Time'], inplace=True)
    df_noA.drop(columns=['Date', 'Time'], inplace=True)

    def convert_datetime_to_timedelta(df):
        df['TimeDelta'] = df.index.get_level_values('DateTime') - df.index.get_level_values('DateTime').min()
        df.set_index('TimeDelta', append=True, inplace=True)
        df.reset_index(level='DateTime', inplace=True)
        return df

    df_A = convert_datetime_to_timedelta(df_A)
    df_noA = convert_datetime_to_timedelta(df_noA)

    def reshape_to_desired_format(df, time_window=(-120, 240), freq=15):
        start_offset, end_offset = time_window
        time_points = range(start_offset, end_offset + 1, freq)
        result = []
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        for event in df.index.get_level_values('Event').unique():
            event_data = df.xs(event, level='Event')
            for t in time_points:
                ref_datetime = event_data['DateTime'].iloc[0]
                time_point = ref_datetime + pd.Timedelta(minutes=t)
                time_range_start = time_point - pd.Timedelta(hours=2)
                time_range_end = time_point + pd.Timedelta(hours=4)
                window_data = event_data[(event_data['DateTime'] >= time_range_start) &
                                        (event_data['DateTime'] <= time_range_end)]
                P50_value = np.percentile(window_data['BTCUSDT.BNF@W_Vol_lmd_900'], 50) if not window_data.empty else np.nan
                P90_value = np.percentile(window_data['BTCUSDT.BNF@W_Vol_lmd_900'], 90) if not window_data.empty else np.nan
                result.append({
                    'Event': event,
                    'index': 'P50',
                    'BTCUSDT.BNF@W_Vol_lmd_900': P50_value,
                    'Time': t,
                    'DateTime': time_point
                })

                result.append({
                    'Event': event,
                    'index': 'P90',
                    'BTCUSDT.BNF@W_Vol_lmd_900': P90_value,
                    'Time': t,
                    'DateTime': time_point
                })
        return pd.DataFrame(result)

    df_A1 = reshape_to_desired_format(df_A)
    df_noA1 = reshape_to_desired_format(df_noA)

    all_events = df_A1["Event"].unique()
    all_times = df_A1["Time"].unique()
    results = []

    for event in all_events:
        df_event_A1 = df_A1.query("Event == @event")
        df_event_noA1 = df_noA1.query("Event == @event")
        event_result = {"Event": event}
        for time in all_times:
            with_event = df_event_A1.query("Time == @time")
            without_event = df_event_noA1.query("Time == @time")
            p50_with = with_event.query('index == "P50"')['BTCUSDT.BNF@W_Vol_lmd_900']
            p90_with = with_event.query('index == "P90"')['BTCUSDT.BNF@W_Vol_lmd_900']
            p50_without = without_event.query('index == "P50"')['BTCUSDT.BNF@W_Vol_lmd_900']
            p90_without = without_event.query('index == "P90"')['BTCUSDT.BNF@W_Vol_lmd_900']
            group1 = p50_with.tolist() + p90_with.tolist()
            group2 = p50_without.tolist() + p90_without.tolist()
            event_result[f"Cohen's_d_{time}min"] = cohens_d(group1, group2)
            event_result[f"With_Event_P50_{time}min"] = p50_with.mean() if not p50_with.empty else np.nan
            event_result[f"Without_Event_P50_{time}min"] = p50_without.mean() if not p50_without.empty else np.nan
            event_result[f"With_Event_P90_{time}min"] = p90_with.mean() if not p90_with.empty else np.nan
            event_result[f"Without_Event_P90_{time}min"] = p90_without.mean() if not p90_without.empty else np.nan
        results.append(event_result)

    results_df = pd.DataFrame(results)
    if save_dir:
        save_name = 'cohen_d_results.csv'
        results_df.to_csv(f'{save_dir}/{save_name}', index=False)
    return results_df

# Main entry point
if __name__ == '__main__':
    #####允许terminal输入参数
    vol_path = '/mnt/sda/NAS/ShareFolder/bb/for_others/wanting/df_vol_btc.parquet'
    res_path = '/mnt/sda/NAS/ShareFolder/bb/data/econ_events/sim/events_20231231_20241231.csv'
    df_res = pd.read_csv(res_path)
    df_vol = pd.read_parquet(vol_path)
    df_res['Time'] = pd.to_datetime(df_res['Time'], errors='coerce')
    # df_vol.to_csv('vol_data.csv')
    # df_res.to_csv('res_data.csv', index=False)

    result = calc_event_vol_profile(df_res, df_vol)
    result.to_csv('/home/bb/temp/cohen_d_results.csv', index=False)
    pass