import pandas as pd

### Series Based
def add_event_series_index(df, dates_series, ahead_time, forward_time, dates_series_number_name='time_idx'):
    df=df.copy()
    df[dates_series_number_name] = 0
    for i, event_time in enumerate(dates_series):
        start_time = event_time - pd.to_timedelta(ahead_time)
        end_time = event_time + pd.to_timedelta(forward_time)
        df.loc[(df.index >= start_time) & (df.index < end_time), dates_series_number_name] = i + 1
    return df

### Interval Based
def add_interval_index(df, interval, interval_col_name='interval_within_event', group_name='time_idx'):
    # add_interval_index
    df=df.copy()
    df.sort_index(inplace=True)
    if group_name:
        def apply_intervals(group):
            date_ranges = pd.date_range(start=group.index.min(), end=group.index.max() + pd.to_timedelta(interval), freq=interval)
            group[interval_col_name] = pd.cut(group.index, bins=date_ranges, labels=False, right=False)
            return group
        df = df.groupby(group_name).apply(apply_intervals)
        return df.reset_index(level=0, drop=True)
    else:
        date_ranges = pd.date_range(start=df.index.min(), end=df.index.max() + pd.to_timedelta(interval), freq=interval)
        df[interval_col_name] = pd.cut(df.index, bins=date_ranges, labels=False, right=False)
        return df

### Seasonal Based
def add_partition_index(df, interval, unit='hour'):
    df=df.copy()
    df['weekday_index'] = df.index.weekday + 1
    if unit == 'hour':
        df['start_hour'] = (df.index.hour // interval) * interval
        df['end_hour'] = df['start_hour'] + interval
        df['interval_str'] = df['start_hour'].astype(str).str.zfill(2) + ":00-" + df['end_hour'].astype(str).str.zfill(2) + ":00"
    elif unit == 'minute':
        df['start_time'] = ((df.index.hour * 60 + df.index.minute )// interval) * interval
        df['end_time'] = df['start_time'] + interval
        df['start_hour'] = df['start_time'] // 60
        df['start_minute'] = df['start_time'] % 60
        df['end_hour'] = df['end_time'] // 60
        df['end_minute'] = df['end_time'] % 60
        df['interval_str'] = df['start_hour'].astype(str).str.zfill(2) + ":" + df['start_minute'].astype(str).str.zfill(2) + "-" + df['end_hour'].astype(str).str.zfill(2) + ":" + df['end_minute'].astype(str).str.zfill(2)
    else:
        raise ValueError("Unit must be either 'hour' or 'minute'")
    df['day_interval_within_week'] = 'weekday_' + df['weekday_index'].astype(str) + ', interval_' + df['interval_str']
    df.drop(columns=['start_time', 'end_time', 'start_hour', 'start_minute', 'end_hour', 'end_minute'], inplace=True, errors='ignore')
    return df


def get_sg_day_hrs(df):
    pass

def get_sg_weekend_hrs(df):
    pass


def add_time_label_singapore(df_vol):
    df_vol = df_vol.copy()
    df_vol.index = df_vol.index + pd.Timedelta('8h') #UTC -> Singapore
    df_vol = add_partition_index(df_vol, interval=60, unit='minute')
    df_vol['sg_weekday'] = df_vol['day_interval_within_week'].apply(lambda x : int(x.split(',')[0].split('_')[1]))
    #df_vol['time'] = df_vol['day_interval_within_week'].apply(lambda x : x.split(',')[1].split('_')[1])
    df_vol['sg_hr'] = df_vol['day_interval_within_week'].apply(lambda x : int(x.split(',')[1].split('_')[1].split(':')[0]))
    return df_vol


def add_time_label_ny(df_vol):
    df_vol.index = df_vol.index - pd.Timedelta('4h') #UTC -> NY
    df_vol = add_partition_index(df_vol, interval=60, unit='minute')
    df_vol['weekday'] = df_vol['day_interval_within_week'].apply(lambda x : int(x.split(',')[0].split('_')[1]))
    df_vol['time'] = df_vol['day_interval_within_week'].apply(lambda x : x.split(',')[1].split('_')[1])
    df_vol['hr'] = df_vol['day_interval_within_week'].apply(lambda x : int(x.split(',')[1].split('_')[1].split(':')[0]))
    return df_vol