import os

import pandas as pd
import pytz
from datetime import datetime, timedelta
from common.events.download_eco_events import get_and_parse_webpage_to_df
import numpy as np

def generate_time_range(start_date, end_date, freq, local_time, locol_timezone, output_timezone):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    local_tz = pytz.timezone(locol_timezone)
    output_tz = pytz.timezone(output_timezone) if output_timezone else pytz.utc
    local_hour, local_minute = map(int, local_time.split(':'))
    data = []
    for date in date_range:
        local_event_time = local_tz.localize(datetime(date.year, date.month, date.day, local_hour, local_minute))
        utc_event_time = local_event_time.astimezone(pytz.utc)
        final_event_time = utc_event_time.astimezone(output_tz)
        data.append(final_event_time)
    return data

def add_special_event(event_df, local_time, local_timezone, event_name,
                             output_timezone='UTC', currency="USD", 
                             impact=None, actual=None, forecast=None, previous=None, alert=None):#, start_date =None, end_date=None
    # start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # end_date = datetime.strptime(end_date, '%Y-%m-%d')
    event_df = event_df.copy()
    days =[]
    for i in range(len(event_df['Time'])):
        try:
            days.append(pd.to_datetime(event_df['Time'][i]).date())  
        except:
            pass
    date_range=set(days)
    start_date = str(min(date_range) - pd.Timedelta(days=1))
    end_date = str(max(date_range) + pd.Timedelta(days=1))
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="B") 
    local_tz = pytz.timezone(local_timezone)
    output_tz = pytz.timezone(output_timezone) if output_timezone else pytz.utc
    local_hour, local_minute = map(int, local_time.split(':'))

    data = []
    for date in date_range:
        local_event_time = local_tz.localize(datetime(date.year, date.month, date.day, local_hour, local_minute))

        utc_event_time = local_event_time.astimezone(pytz.utc)

        final_event_time = pd.Timestamp(utc_event_time.astimezone(output_tz))

        row = [
            final_event_time,#.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
            currency,
            impact,
            event_name,
            actual,
            forecast,
            previous,
            alert
        ]
        data.append(row)

    columns = ["Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous", "Alert"]

    df = pd.DataFrame(data, columns=columns)
    df['Time'] = df['Time'].apply(lambda t: pd.Timestamp(t).tz_convert(output_timezone))

    #df['SGTime'] = df['Time'].apply(lambda t: pd.Timestamp(t).tz_convert('Asia/Singapore'))
    df_res =  pd.concat([event_df, df], ignore_index=True)
    df_res['Time'] = df_res['Time'].apply(pd.to_datetime)
    #df_res['Time'] = df['Time'].apply(lambda t: t.tz_localize('UTC').tz_convert('US/Eastern'))

    df_res = df_res.sort_values('Time')
    return df_res

def obtain_important_action(data, importance_rule, action_rule, output_timezone='Asia/Singapore'):
    data = data.copy() 
    if importance_rule == {}:
        importance_rule = {'importance_list': 'USD', 'importance_column': 'Currency'}

    data = filter_data_by_importance(data, importance_rule['importance_list'], importance_rule['importance_column'])

    raw_df, action_df = gen_action_df(data, action_rule, output_timezone=output_timezone)
    return raw_df, action_df

def add_special_action(action_df, local_time, locol_timezone, action, event_name, weekday_choice, start_date=None, end_date=None, print_date_range=False):
    action_df = action_df.copy()
    output_timezone = action_df['Timezone'].iloc[0] if 'Timezone' in action_df else 'UTC'
    
    date_range = action_df['Time'].dt.date.unique()
    if start_date is None:
        start_date = str(min(date_range) - pd.Timedelta(days=1))
    if end_date is None:
        end_date = str(max(date_range) + pd.Timedelta(days=1))

    if weekday_choice == 'weekday':
        freq = 'B'
    elif weekday_choice == 'Saturday':
        freq = 'W-SAT'
    else:
        raise ValueError("Invalid weekday_choice: choose 'weekday' or 'Saturday'")

    final_event_times = generate_time_range(start_date, end_date, freq, local_time, locol_timezone, output_timezone)

    data = [[event_time, output_timezone, action, event_name] for event_time in final_event_times]
    columns = ["Time", "Timezone", "Action", "Event"]
    df = pd.DataFrame(data, columns=columns)

    action_df = pd.concat([action_df, df], ignore_index=True)
    if print_date_range:
        return action_df.sort_values(by='Time'), start_date, end_date
    else:
        return action_df.sort_values(by='Time')


def filter_data_by_importance(data, importance_list, importance_column):
    return data
    # if importance_column == 'Currency':
    #     return data.query('Currency in @importance_list')
    # elif importance_column == 'Event':
    #     data['Event'] = data['Event'].str.replace(r"\(.*\)","", regex = True).replace(' ','').replace('/','').replace(',','').replace('-','_')
    #     return data.query('Event in @importance_list') if importance_list else data
    
def gen_action_df(data, action_rule={}, output_timezone='Asia/Singapore'):

    raw_df = pd.DataFrame()
    data = data.copy()
    data['Time'] = data['Time'].apply(lambda t : pd.Timestamp(t).tz_convert(output_timezone))
    data = data.sort_values(by='Time').reset_index(drop=True)
    for index, row in data.iterrows():


        if row['Event'] in action_rule:
            before_hour = action_rule[row['Event']][0]
            after_hour = action_rule[row['Event']][1]
        else:
            before_hour = 30
            after_hour = 60


        new_before_time_row = pd.DataFrame([{'Time': row['Time'] - pd.Timedelta(minutes=before_hour),'Timezone': output_timezone  , 'Action': 'close_pos', 'Event': row['Event']}])
        new_after_time_row = pd.DataFrame([{'Time': row['Time'] + pd.Timedelta(minutes=after_hour),'Timezone': output_timezone, 'Action': 'trading', 'Event': row['Event']}])
        raw_df = pd.concat([raw_df, new_before_time_row, new_after_time_row])
    action_df = clean_action_df(raw_df)
    return raw_df, action_df

def clean_action_df(data):
    data = data.copy()

    # start from the first trading action
    final_rows = [data.iloc[0]]
    temp_trading_time = None
    event_set = []
    for _, row in data.iterrows():
        if row['Action'] == 'close_pos':
            if temp_trading_time is None or row['Time'] <= temp_trading_time:
                continue
            trading_row['Event'] = ', '.join(event_set)
            event_set = []
            final_rows.append(trading_row)
            final_rows.append(row)
            temp_trading_time = None  # Reset after a close_pos

        elif row['Action'] == 'trading':
            if temp_trading_time is None or row['Time'] >= temp_trading_time:
                event = row['Event']
                # clean the string of event
                event = event.replace(' ','_').replace('/','').replace(',','_').replace('-','_')
                event_set.append(event)
                temp_trading_time = row['Time']
                trading_row = row

    if temp_trading_time is not None:
        final_rows.append(trading_row)

    final_df = pd.DataFrame(final_rows)
    return final_df


TRIGGERS = [
    {"time": "08:30", "timezone": "America/New_York", "action": "manual_event_III", "event_name": "US_Param", "weekday_choice": "weekday"},
    {"time": "06:30", "timezone": "Asia/Singapore", "action": "manual_event_II", "event_name": "Asia_Param", "weekday_choice": "weekday"},
    {"time": "06:30", "timezone": "Asia/Singapore", "action": "manual_event_I", "event_name": "Weekend_Param", "weekday_choice": "Saturday"},
]

def add_triggers(action_df):
    action_df = action_df.copy()
    start_date, end_date = None, None
    for rule in TRIGGERS:
        action_df, start_date, end_date = add_special_action(
            action_df,
            rule["time"],
            rule["timezone"],
            rule["action"],
            rule["event_name"],
            rule["weekday_choice"],
            start_date=start_date,
            end_date=end_date,
            print_date_range=True
        )
    return action_df





def to_sim_events_format(df, output_csv):
    df = df.copy()

    if len(df['timezone'].unique()) != 1:
        raise ValueError("All timezones must be the same")

    timezone = df['timezone'].iloc[0]
    df['time'] = df['time'].apply(lambda x: x.tz_localize(timezone))
    df['time'] = pd.to_datetime(df['time']).dt.tz_convert('UTC')
    df['timezone'] = 'UTC'
    df['time_stamp_ns'] = df['time'].view('int64')  # Convert datetime to nanoseconds

    # Prepare the new DataFrame
    result_df = df[['time_stamp_ns', 'action']].rename(columns={'action': 'value'})
    result_df['param'] = 'engineUpdate'

    tradingStatusDict = {
        'trading': 'trading',
        'close_pos': 'closePos'
    }

    result_df['param'] = np.where(result_df['value'].isin(tradingStatusDict), 'tradingStatus', 'engineUpdate')

    sg_path = '/nas-1/ShareFolder/bb/data/econ_events/sim'
    wx_path = '/mnt/sda/home/bb/sim'

    e1 = f'{sg_path}/stra_manual_event_I.json'
    e2 = f'{sg_path}/stra_manual_event_II.json'
    e3 = f'{sg_path}/stra_manual_event_III.json'

    result_df['value'] = np.where(result_df['value'] == 'manual_event_I', e1, result_df['value'])
    result_df['value'] = np.where(result_df['value'] == 'manual_event_II', e2, result_df['value'])
    result_df['value'] = np.where(result_df['value'] == 'manual_event_III', e3, result_df['value'])

    result_df['value'] = result_df['value'].map(tradingStatusDict).fillna(result_df['value'])

    # Reorder columns as required
    result_df = result_df[['time_stamp_ns', 'param', 'value']]

    # Save to output CSV
    result_df.to_csv(output_csv, index=False)

    result_df['value'] = \
        result_df['value'].apply(lambda x : x.replace(sg_path, wx_path))

    result_df.to_csv(output_csv+'.wx', index=False)

    pass

    # def replace_double_quotes(input_file, output_file):
    #     """
    #     Reads a file, replaces all occurrences of "" with ", and writes to a new file.
    #
    #     :param input_file: Path to the input file
    #     :param output_file: Path to the output file
    #     """
    #     try:
    #         with open(input_file, 'r') as infile:
    #             content = infile.read()
    #
    #         # Replace "" with "
    #         updated_content = content.replace('""', '"')
    #
    #         with open(output_file, 'w') as outfile:
    #             outfile.write(updated_content)
    #
    #         print(f"File processed successfully. Output written to {output_file}")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #
    # replace_double_quotes(output_csv+'.tmp', output_csv)


# example usage
if __name__ == '__main__':
    #sdate = datetime.now() - timedelta(days=1)#- timedelta(days=365 - 16)
    #edate = datetime.now() + timedelta(days=15)
    sdate = pd.Timestamp('2025-07-08')
    edate = pd.Timestamp('2025-07-23')
    dst = 'prod'

    sdate_str = (sdate).strftime('%Y%m%d')
    edate_str = (edate).strftime('%Y%m%d')

    event_actions_fn = f'/nas-1/ShareFolder/bb/data/econ_events/{dst}/actions_{sdate_str}_{edate_str}.csv'
    live_actions = f'/home/bb/prod_deploy/main/prod_env/live_actions.csv'

    df = get_and_parse_webpage_to_df(sdate=sdate, edate=edate, dst=dst)

    event_df = add_special_event(df, local_time = '09:30', local_timezone = 'America/New_York', output_timezone='Asia/Singapore', event_name = 'US Open')

    raw_df, action_df = obtain_important_action(event_df, importance_rule={'importance_list':'USD', 'importance_column':'Currency'},
                                        action_rule={'AAA' : [75, 180]}, output_timezone='Asia/Singapore')
    added_action_df =add_triggers(action_df)

    added_action_df.columns = ['time', 'timezone', 'action', 'comment']
    added_action_df['comment'] \
        = added_action_df['comment'].str.replace('"', '').str.replace(',', '|').str.replace(' ', '')

    added_action_df['time'] = added_action_df['time'].apply(lambda x: x.replace(tzinfo=None))

    if dst == 'prod':
        #added_action_df.to_csv(event_actions_fn, index=False)
        added_action_df.to_csv(live_actions, index=False)
    elif dst == 'sim':
        to_sim_events_format(added_action_df, event_actions_fn)

    pass
