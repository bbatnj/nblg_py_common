from common.quants.pnl import calc_daily_pnl
from datetime import datetime, timedelta
import pandas as pd
import pickle
from common.binance_client import BinanceClient
from common.quants.pnl import calc_daily_pnl
from datetime import datetime, timedelta
import pandas as pd


default_save_dir = '/nas-1/ShareFolder/bb/prod_details_cache'

# def obtain_binance_detail(symbol, date_strftime, account_name='bin1', api_key='mke26hbUt0SGgqpNCBrh1MTh3os8lZEkH3pZqfZDbSa7npeD4eTYA8sgQbVLQGh8', secret_key='PfV3JLc5DQw4WyqtsC86ewEXFwrn7dA1jbCmB4I260KgaDSP6t7pD6csqANwPN4I', file_root = default_save_dir):
#     client = BinanceClient(account_name, api_key, secret_key)
#     exchange_info = client.client.get_exchange_info()
#     df =client.download_trade_detail(date_strftime, [symbol], file_root)
    
#     end_account_detail = client.get_account_snapshot( date_strftime, file_root, 0)
#     previous_date = datetime.strftime(datetime.strptime(date_strftime, '%Y%m%d') - timedelta(days=1), '%Y%m%d')
#     open_account_detail = client.get_account_snapshot( previous_date, file_root, 0)
#     return df[0], open_account_detail['data'], end_account_detail['data']

def calc_binance_prod_pnl (symbol, date, fee_rate, df_mkt=None, infor_dir=default_save_dir, account_name = 'bin1'):
    #if df_mkt is None, then we will use position downloaded from binance
    # if symbol has "." in it, remove it and all the characters after it
    if '.' in symbol:
        symbol = symbol.split('.')[0]
    cur_date = datetime.strftime(date, '%Y%m%d')
    previous_date = datetime.strftime(date - timedelta(days=1), '%Y%m%d')
    dict_df_by_symbol = {}

    details_dir = infor_dir+'/'+account_name
    df = pd.read_csv(f"{details_dir}/trade_detail_{cur_date}.csv")
    with open(f'{details_dir}/account_detail_{previous_date}.pkl', "rb") as f:
        open_account_detail = pickle.load(f)
    with open(f'{details_dir}/account_detail_{cur_date}.pkl', "rb") as f:
        end_account_detail  = pickle.load(f)      

    dict_df_by_symbol['df_trd'] = df[df['symbol'] == symbol].copy()
    dict_df_by_symbol['df_trd']['side'] = dict_df_by_symbol['df_trd']['side'].apply(lambda x: 1 if x == 'BUY' else -1)
    dict_df_by_symbol['df_trd'] = dict_df_by_symbol['df_trd'].rename(columns={'qty': 'traded_qty'})

    dict_df_by_symbol['df_trd'].index = dict_df_by_symbol['df_trd']['datetime'].apply(pd.Timestamp)
    
    if df_mkt is None:
        dict_df_by_symbol['df_mkt'] = pd.DataFrame()
        for i in open_account_detail['position']:
            if i['symbol'] == symbol:
                new_row = pd.DataFrame([{
                        'panel_pos': float(i['markPrice']) * float(i['positionAmt']),
                        'date': datetime.strftime(date, '%Y-%m-%d')
                    }])
                dict_df_by_symbol['df_mkt'] = pd.concat([dict_df_by_symbol['df_mkt'], new_row], ignore_index=True)
        if dict_df_by_symbol['df_mkt'].empty:
            dict_df_by_symbol['df_mkt'] = dict_df_by_symbol['df_mkt'].append({'panel_pos': float(0), 'date': datetime.strftime(date, '%Y-%m-%d')}, ignore_index=True)
        for i in end_account_detail ['position']:
            if i['symbol'] == symbol:
                new_row = pd.DataFrame([{
                        'panel_pos': float(i['markPrice']) * float(i['positionAmt']),
                        'date': datetime.strftime(date, '%Y-%m-%d')
                    }])
                dict_df_by_symbol['df_mkt'] = pd.concat([dict_df_by_symbol['df_mkt'], new_row], ignore_index=True)

        if dict_df_by_symbol['df_mkt'].shape[0] == 1:
            # dict_df_by_symbol['df_mkt'] = dict_df_by_symbol['df_mkt'].append({'panel_pos': float(0), 'date': datetime.strftime(date, '%Y-%m-%d')}, ignore_index=True)
            new_row = pd.DataFrame([{ 'panel_pos': float(0), 'date': datetime.strftime(date, '%Y-%m-%d')}])
            dict_df_by_symbol['df_mkt'] = pd.concat([dict_df_by_symbol['df_mkt'], new_row], ignore_index=True)

        dict_df_by_symbol['df_mkt'].index = pd.to_datetime(dict_df_by_symbol['df_mkt']['date'])
    else:
        dict_df_by_symbol['df_mkt'] = df_mkt
    date = datetime.strptime(cur_date, '%Y%m%d').date()
    return calc_daily_pnl(dict_df_by_symbol= dict_df_by_symbol,date= date, fee_rate=fee_rate)

def calc_binance_prod_notional  (symbol, date, infor_dir=default_save_dir, account_name = 'bin1'):
    if '.' in symbol:
        symbol = symbol.split('.')[0]
    cur_date = datetime.strftime(date, '%Y%m%d')
    details_dir = infor_dir+'/'+account_name
    trades = pd.read_csv(f"{details_dir}/trade_detail_{cur_date}.csv")
    return trades[trades['symbol'] == symbol]['quoteQty'].sum()
