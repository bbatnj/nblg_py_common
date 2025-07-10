#!/usr/bin/env python
# pip install binance-futures-connector
import os
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
from binance.client import Client
from binance.error import ClientError
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pickle

config_logging(logging, logging.DEBUG)

class BinanceClient:
    def __init__(self, account_name, api_key="", secret_key=""):
        self.account_name = account_name
        self.um_futures_client = UMFutures(key=api_key, secret=secret_key)
        self.client = Client(api_key, secret_key)
        
    def get_fills(self, instr, sdate, edate):
        try:
            response = self.um_futures_client.get_account_trades(
                symbol=instr, 
                startTime=np.int64(datetime.strptime(sdate, "%Y%m%d").timestamp()*1000), 
                endTime=np.int64(datetime.strptime(edate, "%Y%m%d").timestamp()*1000),
                recvWindow=2000)
            #logging.info(response)
            # print(response)
            if len(response) > 0:
                # Convert JSON data to DataFrame
                df = pd.DataFrame(response)
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
                df['date'] = df['time'].apply(lambda x: x.date())
                df['side'] = df['side'].apply(lambda x: 1 if x == 'BUY' else -1)
                df['notional'] = df['quoteQty']
                df['net_qty'] = df['side'] * df['qty']
                df['traded_qty'] = df['qty']
                df['fee'] = df['commission']
                df['realized_pnl'] = df['realizedPnl']
                df = df[['time','date','side','price','qty','traded_qty','notional','net_qty','fee','realized_pnl']]
                df.set_index(['time'], inplace=True)
                return df
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
        return None

    def get_pnls(self, sdate, edate, account_detail_file_root):

        dt_cur = datetime.strptime(sdate, '%Y%m%d')
        dt_end = datetime.strptime(edate, '%Y%m%d')

        dfs = []
        while dt_cur!=0 and dt_cur < dt_end:
            date = dt_cur.strftime('%Y%m%d')
            prev_date = (dt_cur - timedelta(days=1)).strftime('%Y%m%d')
            dt_cur = dt_cur + timedelta(days=1)

            prev_balances = self.get_balances_from_pickle(prev_date, account_detail_file_root)
            balances = self.get_balances_from_pickle(date, account_detail_file_root)
            if prev_balances is None or balances is None:
                continue
            prev_balance = prev_balances['marginBalance']
            balance = balances['marginBalance']
            pnl = balance - prev_balance
            rtn = 100 * pnl / prev_balance

            daily_pnl = {
                #'account': [self.account_name],
                'date': [date],
                'balance': [round(balance, 3)],
                'prev_balance': [round(prev_balance, 3)],
                'pnl': [round(pnl, 3)],
                'rtn': [round(rtn, 3)]
            }
            dfs.append(pd.DataFrame(daily_pnl))

        df = pd.concat(dfs, axis=0)

        df.set_index(['date'], inplace=True)
        return df
    

    def get_trades(self, date, symbols):
        from datetime import timezone, timedelta

        dt = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
        startTime = int(dt.timestamp() * 1000)  
        dt = dt + timedelta(days=1)  
        endTime = int(dt.timestamp() * 1000)  

        all_trades = []  

        for symbol in symbols:
            print(f"Fetching trades for symbol: {symbol}")
            symbol_trades = []  
            current_start = startTime

            while current_start < endTime:
                current_end = min(current_start + 3600000*24, endTime) 
                try:
                    response = self.um_futures_client.get_account_trades(
                        symbol=symbol, startTime=current_start, endTime=current_end, limit=1000
                    )
                    time.sleep(0.2)  
                    if len(response) > 0:
                        
                        df = pd.DataFrame(response)
                        df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
                        symbol_trades.append(df)

                    if len(response) < 1000:
                        current_start = current_end
                    else:
                        last_trade_time = response[-1]["time"]
                        current_start = last_trade_time + 1

                except ClientError as error:
                    logging.error(
                        "Error fetching trades for symbol {}. Status: {}, Code: {}, Message: {}".format(
                            symbol, error.status_code, error.error_code, error.error_message
                        )
                    )
                    break

            if symbol_trades:
                all_trades.append(pd.concat(symbol_trades, ignore_index=True))

        if all_trades:
            result_df = pd.concat(all_trades, ignore_index=True)
        else:
            result_df = pd.DataFrame()  

        return all_trades


    def _trades_to_csv(self, date, trades, trade_detail_file_root):
        '''
        [
            {
                "buyer": false, // 是否是买方
                "commission": "-0.07819010", // 手续费
                "commissionAsset": "USDT", // 手续费计价单位
                "id": 698759,   // 交易ID
                "maker": false, // 是否是挂单方
                "orderId": 25851813, // 订单编号
                "price": "7819.01", // 成交价
                "qty": "0.002", // 成交量
                "quoteQty": "15.63802", // 成交额
                "realizedPnl": "-0.91539999",   // 实现盈亏
                "side": "SELL", // 买卖方向
                "positionSide": "SHORT",  // 持仓方向
                "symbol": "BTCUSDT", // 交易对
                "time": 1569514978020 // 时间
            }
        ]
        '''
        if len(trades) == 0:
            return
        df = pd.concat(trades, axis=0)

        df.set_index(['datetime', 'symbol'], inplace=True)
        df = df.sort_index()
        # drop columns that has same orderID col, not the index, and only keep one
        df = df.drop_duplicates(subset=['id'])
        # print(df)

        if not os.path.isdir(f'{trade_detail_file_root}/{self.account_name}'):
            os.makedirs(f'{trade_detail_file_root}/{self.account_name}')
        df.to_csv(f'{trade_detail_file_root}/{self.account_name}/trade_detail_{date}.csv')

    def trades_from_csv(self, date, trade_detail_file_root):
        return pd.read_csv(f'{trade_detail_file_root}/{self.account_name}/trade_detail_{date}.csv')

    def download_trade_detail(self, date, symbols, trade_detail_file_root = None):
        trades = self.get_trades(date, symbols)
        if trade_detail_file_root is not None:
            self._trades_to_csv(date, trades, trade_detail_file_root)
        return trades

    def get_account_detail(self):

        try:
            response = self.um_futures_client.account(recvWindow=6000)
            print(type(response))
            #logging.info(response)
            # print(response)
            if len(response) > 0:
                return response
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
        return None

    def account_detail_to_pickle(self, date, account_detail, account_detail_file_root):
        if account_detail:
            if not os.path.isdir(f'{account_detail_file_root}/{self.account_name}'):
                os.makedirs(f'{account_detail_file_root}/{self.account_name}')
            with open(f'{account_detail_file_root}/{self.account_name}/account_detail_{date}.pkl', "wb") as f:
                pickle.dump(account_detail, f)

    def account_detail_from_pickle(self, date, account_detail_file_root):
        try:
            with open(f'{account_detail_file_root}/{self.account_name}/account_detail_{date}.pkl', "rb") as f:
                return pickle.load(f)
        except:
            return None
        
    def get_balances_from_pickle(self, date, account_detail_file_root):
        acct = self.account_detail_from_pickle(date, account_detail_file_root)
        if acct is None:
            #print(f"calc_pnl failed, today's account pickle not exist")
            return None
        assets = acct['assets']
        usdt = [a for a in assets if a['asset'] == 'USDT'][0]
        balances = {}
        balances['marginBalance'] = round(float(usdt['marginBalance']),3)
        ##### TODO: these information are not used in the current version
        # balances['unrealizedProfit'] = round(float(usdt['unrealizedProfit']),3)
        # balances['walletBalance'] = round(float(usdt['walletBalance']),3)
        # balances['availableBalance'] = round(float(usdt['availableBalance']),3)
        
        # balance = round(float(acc['totalMarginBalance']),2)  # totalMarginBalance
        return balances
    
    def get_positions_from_pickle(self, date, account_detail_file_root):
        acct = self.account_detail_from_pickle(date, account_detail_file_root)
        if acct is None:
            #print(f"calc_pnl failed, today's account pickle not exist")
            return None
        positions = {}
        for pos in acct['positions']:
            positions[pos['symbol']] = pos
        return positions

    def download_account_detail(self, date, account_detail_file_root = None):
        account_detail = self.get_account_detail()
        if account_detail and account_detail_file_root is not None:
            self.account_detail_to_pickle(date, account_detail, account_detail_file_root)
        return account_detail
    
    def get_account_snapshot(self, start_date, account_detail_file_root = None, day_range=3):
        dt = datetime.strptime(start_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        print("Start date (UTC):", dt)

        end_time = dt + timedelta(days=day_range)
        print("End date (UTC):", end_time)
        snapshots = self.client.get_account_snapshot(
        type='FUTURES',
        startTime=int(dt.timestamp() * 1000),   
        endTime=int(end_time.timestamp() * 1000)    
    )
        for snapshot in snapshots['snapshotVos']:
            snapshot_date = datetime.utcfromtimestamp(snapshot['updateTime'] / 1000)
            print(snapshot_date)
            # rewrite the date into like 20241130
            snapshot_date = snapshot_date.strftime("%Y%m%d")
            if  account_detail_file_root is not None:
                self.account_detail_to_pickle(snapshot_date, snapshot['data'], account_detail_file_root)
            print(f"Snapshot for {snapshot_date}:")
            print(snapshot['data'])
        return snapshots['snapshotVos'][0]

    def read_trade_detail(self, date, trade_detail_file_root):
        # if exists, read from csv
        if os.path.exists(f'{trade_detail_file_root}/{self.account_name}/trade_detail_{date}.csv'):
            return self.trades_from_csv(date, trade_detail_file_root)
        else:
            # if not exists, download from binance
            return self.download_trade_detail(date, ["BTCUSDT", "BTCUSDC", "ETHUSDT", "SOLUSDT"], trade_detail_file_root)   
    
    def calc_pnl_notional_from_trade_detail(self, date, trade_detail_file_root):
        # if exists, read from csv
        if os.path.exists(f'{trade_detail_file_root}/{self.account_name}/trade_detail_{date}.csv'):
            trades = self.read_trade_detail(date, trade_detail_file_root)
        else:
            # if not exists, jump out
            return None
        symbols = trades['symbol'].unique()
        volumes = {}
        for symbol in symbols:
            # volumns is the sum of the quoteQty row
            volumes[symbol] = trades[trades['symbol'] == symbol]['quoteQty'].sum()
        volumes['total'] = sum(volumes.values())
        print(f"{date} volumes: {volumes}")
    

if __name__ == '__main__':

    # test
    account_name='bin1'
    api_key='mke26hbUt0SGgqpNCBrh1MTh3os8lZEkH3pZqfZDbSa7npeD4eTYA8sgQbVLQGh8'
    secret_key='PfV3JLc5DQw4WyqtsC86ewEXFwrn7dA1jbCmB4I260KgaDSP6t7pD6csqANwPN4I'
    
    # default_save_dir = '/nas-1/ShareFolder/bb/prod_details_cache'
    file_root = '/nas-1/ShareFolder/bb/prod_details_cache'
    date = datetime.now().strftime("%Y%m%d")
    client = BinanceClient(account_name, api_key, secret_key)
    exchange_info = client.client.get_exchange_info()

    # symbols = [symbol["symbol"] for symbol in exchange_info["symbols"]]
    # print(symbols)
    # # print('get_fills',client.get_fills('BTCUSDT','20241130','20241201'))
    #RH: 1. get all trades for all coins
#        2. from those trades recalc Volume and PnL
#         Bin/Volume vs Engine/Volume
#         Bin/PnL vs Engine/PnL
    # date是六天前的日期
    client.get_account_snapshot( '20241128', file_root, 20)
         
    # print(client.get_pnls('20241127', '20241203', file_root))

    # print(client.download_account_detail(date, file_root))

    # balances = client.get_balances_from_pickle(date, file_root)
    # print(f"balances: {balances}")

    # positions = client.get_positions_from_pickle(date, file_root)
    # for s, p in positions.items():
    #     print(f"pos: {s}, {p}")

    symbols = ["BTCUSDT" , "BTCUSDC", "ETHUSDT", "SOLUSDT"]#"EOSUSDT",
    # date range from 20241127 to 20241203
    for date in  pd.date_range(start="20241128", end="20241230").strftime("%Y%m%d"):
        trades = client.download_trade_detail(date, symbols, file_root)
        
    for date in  pd.date_range(start="20241128", end="20241230").strftime("%Y%m%d"):
        trades = client.calc_pnl_notional_from_trade_detail(date, file_root)
    # # for t in trades:
    # #     print(f"trade: {t}")
