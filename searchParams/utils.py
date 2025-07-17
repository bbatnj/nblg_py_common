import os
import numpy as np
import pandas as pd

from common.quants.corr import calc_auto_corr
from common.quants.pnl import calc_trading_pnl, calc_daily_pnl

from collections import defaultdict as dd


def create_trd_mkt_pair(dict_df, symbol, fee_rate):
    df_panel = dict_df["panel"].query('instr == @symbol').copy()
    df_order = dict_df["order"].query('instr == @symbol').copy()
    # df_order f fillna, fill by previous value
    #df_order = df_order.fillna(method="ffill")
    df_mkt = df_panel[["bp", "ap", "pos"]].copy()
    df_mkt["mid"] = df_mkt.eval('0.5 * (bp + ap)')
    df_mkt = df_mkt.rename({"pos": "panel_pos"}, axis=1)
    # BB: why we need this ??? comment this out to see why there are NaT in df_mkt index
    df_mkt.index = pd.Series(df_mkt.index).ffill()

    df_trd = None
    try:
        cols = ["px", "side", "traded_qty", 'cancel_time', 'status', 'bu0', 'bu1', 'bu2', 'bu3']
        df_trd = df_order.query('abs(traded_qty) > 0').sort_index()[cols].copy()  #'fee',,'sent_time'
        df_trd["net_qty"] = df_trd.eval("side * traded_qty")

        if df_trd.shape == 0:
            print(f'"No trades found for "{symbol}"')
            raise Exception()

        df_trd = df_trd.rename({"px" : "price"}, axis=1)
        df_trd["fee"] = df_trd.eval('price * abs(traded_qty) * @fee_rate')
        df_trd = df_trd.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        print(f"Error in creating df_trd for {symbol}: {e}")

    return {
        "df_mkt": df_mkt,
        "df_trd": df_trd,
    }


def calc_auto_lags(df_mkt):
    # break df by day
    df_mkt = df_mkt.copy()
    df_mkt["date"] = df_mkt.index.date.astype(str)
    date = df_mkt["date"].unique()
    auto_corr = {}
    for day in date:
        df_mkt_day = df_mkt.loc[df_mkt["date"] == day]
        auto_corr_df = calc_auto_corr(df_mkt_day, "panel_pos", lags=[5, 60, 600, 3600], freq="1s")
        auto_corr[day] = auto_corr_df

    auto_corr = pd.concat(auto_corr)
    return auto_corr


def get_df_fills(df_trd):
    traded_df = df_trd.query("traded_qty > 0").dropna(subset='price side traded_qty cancel_time status net_qty fee'.split()).copy()
    traded_df["notional"] = traded_df.eval("traded_qty * price")
    traded_df["date"] = traded_df.index.date.astype(str)
    traded_df["qty"] = traded_df['traded_qty']
    # grouped = traded_df.groupby("date")
    # return grouped["notional"].describe()
    return traded_df

ORDER_STATUS = {
    'invalid':0,
    'pending':1,
    'new':2,
    'partial_fill':3,
    'fill':4,
    'cancel':5,
    'reject':6,
}
def calc_order_stat(df, sdate='1970-01-01', edate='2070-01-01'):
        ''' call report() before this function '''
        if df.empty:
            print(f'no order found during {sdate} to {edate}')
            return {}
        df_order = df.query('@sdate <= time <= @edate').copy()
        total_cnt = len(df_order)
        df_order = df_order.query('status > 3').copy() #  remove all those status <4
        total_qty = df_order['qty'].sum()
        rej_status = ORDER_STATUS["reject"]
        reject_cnt = len(df_order.query('status == @rej_status'))
        # once cancel command has been sent, there's a Core order log with action=cancel
        # in the implementation above, we would set cancel_time
        # we do not filter on status==cancel since some order would be filled since late cancel
        # in that case, order status=filled
        cancel_before_fill_cnt = len(df_order[(df_order['status'] != ORDER_STATUS["reject"]) & ((df_order['traded_qty'] == 0) | (df_order['traded_qty'].isna()))])
        late_cancel_cnt = len(df_order[(df_order['traded_qty'] > df_order['traded_qty_when_cancelling']) & ~df_order['cancel_time'].isna()])
        fill_without_cancel_cnt = len(df_order[(df_order['status'] != ORDER_STATUS["reject"]) & df_order['cancel_time'].isna()])
        trade_cnt = len(df_order[df_order['traded_qty'] > 0])
        sum_cnt = reject_cnt + cancel_before_fill_cnt + fill_without_cancel_cnt + late_cancel_cnt
        
        if sum_cnt != total_cnt:
            #print('calc_order_stat need check!!!')
            #print(f'rej({reject_cnt}) + cancel_before_fill({cancel_before_fill_cnt}) + fill_without_cancel({fill_without_cancel_cnt}) + late_cancel({late_cancel_cnt}) = {sum_cnt} != total({total_cnt})')
            pass

        # late_cancel_qty = (df_cancel['traded_qty'] - df_cancel['traded_qty_when_cancelling']).sum()
        if total_cnt == 0:
            result ={
                'reject_ratio': 0,
                'cancel_ratio': 0,
                'fill_ratio': 0,
                'late_ratio': 0,
                'late_cancel_cnt/trade_cnt': 0,
            }
        else:
            result = {
                'reject_ratio': reject_cnt / total_cnt,  ############reject_cnt/total_cnt, total_cnt might be 0
                'cancel_ratio': cancel_before_fill_cnt / total_cnt,  ###############cancel_before_fill_cnt/total_cnt
                'fill_ratio': fill_without_cancel_cnt / total_cnt, #################fill_without_cancel_cnt/total_cnt
                'late_ratio': late_cancel_cnt / total_cnt, ###################late_cancel_cnt/total_cnt
                'late_cancel_cnt/trade_cnt': late_cancel_cnt / trade_cnt if trade_cnt > 0 else np.nan
            }
        value_counts = df_order['r'].value_counts()
        proportions = (value_counts / len(df_order)).to_dict()
        proportions_with_prefix = {f"r: {k}": v for k, v in proportions.items()}
        result.update(proportions_with_prefix)

        if 'lifespan' in df_order.columns:
            result['mean_lifespan'] = df_order['lifespan'].mean().total_seconds()
            result['sum_lifespan'] = df_order['lifespan'].sum().total_seconds()
            result['lifespan_25_pct'] = df_order['lifespan'].quantile(0.25).total_seconds()
            result['lifespan_50_pct'] = df_order['lifespan'].quantile(0.50).total_seconds()
            result['lifespan_75_pct'] = df_order['lifespan'].quantile(0.75).total_seconds()
            result['lifespan_non_na_count'] = int(df_order['lifespan'].count())
        return result
    
def calc_trading_metrics(dict_df, fee_rate, Ts_in_sec=[20, 60, 180, 600, 1800, 1e9], account_name='bin1'):
    #METRIC_COLUMNS = ['date',"daily_pnl","prod_pnl", 'total_pnl', 'instr', 'notional','prod_notional', 'gross_book_size', 'net_book_size']
    symbols = set(dict_df["order"].groupby("instr").groups.keys()) | set(dict_df["panel"].groupby("instr").groups.keys())
    #df_metric = pd.DataFrame(columns=METRIC_COLUMNS)
    df_metric = []
    instr2dfs = {}
    for symbol in symbols:
        try:
            dict_df_by_symbol = create_trd_mkt_pair(dict_df, symbol, fee_rate)

            if dict_df_by_symbol["df_trd"].empty or dict_df_by_symbol["df_mkt"].empty:
                #print(f"{symbol}: has empty trd or mkt")
                continue

            # Calculate PnL and metrics
            pnl_df = calc_trading_pnl(
              dict_df_by_symbol["df_trd"].dropna(subset='price side traded_qty cancel_time status net_qty fee'.split()),
              dict_df_by_symbol["df_mkt"],
              Ts_in_sec=Ts_in_sec,
              freq="1s",
            )
            
            instr2dfs[symbol] = {}
            instr2dfs[symbol]["pnl"] = pnl_df
            instr2dfs[symbol]["auto_corr"] = calc_auto_lags(dict_df_by_symbol["df_mkt"])
            
            df_notional = get_df_fills(dict_df_by_symbol["df_trd"])
            instr2dfs[symbol]["fills"] = df_notional
            
            # Add a date column to split metrics by date
            pnl_df['date'] = pnl_df.index.date
            dict_df_by_symbol['df_mkt']['date'] = dict_df_by_symbol['df_mkt'].index.date
            df_notional['date'] = df_notional.index.date
            
            grouped_pnl = pnl_df.groupby('date')
            grouped_notional = df_notional.groupby('date')
            grouped_mkt = dict_df_by_symbol['df_mkt'].groupby('date')

            for date, pnl_group in grouped_pnl:
                metric = {'date': date, 'instr': symbol}

                try:
                    from common.binance_prod import calc_binance_prod_pnl, calc_binance_prod_notional
                    metric["prod_pnl"] = calc_binance_prod_pnl(symbol, date, fee_rate, account_name=account_name)#, df_mkt=dict_df_by_symbol["df_mkt"])
                except Exception as e:
                    #print(f"Error calculating prod_pnl for {symbol} on {date}: {e}")
                    metric["prod_pnl"] = np.nan

                try:
                    metric["daily_pnl"] = calc_daily_pnl(dict_df_by_symbol, date, fee_rate)
                except Exception as e:
                    #print(f"Error calculating daily_pnl for {symbol} on {date}: {e}")
                    metric["daily_pnl"] = np.nan

                try:
                    metric["total_pnl"] = pnl_group["total_pnl"].iloc[-1] - pnl_group["total_pnl"].iloc[0]
                except Exception as e:
                    #print(f"Error calculating total_pnl for {symbol} on {date}: {e}")
                    metric["total_pnl"]= np.nan

                try:
                    notional_group = grouped_notional.get_group(date)
                    metric['notional'] = notional_group['notional'].sum()
                except Exception as e:
                    #print(f"Error calculating notional for {symbol} on {date}: {e}")
                    metric['notional'] = np.nan

                try:
                    metric['prod_notional'] = calc_binance_prod_notional(symbol, date)
                except Exception as e:
                    #print(f"Error calculating prod_notional for {symbol} on {date}: {e}")
                    metric['prod_notional'] = np.nan

                try:
                    mkt_group = grouped_mkt.get_group(date)
                    gross_book_size = mkt_group['panel_pos'].resample('1s').last().abs()
                    metric['gross_book_mean'] = gross_book_size.mean()
                    metric['gross_book_90%'] = gross_book_size.quantile(0.9)
                    metric['gross_book_70%'] = gross_book_size.quantile(0.7)
                except Exception as e:
                    #print(f"Error calculating gross_book_size for {symbol} on {date}: {e}")
                    metric['gross_book'] = np.nan

                try:
                    net_book = mkt_group['panel_pos'].resample('1s').last()
                    metric['net_book_mean'] = net_book.mean()
                    metric['net_book_90%'] = net_book.quantile(0.9)
                    metric['net_book_70%'] = net_book.quantile(0.7)
                    metric['net_book_30%'] = net_book.quantile(0.3)
                    metric['net_book_20%'] = net_book.quantile(0.2)
                except Exception as e:
                    #print(f"Error calculating net_book_size for {symbol} on {date}: {e}")
                    metric['net_book'] = np.nan

                # Append to the result dataframe
                df_metric.append(metric)
        except Exception as e:
            pass
            #print(f"Error in calculating metrics for {symbol}: {e}")

    df_metric = pd.DataFrame(df_metric)#
    return df_metric, instr2dfs

#BB: use analyze fun insider logParser



