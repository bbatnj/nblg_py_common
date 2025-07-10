import numpy as np
import pandas as pd
import glob

from bidict import bidict
from common.log_parser import LogParser

from common.miscs.basics import format_df_nums

from .pnl import calc_trading_pnl, sharpe, win_ratio, calc_ems_net_qty, calc_trade_markout
from .corr import calc_auto_corr
from .vol import calc_ems_vol
from .filter import calc_ts_ems


def diff_trade_dirs(dir1, dir2, instr):
    from common.miscs.basics import merge_by_index
    log_parser1 = LogParser(dir1)
    log_parser2 = LogParser(dir2)

    stats_1, metric_df_1, dict_df_detail_1 = log_parser1.analyze()
    stats_2, metric_df_2, dict_df_detail_2 = log_parser2.analyze()

    df_order_stats = pd.DataFrame({'1': stats_1, '2': stats_2})

    dfs1 = log_parser1.get_dfs()
    dfs2 = log_parser2.get_dfs()
    df_panel_1 = dfs1['panel'].query('instr == @instr').copy()
    df_panel_2 = dfs2['panel'].query('instr == @instr').copy()
    df_merged = merge_by_index(df_panel_1, df_panel_2)
    return df_merged, df_order_stats

# class TradeAnalyzer:
#     def __init__(self, df_orders, df_panels):
#         self.df_orders = df_orders.copy()
#         self.df_trades = df_orders.query('qty > 0.0').copy()
#         self.df_panels = df_panels.copy()
#
#         self.id2instr = bidict({
#             0: 'BTC.USDC.BNF',
#             1: 'BTC.USDT.BNF'
#         })
#
#         self.df_ts, self.df_markout = None, None
#
#     def calc_ts(self, Ts_in_sec = [15, 60, 300, 1800, 7200, 14400, np.inf]):
#         df_panels = self.df_panels.copy()
#         df_trades = self.df_trades
#         id2instr = self.id2instr
#
#         df_instr_pnl = {}
#
#         for instr_id in id2instr:
#             df_mkt = df_panels.query('instr_id == @instr_id')[['mid', 'pos']].copy()
#             df_mkt = df_mkt.rename({'pos' : 'panel_pos'}, axis = 1)
#             df_trd = df_trades.query('instr_id == @instr_id')[['fee', 'price', 'net_qty', 'qty', 'a', 'o']].copy()
#             df_ts = calc_trading_pnl(df_trd, df_mkt, Ts_in_sec)
#             instr = id2instr[instr_id]
#             df_ts = df_ts[[c for c in df_ts if 'trading_pnl_' in c
#                            or c in {'total_pnl', 'fee', 'a', 'o', 'net_qty', 'mid', 'panel_pos'}]].copy()
#
#             for c in ['a', 'o', 'net_qty', 'mid', 'panel_pos']:
#                 df_ts[c] = df_ts[c].ffill().fillna(0.0)
#
#             df_ts['pos'] = df_ts['net_qty'].cumsum()
#             df_ts['notional'] = df_ts.eval('pos * mid')
#
#             df_ts[f'fee_pnl'] = - df_ts['fee'].cumsum()
#             df_ts = df_ts.drop('fee', axis=1)
#             df_ts.columns = [f'{instr}_' + c.replace('trading_pnl_', '').replace('total_pnl', 'total') for c in df_ts]
#             df_instr_pnl[instr] = df_ts
#
#         df_res = pd.concat([df_instr_pnl[k] for k in df_instr_pnl], axis=1).ffill().fillna(0.0)
#         hzs = [f'_{t}s' for t in Ts_in_sec if np.isfinite(t)] + ['_total']
#         for hz in hzs:
#             df_res[f'All{hz}'] = df_res[[c for c in df_res if c.endswith(hz)]].sum(axis=1)
#
#         df_res[f'All_fee_pnl'] = df_res[[c for c in df_res if 'fee_pnl' in c]].sum(axis=1)
#         df_res['All_notional'] = df_res[[c for c in df_res if 'notional' in c]].sum(axis=1)
#
#         self.df_ts = df_res
#
#     def calc_markout(self, Ts_in_sec = [15, 60, 300, 1800, 7200, 14400]):
#         df_panels = self.df_panels.copy()
#         df_trades = self.df_trades
#         id2instr = self.id2instr
#
#         df_instr_markout = []
#         for instr_id in id2instr:
#             df_mkt = df_panels.query('instr_id == @instr_id')[['mid']].copy()
#             df_trd = df_trades.query('instr_id == @instr_id')[['fee', 'price', 'net_qty', 'qty', 'a', 'o']].copy()
#             df_markout = calc_trade_markout(df_trd, df_mkt, Ts_in_sec)
#
#             instr = id2instr[instr_id]
#             df_markout = df_markout[[c for c in df_markout if 'markout_' in c or c in {'a', 'o'}]].copy()
#
#             df_markout['instr'] = instr
#             #df_markout.columns = [f'{instr}_' + c.replace('markout_', '')
#                                   #for c in df_markout]
#             df_instr_markout.append(df_markout)
#
#         df_res = pd.concat(df_instr_markout, axis = 0)
#
#         self.df_markout = df_res
#




# class SimPostTradeAnalysis:
#     def __init__(self, instr, dfs): ##TODO: instr is not used.
#         pattern = ""
#         prev_trunk_end_pnl, df_stats_list = 0.0, []
#         for df_temp in dfs:
#             df_temp['PnL'] += prev_trunk_end_pnl
#             df_temp['PnL'] = df_temp['PnL'].ffill()
#             df_temp['ts'] = df_temp['time']
#             df_temp['dt'] = df_temp['ts'].diff().fillna(0.0)
#             df_temp['time'] = df_temp['time'].apply(pd.Timestamp)
#             df_temp['date'] = df_temp['time'].apply(lambda t: t.date())
#             prev_trunk_end_pnl = df_temp['PnL'].iloc[-1]
#             df_stats_list.append(df_temp)
#
#         self.df_stats_list = df_stats_list
#         df_stats = pd.concat(self.df_stats_list).sort_index()
#
#         df_stats = df_stats.set_index('time').sort_index()
#         self.df_stats = df_stats
#         self.sdate, self.edate = self.df_stats['date'].values[0], self.df_stats['date'].values[-1]
#         self.pos_corr_lags = [60, 300, 900, 1800]
#
#         # df_trades
#         # RH: fix this, use sim_name ev_path to get all the files
#         df_trades = pd.concat([pd.read_parquet(fn) for fn in sorted(glob.glob(pattern))], axis = 0).sort_index() ##TODO: there is no "pattern" or "path" anymore
#         df_trades['time'] = df_trades['time'].apply(pd.Timestamp)
#         df_trades = df_trades.sort_values('time')
#         df_trades['dt'] = df_trades['time'].diff().apply(lambda t: t.total_seconds()).fillna(0.0)
#         df_trades['date'] = df_trades['time'].apply(lambda t: t.date())
#         df_trades = df_trades.set_index('time')
#
#         side2sgn = {b'SELL': -1, b'BUY': 1}
#         df_trades['side'] = df_trades['side'].apply(lambda x: side2sgn[x])
#         df_trades['net_qty'] = df_trades.eval('qty * side')
#         df_trades['cum_net_qty'] = df_trades['net_qty'].cumsum()
#         self.df_trades = df_trades
#
#         # RH: fix this, use sim_name ev_path to get all the files
#         df_order = []
#         for fn in sorted(glob.glob(pattern)): ##TODO: there is no "pattern" or "path" anymore
#             df_order.append(pd.read_parquet(fn))
#         df_order = pd.concat(df_order).sort_index()
#         df_order['time'] = df_order['time'].apply(pd.Timestamp)
#         df_order['createTime'] = df_order['createTime'].apply(pd.Timestamp)
#
#         order_cols = ['time', 'handle', 'limitPrice', 'avgExecutedPrice', 'remainingQty',
#                       'executedQty', 'pendingExecutedQty', 'side', 'type', 'timeInForce',
#                       'done', 'tradeOriginalQty',
#                       'doneReason', 'createTime', 'responseSuccess',
#                       'responseOrderRejectReason']
#         df_order = df_order[order_cols]
#         self.df_order = df_order.set_index('time').sort_index()
#
#         self.grid_freq_ms = 1000
#         #mid
#         self.df_mid = None #TODO fix me!!!
#
#     def calc_bucketed_trade_markout(self):
#         df_net_qty = calc_ems_net_qty(self.df_trades.copy())
#         df_net_qty.index = df_net_qty.index + pd.Timedelta('1s')
#         df_vol = calc_ems_vol(self.df_mid)
#         df_vol.index = df_vol.index + pd.Timedelta('1s')
#
#         df_trade_markout = calc_trade_markout(self.df_trades, self.df_mid, Ts=[180, 600, 1800])
#
#         df_merged = pd.merge_asof(df_trade_markout, df_vol, left_index=True, right_index=True)
#         df_merged = pd.merge_asof(df_merged, df_net_qty, left_index=True, right_index=True)
#
#         for c in df_merged:
#             if 'markout_' in c:
#                 df_merged[c] = df_merged.eval(f'{c} / qty')
#
#         df_merged['net_qty_dis'] = pd.qcut(df_merged['net_qty_ems_30'].abs(), q=5)
#         df_merged['vol_dis'] = pd.qcut(df_merged['vol_300'], q=5)
#
#         col_markouts = [c for c in df_merged if 'markout_' in c and '_adj' not in c]
#
#         df_vol = df_merged.groupby(['vol_dis'])[col_markouts].agg(['mean', 'median', 'std'])
#         df_net_qty = df_merged.groupby(['net_qty_dis'])[col_markouts].agg(['mean', 'median', 'std'])
#         return df_vol, df_net_qty
#
#     def calc_order_stats(self):
#         # null_val = b'NULL_VAL'
#         # powe = b'POST_ONLY_WOULD_EXECUTE'
#         # ioc = b'IOC'
#         # gtx = b'GTX'ta = TradeAnalyzer(df_orders, df_panels)
#         # cancelled = b'CANCELLED'
#         # filled = b'FILLED'
#         # rejected = b'REJECTED'
#
#         f_exec_stats = lambda df: df.query('doneReason != b"NULL_VAL"').groupby('handle').last().groupby(['timeInForce', 'doneReason'])[
#             ['remainingQty', 'executedQty']].agg(['sum', 'count'])
#         df_exec_stats = self.df_order.groupby(lambda time: time.date()).apply(f_exec_stats)
#         df_exec_stats.columns = ['_'.join(c) for c in df_exec_stats.columns]
#         df_exec_stats = df_exec_stats.groupby(['timeInForce', 'doneReason']).agg(['mean', 'median', 'std'])
#         return df_exec_stats
#
#     # def calc_pnl_stats(self, exclusive_dates = None):
#     #     df_trades = self.df_trades
#     #     pnl_cols = [c for c in df_trades if '_pnl' in c and 'rolling_' not in c]
#     #     df_daily_pnl = df_trades.groupby('date').apply(lambda x: x[pnl_cols].iloc[-1] - x[pnl_cols].iloc[0])
#
#     def calc_trading_pnls(self, holdings_sec: list):
#         df_res = []
#         holdings_sec.append(np.inf) #total pnl
#         for t in holdings_sec:
#             df_pnl = calc_trading_pnl(self.df_trades, self.df_mid, T=t)
#             df_res.append(df_pnl)
#         df_res = pd.concat(df_res, axis=1)
#         df_res = df_res.loc[:, ~df_res.columns.duplicated()]
#         df_res['cum_net_qty'] = df_res['net_qty'].cumsum()
#         df_res['cum_gross_qty']= df_res['net_qty'].abs().cumsum()
#
#         cols = [c for c in df_res if 'trading_pnl_' in c]
#         cols += 'mid cum_net_qty qty net_qty total_pnl fee'.split()
#         df_res = df_res[cols].fillna(0.0)
#         df_res = pd.merge_asof(df_res,
#                                self.df_stats[['PnL']].rename({'PnL' : 'PnL_from_engine'}, axis=1),
#                                left_index= True, right_index=True)
#         return df_res
#
#     def calc_daily_trades(self):
#         df_daily = self.df_trades.groupby('date')[['notional']].agg(['sum', 'count'])
#         df_daily.columns = ['notional', 'trd_cnt']
#         return df_daily
#
#     def calc_pos_corr(self):
#         df_corr = self.df_stats.groupby('date')[['position']].apply(
#             lambda df: calc_auto_corr(df, 'position', self.pos_corr_lags))
#         df_corr.index.rename('lag', level=1, inplace=True)
#         max_func = lambda x: x.abs().max()
#         max_func.__name__ = 'max_abs'
#         df_corr = df_corr.groupby('lag').agg(['mean', 'median', 'std', max_func])
#         df_corr = df_corr.rename({'position': 'pos_corr'}, axis=1)
#         df_corr.columns = ['_'.join(c) for c in df_corr]
#         return df_corr
#
# class GroupSimPTA:
#     def __init__(self, instr, sim_name: str, tactic_pattern='*'):
#         ev_path = '' #RH : TODO FIX ME
#         self.sim_name = sim_name
#         n_samples = len(glob.glob(f'{ev_path}/{sim_name}/*/sample_*'))
#         df_res, daily_pnls = {}, []
#
#         self.ptas = []
#         print(f'n_sims = {n_samples}')
#
#         for i in range(1, n_samples + 1):
#             sample = f'sample_{i}'
#             pta = SimPostTradeAnalysis(instr, sim_name, ev_path, sample=sample, tactic_pattern=tactic_pattern) ##TODO: instr is not used; SimPostTradeAnalysis only has two parameters now (instr, dfs)
#             self.ptas.append(pta)
#
#     def summarize(self, exclusive_dates={}):
#         df_res, daily_pnls = [], []
#         for pta in self.ptas:
#             df_pnl = pta.calc_pnl_stats(exclusive_dates=exclusive_dates)
#             df_volume = pta.calc_daily_trades()
#             df_daily = pd.concat([df_pnl, df_volume], axis=1)
#             pnl = df_daily['PnL']
#
#             order_stats = pta.calc_order_stats()
#             exe_qty = order_stats['executedQty_sum']['mean'].sum(
#                 axis=0)
#             gtx_exe_qty = order_stats.query('timeInForce == b"GTX"')['executedQty_sum']['mean'].sum(axis=0)
#             unfilledqty = order_stats['remainingQty_sum']['mean'].sum(axis=0)
#             qty_sent = unfilledqty + exe_qty
#
#             df_pos_corr = pta.calc_pos_corr()
#             res = {
#                 'sharpe': sharpe(pnl),
#                 'pnl_mean': pnl.mean(),
#                 'pnl_median': pnl.median(),
#                 'pnl_std': pnl.std(),
#                 'pnl_min': pnl.min(),
#                 'win_ratio': win_ratio(pnl),
#                 # activities
#                 'avg_notional': df_daily['notional'].mean(),
#                 'avg_trade_cnt': df_daily['trd_cnt'].mean(),
#                 'exe_qty': exe_qty,
#                 'qty_sent': qty_sent,
#                 'qty_fill_ratio': exe_qty / qty_sent,
#                 'post_qty_exe_ratio': gtx_exe_qty / exe_qty,
#                 # 'orders_sent' : np.nan,  #todo
#                 # 'gtx_order_cancel_ratio' : np.nan, #todo
#                 # 'ioc_fill_ratio' : np.nan
#             }
#
#             # pos corr
#             for lag in pta.pos_corr_lags:
#                 res[f'pos_corr_{lag}'] = df_pos_corr.loc[lag, 'pos_corr_median']
#             df_res.append(res)
#
#         df_res = pd.DataFrame(df_res)
#         comma_sep_cols = [c for c in df_res if 'pnl' in c or 'notional' in c]
#         return format_df_nums(df_res, comma_sep_cols)  # , daily_pnls
