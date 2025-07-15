import pandas as pd
from typing import Tuple

from .filter import calc_ts_ema, calc_ts_ems
from numba import jit
import numpy as np

from common.miscs.basics import merge_by_index

def calc_ts_ema(x, dt, halflife_in_s):
    if type(x) != np.ndarray:
        raise Exception(f'x should be np.ndarray, but got {type(x)}')

    eta = np.exp(-1.0 * dt / halflife_in_s)
    eta = np.clip(eta, 0, np.inf)
    if np.isscalar(eta):
        eta = np.full(x.shape[0], eta)

    @jit(nopython=True)
    def __calc__(x, dt, eta):
        ome = 1.0 - eta
        n, y = x.shape[0], x.copy()

        for i in np.arange(1, n):
            if not np.isfinite(x[i]):
                y[i] = y[i - 1]
                continue

            y[i] = y[i - 1] * eta[i] + ome[i] * x[i]
        return y

    return __calc__(x, dt, eta)

def calc_ts_ems(x, dt, halflife_in_s):
    if type(x) != np.ndarray:
        raise Exception(f'x should be np.ndarray, but got {type(x)}')

    eta = np.exp(-1.0 * dt / halflife_in_s)
    eta = np.clip(eta, 0, np.inf)
    if np.isscalar(eta):
        eta = np.full(x.shape[0], eta)

    @jit(nopython=True)
    def __calc__(x, dt, eta):
        n, y = x.shape[0], x.copy()
        for i in np.arange(1, n):
            if not np.isfinite(x[i]):
                y[i] = y[i - 1]
                continue

            y[i] = y[i - 1] * eta[i] + x[i]
        return y

    return __calc__(x, dt, eta)

def fifo_match(trades: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Takes a dataframe of trades and produces a dataframe of
    FIFO matches for entries/exits. Trades where we entered
    long have sign 1, trades where we entered short have sign -1.
    :param trades: a dataframe with the following columns:
        * side: 0 for long, 1 for short
        * quantity: quantity of the trade
        * price: price of the trade
        * fee: fee of the trade. payable.
        * receipt_time: time of receipt of the trade
    :return: a dataframe of FIFO-matched trades
    """
    trades = trades.copy()
    trades["sign"] = -(trades.side * 2 - 1)
    open_qty = 0
    open_trades = []
    res = []
    for i, t in trades.iterrows():
        if open_qty == 0:
            # No open trades, so we open one
            open_trades.append((i, t.receipt_time, t.price, t.quantity, t.sign))
            open_qty = t.quantity * t.sign
        else:
            # We have an open trade
            if t.sign * open_qty > 0:
                # This increases our exposure
                open_qty += t.quantity * t.sign
                open_trades.append((i, t.receipt_time, t.price, t.quantity, t.sign))
            else:
                # This decreases our exposure
                cur_qty = t.quantity
                while cur_qty > 0 and len(open_trades) > 0:
                    if open_trades[0][3] > cur_qty:
                        # The current trade reduces the exposure, but does not
                        # eliminate/reverse it
                        res.append(
                            (
                                open_trades[0][0],
                                open_trades[0][1],
                                open_trades[0][2],
                                cur_qty,
                                open_trades[0][4],
                                i,
                                t.receipt_time,
                                t.price,
                            )
                        )
                        open_trades[0] = (
                            open_trades[0][0],
                            open_trades[0][1],
                            open_trades[0][2],
                            open_trades[0][3] - cur_qty,
                            open_trades[0][4],
                        )
                        open_qty += cur_qty * t.sign
                        cur_qty = 0
                    else:
                        # The current trade at least eliminates the exposure of
                        # the first open trade
                        cur_qty -= open_trades[0][3]
                        res.append(
                            (
                                open_trades[0][0],
                                open_trades[0][1],
                                open_trades[0][2],
                                open_trades[0][3],
                                open_trades[0][4],
                                i,
                                t.receipt_time,
                                t.price,
                            )
                        )
                        open_qty -= open_trades[0][3] * open_trades[0][4]
                        open_trades.pop(0)
                if cur_qty > 0:
                    # The current trade reverses the exposure
                    open_trades.append((i, t.receipt_time, t.price, cur_qty, t.sign))
                    open_qty += cur_qty * t.sign
    res = pd.DataFrame(
        res,
        columns=[
            "entry_trade_id",
            "entry_time",
            "entry_price",
            "quantity",
            "sign",
            "exit_trade_id",
            "exit_time",
            "exit_price",
        ],
    )
    res["net_pnl"] = (res.exit_price - res.entry_price) * res.sign * res.quantity
    # account for fees
    res["fee_pnl"] = 0
    #buy_trade_frac = res.quantity / trades.loc[res.entry_trade_id].quantity.values
    #sell_trade_frac = res.quantity / trades.loc[res.exit_trade_id].quantity.values
    #res.fee_pnl -= trades.loc[res.entry_trade_id].fee.values * buy_trade_frac
    #res.fee_pnl -= trades.loc[res.exit_trade_id].fee.values * sell_trade_frac
    res["pnl"] = res.net_pnl + res.fee_pnl
    return res, open_trades

def calc_trade_markout(df_trades, df_mkt_px, Ts_in_sec,
                       resampling_frequency='1s'):  # forward trade pnl marked at time of trade
    df_mkt_px = df_mkt_px.resample(rule=f'{resampling_frequency}').last().ffill()
    df_merged = pd.merge_asof(df_trades, df_mkt_px,
        left_index=True, right_index=True, direction='backward')
    # pnl_adj by formula
    df_merged['markout_adj'] = df_merged.eval("net_qty * (mid - price)")

    # Resample trades and sum up fee, signed quantity and pnl_adj
    # agg_map = {k: "sum" for k in ["fee", "qty", "net_qty", "markout_adj"]}
    #
    # agg_map |= {k: 'last' for k in ['mid', 'price', 'rv', 'bfm', 'sfm', 'f']}#, 'a', 'o']}
    #
    # df_merged = df_merged.resample('1s').agg(agg_map)
    # df_merged = df_merged.query('qty > 0').copy()

    # df_merged.ffill(inplace=True)  # otherwise fill NA with 0

    for n in Ts_in_sec:
        if n >= 1e9:
            n = int(1e9)
            markout_col = 'total_markout'
        else:
            markout_col = f'markout_{n}s'

        #fwd_col = f'fwd_px_{n}'
        #df_merged[fwd_col] = df_merged['mid'].shift(-offset)
        #df_merged['dpx'] = df_merged.eval(f'{fwd_col} - mid')
        df_fwd = df_mkt_px[['mid']].copy()
        df_fwd.index = df_fwd.index.map(lambda x : x - pd.Timedelta(f'{n}s'))
        fwd_col = f'fwd_px_{n}s'
        df_fwd.columns = [fwd_col]

        df_merged = pd.merge_asof(df_merged, df_fwd, left_index=True, right_index=True)
        df_merged[markout_col] = df_merged.eval(f'net_qty * ({fwd_col} - mid) + markout_adj - fee')
        df_merged[markout_col] = df_merged.eval(f'{markout_col} / (qty * mid) * 1e4')

    df_merged = df_merged.dropna(how='any', subset=[c for c in df_merged if 'markout' in c])
    return df_merged  # .drop('dpx',axis=1)

def calc_ems_net_qty(df_trades, hls = [30, 120, 300]):
    df_trades = df_trades.copy()
    for hl in hls :
        df_trades[f'net_qty_ems_{hl}'] = calc_ts_ems(df_trades['net_qty'].values, df_trades['dt'].values, hl)
    return df_trades[[c for c in df_trades if 'net_qty_ems' in c]]


def calc_trading_pnl(df_trades, df_mkt_px, Ts_in_sec=[300], freq='1s'):  # pnl marketd at time of px change
    df_trades = df_trades.copy()
    df_mkt_px = df_mkt_px.copy()
    df_mkt_px = df_mkt_px.resample(rule=f'{freq}').last().ffill()

    #df_trades = df_trades.query('time == time').copy()
    df_merged = merge_by_index(df_trades, df_mkt_px)

    # pnl_adj by formula
    df_merged['pnl_adj'] = df_merged['net_qty'] * (df_merged['mid'] - df_merged['price'])

    # Resample trades and sum up fee, signed quantity and pnl_adj
    agg_map = {k: 'sum' for k in ["fee", "net_qty", "pnl_adj", 'panel_pos']} # "qty"
    # agg_map |= {k : 'last' for k in ['a', 'o']}

    df_merged = df_merged.resample(freq).agg(agg_map)

    # only mid and change in mid should be precisely evaluated at the end of each second
    df_merged['mid'] = df_mkt_px[df_mkt_px.index.isin(df_merged.index)]['mid'].bfill()
    df_merged['dpx'] = df_merged['mid'].diff()
    df_merged['dpx'].values[0] = 0.0
    # df_merged.fillna(0, inplace=True)# otherwise fill NA with 0

    for T in Ts_in_sec:
        if T >= 1e9:
            T = int(1e9)
            pnl_col = 'total_pnl'
        else:
            pnl_col = f'trading_pnl_{T}s'

        # Calculate the first term in the trading PnL (the second term is pnl_adj)
        window = T / (pd.Timedelta(freq).total_seconds())
        window = int(np.round(window))
        df_merged['net_qty_rolling_sum'] = df_merged["net_qty"].rolling(window, closed='left', min_periods=0).sum()
        df_merged['rolling_sample_pnl'] = df_merged.eval("dpx * net_qty_rolling_sum")

        # Calculate rolling trading_pnl
        df_merged['rolling_trading_pnl'] = df_merged.eval('rolling_sample_pnl + pnl_adj - fee')
        df_merged[pnl_col] = df_merged['rolling_trading_pnl'].cumsum()
    return df_merged

def calc_daily_pnl(dict_df_by_symbol, date, fee_rate):
    df_order = dict_df_by_symbol["df_trd"].copy()
    df_order['cash_delta'] = df_order.eval(f'traded_qty * price* (-side - {fee_rate})')
    df_order['date'] = df_order.index.date
    cash_delta = df_order.groupby('date')['cash_delta'].sum()
    df_panel = dict_df_by_symbol["df_mkt"].copy()                    
    cash_delta_at_date = cash_delta.loc[date]
    df_panel['date'] = df_panel.index.date
    # end_pos = df_panel.loc[df_panel['date'] == date, 'panel_pos'].iloc[-1]
    # start_pos = df_panel.loc[df_panel['date'] == date, 'panel_pos'].iloc[0]
    end_pos = df_panel.query('date == @date')['panel_pos'].iloc[-1]
    start_pos = df_panel.query('date == @date')['panel_pos'].iloc[0]
    # print(f'end_pos: {end_pos}, start_pos: {start_pos}, cash_delta_at_date: {cash_delta_at_date}')
    return end_pos - start_pos + cash_delta_at_date

def sharpe(s, days_per_year = 365):
  n = len(s)
  if n <= 1:
    return np.nan

  std = max(0.01, s.std())
  return (s.mean() / std) * np.sqrt(days_per_year)

def win_ratio(s):
  return np.sum(s > 0) / len(s)

def win_ratio_lax(s):
  return np.sum(s >= 0) / len(s)

def n_count(s):
  return len(s)