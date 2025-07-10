import numpy as np
from .filter import calc_ts_ema, calc_ts_ems, add_ems_



def calc_ems_vol(df_mid, mid_fld, freq_ms, vol_hls=[300, 3600, 3600 * 12], max_vol=np.inf, mul=5.0):
    df_vol = df_mid[[mid_fld]].copy()
    freq_sec = freq_ms / 1e3
    n_samples_per_year = 365 * 24 * 3600 / freq_sec
    max_ret = max_vol / np.sqrt(n_samples_per_year) * mul
    df_vol['var_raw'] = (df_vol[mid_fld].shift(1) / df_vol[mid_fld]).fillna(1.0).apply(np.log).clip(-max_ret, max_ret)
    df_vol['var_raw'] = df_vol['var_raw'].apply(np.square)
    df_vol['dt'] = df_vol.index
    df_vol['dt'] = df_vol['dt'].diff().apply(lambda x: x.total_seconds())
    for hl in vol_hls:
        scaler = (1 - np.exp(- df_vol['dt'].mean() / hl))
        df_vol[f'vol_{hl}'] = calc_ts_ems(df_vol['var_raw'].values, df_vol['dt'].values, hl) * scaler
        df_vol[f'vol_{hl}'] = df_vol[f'vol_{hl}'].apply(lambda var: np.sqrt(var * n_samples_per_year))
    return df_vol  #[[c for c in df_vol if 'vol_' in c]].copy()



def calc_ems_log_var(df_mid, mid_fld, hls=[300, 3600, 3600 * 12]):
    df_vol = df_mid[[mid_fld]].copy()
    df_vol['ret'] = (df_vol[mid_fld].shift(1) / df_vol[mid_fld]).fillna(1.0)
    df_vol['log_var'] = df_vol.eval('log(ret * ret * 1e8 + 1.0)')
    add_ems_(df_vol, 'log_var', hls)
    return df_vol



def calc_ems_vol_by_ret(df_mid, ret_col, freq_ms, vol_hls=[300, 3600, 3600 * 12], max_vol=np.inf, mul=5.0):
    df_vol = df_mid[[ret_col]].copy()
    freq_sec = freq_ms / 1e3
    n_samples_per_year = 365 * 24 * 3600 / freq_sec
    max_ret = max_vol / np.sqrt(n_samples_per_year) * mul
    df_vol['var_raw'] = df_vol[ret_col].fillna(0.0).clip(-max_ret, max_ret).apply(lambda x: x * x)
    df_vol['dt'] = df_vol.index
    df_vol['dt'] = df_vol['dt'].diff().apply(lambda x: x.total_seconds())
    for hl in vol_hls:
        scaler = (1 - np.exp(- df_vol['dt'].mean() / hl))
        df_vol[f'vol_{hl}'] = calc_ts_ems(df_vol['var_raw'].values, df_vol['dt'].values, hl) * scaler
        df_vol[f'vol_{hl}'] = df_vol[f'vol_{hl}'].apply(lambda var: np.sqrt(var * n_samples_per_year))
    return df_vol  #[[c for c in df_vol if 'vol_' in c]].copy()
