from itertools import product as cartesian_product
from numba import jit
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
def add_ema_(df, cols, hls_sec):
    if type(cols) is str:
        cols = [cols]

    if 'dt' not in df:
        df['dt'] = df.index
        df['dt'] = df['dt'].diff().apply(lambda x: x.total_seconds())
        df['dt'] = df['dt'].fillna(0.0)

    for col, hl_in_sec in cartesian_product(cols, hls_sec):
        df[f'{col}_ema_{hl_in_sec}'] = calc_ts_ema(df[col].values, df['dt'].values, hl_in_sec)

def add_ems_(df, cols, hls_sec):
    if type(cols) is str:
        cols = [cols]

    if 'dt' not in df:
        df['dt'] = df.index
        df['dt'] = df['dt'].diff().apply(lambda x: x.total_seconds())
        df['dt'] = df['dt'].fillna(0.0)

    for col, halflife_in_s in cartesian_product(cols, hls_sec):
        df[f'{col}_ems_{halflife_in_s}'] = calc_ts_ems(df[col].values, df['dt'].values, halflife_in_s)

# to be implemented!!!!
# @jit
# def calc_ts_ems_normalized(x, dt, halflife_in_s):
#     eta = np.exp(-1.0 * dt / halflife_in_s)
#     eta = np.clip(eta, 0, np.inf) #
#     eta_cum = np.cumsum(eta)
#     n, y = x.shape[0], x.copy()
#
#     for i in np.arange(1, n):
#         if not np.isfinite(x[i]):
#             y[i] = y[i - 1]
#             continue
#
#         y[i] = (y[i - 1] * eta[i] + x[i])
#     return y


def pca(df, label, threshold=0.95):
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(df)
  pca = PCA(n_components=threshold)
  df_pca = pca.fit_transform(scaled_data)
  df_pca = pd.DataFrame(df_pca, columns=[f'{label}_pca_{i}' for i in range(df_pca.shape[1])])
  df_pca.index = df.index.copy()
  print(f'{df_pca.shape[1]}/ {df.shape[1]}')
  return df_pca