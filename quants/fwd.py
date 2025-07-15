import numpy as np
import pandas as pd

def calc_expect_ret_bps(t_sec, vol = 0.6):
    return 1e4  * np.sqrt(t_sec / 3600 / 24 / 365 ) * vol

def add_fwd_ret_by_n_(df: pd.DataFrame, kind: str, col, n_s: list, s2col = None):
  if kind not in {"abs", "log"}:
    raise Exception(f"unknown kind for using add fwd_ret_by_n")

  for n in n_s:
    c, fwd_c = f'{col}_ret_{n}_n', f'{col}_fwd_{n}_n'
    df[fwd_c] = df[col].shift(-n)

    if kind == "abs":
      df[c] = df.eval(f'{fwd_c} - {col}')
    elif kind == "log":
      df[c] = df.eval(f'{fwd_c}/{col}').apply(np.log)

    if s2col:
      df[f'wgt_{c}'] = df[s2col] + df[s2col].shift(-n)
      df[f'wgt_{c}'] = 1.0 / df[f'wgt_{c}']

