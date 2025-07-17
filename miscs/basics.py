import inspect
import numpy as np
import pandas as pd
import multiprocessing
from timeit import default_timer as timer
import os
from itertools import product as cartesian_product


def sort_files_by_creation_time(directory, pattern="*"):
  """
  Sorts files in the given directory by creation time (newest to oldest).

  :param directory: The path to the folder containing files.
  :param pattern: The file pattern to match (default is '*', meaning all files).
  :return: List of sorted file paths.
  """

  from pathlib import Path
  import glob

  files = [Path(f) for f in glob.glob(f"{directory}/{pattern}") if Path(f).is_file()]
  files_sorted = sorted(files, key=lambda f: f.stat().st_ctime, reverse=False)
  return [str(f) for f in files_sorted]


def convert_to_float(value):
  if pd.isna(value) or value == "NaN":  # Handle NaNs
    return np.nan
  value = str(value).replace(",", "")  # Remove commas
  if value.endswith("%"):  # Convert percentages to decimals
    return float(value.strip("%")) / 100
  if value.endswith("M"):  # Convert Millions
    return float(value.strip("M")) * 1e6
  if value.endswith("K"):  # Convert Thousands
    return float(value.strip("K")) * 1e3
  return float(value)  # Convert normal numbers

def remove_small_files(directory, suffix, size_limit_kb=100):
  """
  Recursively removes .txt.gz files smaller than the given size limit.

  Args:
      directory (str): The base directory to start the search.
      size_limit_kb (int): The size limit in kilobytes. Files smaller than this will be deleted.
  """
  size_limit_bytes = size_limit_kb * 1024  # Convert KB to bytes

  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(suffix):
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)

        if file_size < size_limit_bytes:
          print(f"Removing: {file_path} (Size: {file_size} bytes)")
          os.remove(file_path)


#write a func if outfn exist, rename it to outfn.bak.i which i is the smallest number that outfn.bak.i not exist
def make_old(outfn):
  if not os.path.exists(outfn):
    return

  i = 0
  dst_fn = f'{outfn}.bak.{i}'
  while os.path.exists(dst_fn):
    i += 1
    dst_fn = f'{outfn}.bak.{i}'

  os.system(f'mv {outfn} {dst_fn}')


def take_all_cols(x_cols):
  #x_cols is None or 'All' or []
  return not x_cols or (type(x_cols) is str and x_cols.upper() == 'ALL') or len(x_cols) == 0

def df2float32_(df):
  for c in df:
    if df.dtypes[c] == 'float64' or df.dtypes[c] == np.float64:
      df[c] = df[c].astype(np.float32)

def format2kmgt(num):
  magnitude = 0
  while abs(num) >= 1000:
    magnitude += 1
    num /= 1000.0
  return f'{num:.1f}{" " + " kMGTPEZY"[magnitude]}'


def print_df_mem_usage(df: pd.DataFrame):
  x = df.memory_usage().sum()
  print(f'{format2kmgt(x)}B')


def print_df_stats(df):
  df_stats = df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
  df_stats['normalized_mean'] = df_stats.eval('mean / std')
  df_stats['skew'] = df.skew()
  df_stats['kurtosis'] = df.kurtosis()
  return df_stats


def group_and_concat(df, gcol, cols=None):
  df_merged = None

  for glabel, df in df.groupby(gcol):
    if cols:
      df = df[cols].copy()
    df.columns = [f'{glabel}@{c}' for c in df]

    if df_merged is None:
      df_merged = df
    else:
      df_merged = merge_by_index(df_merged, df)
  return df_merged


def m_qcut_(df, cols, nbins):
  if isinstance(nbins, int) or isinstance(nbins, float):
    nbins = [int(nbins)] * len(cols)

  for nb, c in zip(nbins, cols):
    df[f'{c}_bin_label'] = pd.qcut(df[c], nb)
    df[f'{c}_bin_label'] = df[f'{c}_bin_label'].astype(str)

  df['bin_label'] = ''
  for c in cols:
    df['bin_label'] += df[f'{c}_bin_label']


def merge_by_index(df_left, df_right):
  left_index, right_index = df_left.index, df_right.index

  #####!
  if df_left.index.dtype != df_right.index.dtype:
    df_left.index = df_left.index.astype('datetime64[ns]')
    df_right.index = df_right.index.astype('datetime64[ns]')

  df_res = pd.merge_asof(df_left, df_right, left_index=True, right_index=True)

  df_left.index, df_right.index = left_index, right_index
  return df_res




def merge_asof(df_left, df_right):
  # return pd.concat([df_left, df_right], axis=1)#.ffill().dropna(how='any')
  return merge_by_index(df_left, df_right)


def toDf32_(df):  # _ indicate change df inplace
  for c, kind in df.dtypes.items():
    try:
      if np.issubdtype(kind, np.float64):
        df[c] = df[c].astype(np.float32)
      elif np.issubdtype(kind, np.int64):
        df[c] = df[c].astype(np.int32)
    except Exception as e:
      print(f'Error converting {c} to float32: {e}')


def format_df(df, cols_with_comma=[]):
  return format_df_nums(df, cols_with_comma)


def format_df_nums(df, cols_with_comma=[]):
  def custom_format(number):
    # Use scientific notation for very small or very large numbers
    if abs(number) < 0.001 :
      return f'{number:.3e}'
    else:
      return f'{number:.3g}'

  df = df.copy()
  for c, t in df.dtypes.items():

    if np.issubdtype(t, np.number):

      if c in cols_with_comma:
        df[c] = df[c].apply(lambda x: f'{x:,.2f}')
      else:
          df[c] = df[c].apply(custom_format)
  return df


class Timer:
  def __init__(self, des):
    self.des = des

  def __enter__(self):
    self.start = timer()

  def __exit__(self, exc_type, exc_value, traceback):
    end = timer()
    self.dur = pd.Timedelta(end - self.start, unit="second")
    print(f'{self.des} finished in : {self.dur.total_seconds()} secs')


def cpu_at_80():
  return int(multiprocessing.cpu_count() * 0.8)


def cpum2():
  return max(multiprocessing.cpu_count() - 2, 1)


def to_cents(x):
  return np.round(x * 100)


def round2int(x):
  return int(np.round(x))


# alias
px2cents = to_cents


def substr(s, i):
  re = s.split()
  if i < len(re):
    return re[i]
  else:
    return None


def str2ts(s):
  try:
    return pd.Timestamp(s)
  except:
    return None


def ymd(d):
  return str(pd.Timestamp(d).date()).replace('-', '')


def ymd_dir(d, sep='/'):
  d = str(pd.Timestamp(d).date()).replace('-', '')
  return sep + sep.join([d[0:4], d[4:6], d[6:8]])


def index2dt(df):
  return df.index.map(lambda x: str(pd.Timestamp(x).date()))


def get_tmr():
  from datetime import date, timedelta

  tomorrow = date.today() + timedelta(days=1)
  formatted_tomorrow = tomorrow.strftime("%Y-%m-%d")
  return formatted_tomorrow


class SaveInputArgs:
  def __init__(self, ignore=[]):  # this just set the input args as class attr.
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k: v for k, v in local_vars.items()
                    if k not in set(ignore + ['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
      setattr(self, k, v)

