import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from .filter import calc_ts_ema, calc_ts_ems
from tqdm import tqdm

from .pnl import win_ratio


def calc_corr_by_prefix(df, prefixes):
    results = []
    for prefix in prefixes:
        x_col = f"{prefix}_x"
        y_col = f"{prefix}_y"

        if x_col in df.columns and y_col in df.columns:
            pearson_corr = df[x_col].corr(df[y_col], method='pearson')
            spearman_corr = df[x_col].corr(df[y_col], method='spearman')
            results.append([prefix, pearson_corr, spearman_corr])
        else:
            print(f"Columns for prefix '{prefix}' not found in the DataFrame.")

    return pd.DataFrame(results, columns=['prefix', 'corr', 'rank'])


def pair_corr(df, x_cols, y_cols, method='pearson'):
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression as LR
    # Dictionary to store correlation results
    correlation_dict = {'x': [], 'y': [], 'corr': [], 'beta': [], 'win_ratio': []}

    # Calculate correlation for each pair of x_col and y_col
    for x in x_cols:
        for y in y_cols:
            if x == y:
                continue
            corr_value = df[[x, y]].corr(method=method).iloc[0,1]
            df_temp = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            lr = LR(fit_intercept=False)
            lr.fit(df_temp[[x]].values, df_temp[y].values)
            correlation_dict['beta'].append(lr.coef_[0])
            correlation_dict['x'].append(x)
            correlation_dict['y'].append(y)
            correlation_dict['corr'].append(corr_value)
            correlation_dict['win_ratio'].append(win_ratio(df_temp[x] * df_temp[y]))

    # Convert to DataFrame for better readability
    result_df = pd.DataFrame(correlation_dict)
    return result_df


def calculate_corr_by_bucket(df, target_col, bucket_size_days=7):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['bucket'] = (df.index - df.index[0]).days // bucket_size_days
    corr_results = []
    for bucket, group in tqdm(df.groupby('bucket'), total=df['bucket'].nunique(), desc="Calculating Correlations"):
        corr_dict = {'bucket': bucket}
        for col in df.columns:
            if col != target_col and col != 'bucket':
                corr_dict[col] = group[col].corr(group[target_col])
        corr_results.append(corr_dict)
    result_df = pd.DataFrame(corr_results)
    return result_df

def calc_auto_corr(df_in, cols, lags, freq='1s', missing_handle='ffill'):
    if type(cols) == str:
        cols = [cols]

    if np.issubdtype(type(lags[0]), np.number): #otherwise timedelta would think it is in ns unit
        lags = [str(int(_)) + 's' for _ in lags]

    def __calc__(col):
        if missing_handle == 'ffill':
            df = df_in[[col]].resample(freq).last().ffill()
        elif missing_handle == 'ff0' :
            df = df_in[[col]].resample(freq).last().fillna(0)
        else:
            print(f'missing handle must be either ffill or ff0, not {missing_handle}')

        for lag in lags:
            #df[f'{col}_{lag}'] = df[col].shift(lag) #fix this RH
            lag_in_sample = round(pd.Timedelta(lag).total_seconds() / pd.Timedelta(freq).total_seconds())
            df[f'{col}_{lag}_sec'] = df[col].shift(lag_in_sample)
        df_corr = df[[c for c in df if col in c]].corr()
        df_corr.index = np.array(['lag_0s'] + ['lag_' + l for l in lags])
        df_corr = df_corr.iloc[1:, :].copy()
        return df_corr[[col]].copy()

    df_res = [__calc__(col) for col in cols]
    df_res = pd.concat(df_res, axis=1).T
    df_res.columns = ['lag_' + l for l in lags]
    return df_res

#given two columns and list of lags, cal cross corr
def calc_cx_corr(df_in, x_col, y_col, lags, unit, sample_mul, dir='forward'):
    if np.issubdtype(type(lags[0]), np.number):  # otherwise timedelta would think it is in ns unit
        lags = [str(int(round(_))) + unit for _ in lags]

    freq = f'{sample_mul}{unit}'
    df = df_in[[x_col, y_col]].resample(freq).last(freq).ffill()

    lag_in_samples = [int(round(pd.Timedelta(lag).total_seconds() / pd.Timedelta(freq).total_seconds())) for lag in
                      lags]

    def __calc__(lag_in_sample):
        lag_col = f'{y_col}_{lag_in_sample}'
        if dir == 'forward':
            df[lag_col] = df[y_col].shift(-lag_in_sample)
        else:
            df[lag_col] = df[y_col].shift(lag_in_sample)

        return df[[x_col, lag_col]].corr().iloc[0, 1]

    df_res = {f'{lis}_sample': [__calc__(lis)] for lis in lag_in_samples}
    return pd.DataFrame(df_res)

#exp weighted beta of col_x vs col_y
def calc_weighted_correlations(df, first_cols, second_cols, weight_col):
    correlations = []
    for col1 in first_cols:
        row = []
        for col2 in second_cols:
            correlation = np.corrcoef(df[col1], df[col2], aweights=df[weight_col])[0, 1]
            row.append(correlation)
        correlations.append(row)

    correlation_df = pd.DataFrame(correlations, index=first_cols, columns=second_cols)

    return correlation_df

calculate_weighted_correlations = calc_weighted_correlations


def calc_exp_weighted_beta_(df, x_col, y_col, dt, hls, std_mul=np.inf):
    # check dt must be number of np array
    if not isinstance(dt, np.ndarray) or not np.issubdtype(dt.dtype, np.number):
        raise ValueError("dt must be a numpy array of numbers")

    x_std = df[x_col].std()
    y_std = df[y_col].std()

    mx, my = df[x_col].mean(), df[y_col].mean()
    x = df[x_col].clip(mx - std_mul * x_std, mx + std_mul * x_std).values
    y = df[y_col].clip(my - std_mul * y_std, my + std_mul * y_std).values

    for hl in hls:
        xtx = calc_ts_ems(x * x, dt, hl)
        xty = calc_ts_ems(x * y, dt, hl)
        df[f'{x_col}_VS_{y_col}_beta_{hl}'] = xty / xtx

def bucket_corr(df, x_cols, y_col, bucket_col):
    df_corr = df.groupby(bucket_col, observed=False)[x_cols + [y_col]].apply(lambda x: x.corr().loc[x_cols, y_col])
    return df_corr

def bucket_mmks(df, x_cols, y_col, bucket_col): #mean and median keeping sign
    def __calc__(df):
        results = {}
        for x in x_cols:
            mean   = np.mean(df[x].apply(np.sign) * df[y_col])
            median = np.median(df[x].apply(np.sign) * df[y_col])
            results[x] = f'mean : {mean:g}, median : {median:g}'
        return pd.Series(results)

    df_corr = df.groupby(bucket_col, observed=False)[x_cols + [y_col]].apply(__calc__)
    return df_corr


def bucket_rank(df, x_cols, y_col, bucket_col):
    def __calc__(df):
        results = {}
        for x in x_cols:
            results[x] = pearsonr(df[x], df[y_col].values)[0]
        return pd.Series(results)

    df_corr = df.groupby(bucket_col, observed=False)[x_cols + [y_col]].apply(__calc__)
    return df_corr


def bucket_win(df, x_cols, y_col, bucket_col, threshold=0.0, strict=False):
    def __calc__(group):
        results = {}
        for x_col in x_cols:
            if strict:
                expression = f'{x_col} * {y_col} > {threshold}'
            else:
                expression = f'{x_col} * {y_col} >= {threshold}'
            positive_product = group.eval(expression).mean()
            results[x_col] = positive_product
        return pd.Series(results)

    return df.groupby(bucket_col).apply(__calc__)

def bucket_lr(df, x_cols, y_col, bucket_col, fit_intercept=False):
    def __calc__(group):
        lr = LinearRegression(fit_intercept = fit_intercept)
        X = group[x_cols].values
        y = group[y_col].values
        lr.fit(X, y)
        return np.sqrt(lr.score(X, y))

    return df.groupby(bucket_col).apply(__calc__)

def bucket_analyze_generic(df, x_cols, bucket_col, funcs):
    return df.groupby(bucket_col)[x_cols].agg(funcs)

def proc_col_(df, y_col, scaler, std_mul = 4.0):
    df[y_col] -= df[y_col].mean()
    std = df[y_col].std()
    df[y_col] = df[y_col].clip(-std_mul * std, std_mul * std)
    df[y_col] *= scaler

def target_col_corr(df, target_column):
    correlations = {}
    target_series = df[target_column]
    for column in tqdm(df.columns, desc="Calculating Correlations"):
        if column != target_column:
            correlations[column] = target_series.corr(df[column])
    return correlations


# def calculate_corr_by_bucket(df, target_col, bucket_size_days=7):
#     # 确保index是datetime类型
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#
#     # 按照每七天划分bucket
#     df['bucket'] = (df.index - df.index[0]).days // bucket_size_days
#
#     # 存储结果的列表
#     corr_results = []
#
#     # 计算每个bucket中各列与target列的相关性
#     for bucket, group in tqdm(df.groupby('bucket'), total=df['bucket'].nunique(), desc="Calculating Correlations"):
#         corr_dict = {'bucket': bucket}
#         for col in df.columns:
#             if col != target_col and col != 'bucket':
#                 corr_dict[col] = group[col].corr(group[target_col])
#         corr_results.append(corr_dict)
#
#     # 将结果转换为DataFrame
#     result_df = pd.DataFrame(corr_results)
#
#     return result_df


def calculate_windowed_corr(df, ratio, window_size=24 * 60 * 60, columns_to_compare=None):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if columns_to_compare is None:
        column_pairs = combinations(df.columns, 2)
    else:
        column_pairs = []
        for col_pair in columns_to_compare:
            if all(col in df.columns for col in col_pair):
                column_pairs.append(col_pair)
            else:
                missing_cols = [col for col in col_pair if col not in df.columns]
                raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
    correlations = {}
    step_length = int(window_size * ratio)
    for col1, col2 in column_pairs:
        corrs = []
        for start in range(0, len(df) - window_size + 1, step_length):
            window = df.iloc[start:start + window_size]
            corr = window[col1].corr(window[col2])
            corrs.append(corr)
        correlations[(col1, col2)] = corrs
    return correlations


##YYT

def calculate_rolling_corr_selfmade(df, window_size=24 * 60 * 60, columns_to_compare=None):
    df = df.dropna()
    # Check if the DataFrame index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if columns_to_compare is None:
        column_pairs = list(combinations(df.columns, 2))  # Generate all possible combinations of column pairs
    else:
        column_pairs = {
            tuple(sorted([col1, col2]))  # Sort column names to ensure consistent order
            for col1, col2 in columns_to_compare
            if col1 in df.columns and col2 in df.columns
        }
    rolling_stats = {
        column: {
            'mean': df[column].rolling(window=window_size).mean(),  # Rolling mean
            'mean_square': (df[column] ** 2).rolling(window=window_size).mean(),  # Rolling mean of squared values
        }
        for column in {col for pair in column_pairs for col in pair}  # Unique set of column names across all pairs
    }
    rolling_corrs = {}
    for col1, col2 in column_pairs:
        mean_xy = (df[col1] * df[col2]).rolling(window=window_size).mean()
        mean_x = rolling_stats[col1]['mean']
        mean_y = rolling_stats[col2]['mean']
        mean_x2 = rolling_stats[col1]['mean_square']
        mean_y2 = rolling_stats[col2]['mean_square']
        numerator = mean_xy - mean_x * mean_y
        denominator = ((mean_x2 - mean_x ** 2) * (mean_y2 - mean_y ** 2)).apply(lambda z: max(z, 0))
        rolling_corr = numerator / denominator ** 0.5
        valid_rolling_corr = rolling_corr.dropna()  # Drop NaN values
        if valid_rolling_corr.empty:
            print(f"Insufficient data to calculate rolling correlation between {col1} and {col2} for a full window.")
            continue
        rolling_corrs[(col1, col2)] = valid_rolling_corr
    return rolling_corrs


def calculate_rolling_corr_pandas(df, window_size=24 * 60 * 60, columns_to_compare=None):
    df = df.dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if columns_to_compare is None:
        column_pairs = combinations(df.columns, 2)
    else:
        column_pairs = []
        for col_pair in columns_to_compare:
            if all(col in df.columns for col in col_pair):
                column_pairs.append(col_pair)
            else:
                missing_cols = [col for col in col_pair if col not in df.columns]
                raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
    rolling_corrs = {}
    for col1, col2 in column_pairs:
        rolling_corr = df[col1].rolling(window=window_size).corr(df[col2])
        valid_rolling_corr = rolling_corr.dropna()
        if valid_rolling_corr.empty:
            print(f"Insufficient data to calculate rolling correlation between {col1} and {col2} for a full window.")
            continue
        rolling_corrs[(col1, col2)] = valid_rolling_corr
    return rolling_corrs
