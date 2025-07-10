import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy import stats


def analyze_evaluation_results(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = evaluation_results.select_dtypes(include=[np.number]).columns
    data = evaluation_results[numeric_cols]
    desc = data.describe()
    additional_stats = pd.DataFrame(index=['t_stat', 'p_value', '95% CI Lower', '95% CI Upper', '0_in_95%'],
                                    columns=data.columns)

    for col in data.columns:
        values = data[col].dropna()
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        sem = std / np.sqrt(n)

        t_stat, p_value = stats.ttest_1samp(values, popmean=0)
        ci_low, ci_upp = stats.t.interval(
            df=n - 1,
            loc=mean,
            scale=sem,
            confidence=0.95
        )

        additional_stats.loc['t_stat', col] = t_stat
        additional_stats.loc['p_value', col] = p_value
        additional_stats.loc['95% CI Lower', col] = ci_low
        additional_stats.loc['95% CI Upper', col] = ci_upp
        additional_stats.loc['0_in_95%', col] = (ci_low <= 0) & (ci_upp >= 0)
    result = pd.concat([desc, additional_stats])
    return result.T


def mape(y_hat, y_true):
    mae = abs(y_true - np.mean(y_true)).mean()
    mape = 1 - abs(y_true - y_hat).mean() / mae
    return mape
    # import torch
    # from torchmetrics.regression import MeanAbsolutePercentageError
    # mean_abs_percentage_error = MeanAbsolutePercentageError()
    # res = mean_abs_percentage_error(preds=torch.tensor(y_hat), target=torch.tensor(y))
    # return float(res)

def t_test_measure(dataframe):
    means = dataframe.mean()
    stds = dataframe.std()
    df_res = pd.DataFrame({'t_test': np.where(stds != 0, abs(means / stds), np.nan), 'Mean': abs(means)},
                          index=dataframe.columns)
    df_res.dropna(inplace=True)
    return df_res


def calc_cdf(df, x_col, x_s):
    x_s_sorted = np.sort(x_s)
    sorted_col = np.sort(df[x_col].values)
    cdf_col = np.searchsorted(sorted_col, x_s_sorted, side='right') / df.shape[0]
    return pd.DataFrame({x_col: x_s_sorted, 'CDF': cdf_col})


def cohens_d(group1, group2, down_sample_hf=1):
    if len(group1) == 0 or len(group2) == 0:
        return np.nan  
    diff = np.mean(group1) - np.mean(group2)
    n1, n2 = len(group1) / down_sample_hf, len(group2) / down_sample_hf
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return diff / np.sqrt(pooled_var)


def des(df, down_sample_hf=1):
    stats = {
        '75%': df.quantile(0.75),
        '25%': df.quantile(0.25),
        'mean': df.mean(),
        'median': df.median(),
        'min': df.min(),
        'max': df.max(),
        'std': df.std(),
        'kurt': df.kurt(),
        'skew': df.skew(),
        'mean/std': df.mean() / df.std(),
        'mean/std*sqrt(n/down_sample_hf)': (df.mean() / df.std()) * np.sqrt(df.count() / down_sample_hf)
    }
    return pd.DataFrame(stats).T


def group_percentiles(df, group_col, target_col, percentiles):
    result = df.groupby(group_col)[target_col].quantile(percentiles).unstack(level=1)
    result.columns = [f'{int(p*100)}%' for p in percentiles]
    return result.reset_index()

def group_analysis(df, grp_col, xcols):
    grouped = df.groupby(grp_col)[xcols]
    desc = grouped.describe()
    skew = grouped.agg('skew')
    skew.columns = pd.MultiIndex.from_product([skew.columns, ['skew']])
    kurt = grouped.apply(lambda x: pd.DataFrame.kurt(x))
    kurt.columns = pd.MultiIndex.from_product([kurt.columns, ['kurt']])
    group_stats = pd.concat([desc, skew, kurt], axis=1).T.sort_index(level=0)

    for value_col in xcols:
        cohens_d_rslt = {}
        unique_intervals = df[grp_col].unique()
        for interval in unique_intervals:
            group1 = df[df[grp_col] == interval][value_col]
            group2 = df[df[grp_col] != interval][value_col]
            d_value = cohens_d(pd.Series(group1), pd.Series(group2))
            cohens_d_rslt[interval] = d_value
        cohens_d_df = pd.DataFrame(list(cohens_d_rslt.items()), columns=['day_interval_index', 'Cohens d']).set_index(
            'day_interval_index')
        cohens_d_df.columns = pd.MultiIndex.from_product([[value_col], cohens_d_df.columns])
        group_stats = pd.concat([cohens_d_df.T, group_stats])

    return group_stats

def generalized_logistic(x, L, k, p):
    return L / pow(1 + np.exp(-k * x), p)

def fit_1d(target_f, df, x_col, y_col, p0):
    df = df.copy()
    fitted_param, _ = curve_fit(generalized_logistic, df[x_col].values, df[y_col].values, p0)
    df['fitted'] = target_f(df[x_col].values, *fitted_param.tolist())
    df[[x_col, y_col, 'fitted']].set_index(x_col).plot(grid=True)
    return fitted_param