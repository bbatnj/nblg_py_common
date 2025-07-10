import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.integrate import quad
from scipy.stats import pearsonr, spearmanr, kendalltau

def loocv_corr(df, col_x, col_y, method='pearson'):
    n = len(df)
    r_values = []

    corr_func = {
        'pearson': pearsonr,
        'spearman': spearmanr,
        'kendall': kendalltau
    }[method]

    for i in range(n):
        df_loo = df.drop(index=df.index[i])
        r, _ = corr_func(df_loo[col_x], df_loo[col_y])
        r_values.append(r)

    r_values = np.array(r_values)
    return {
        'Mean': np.mean(r_values),
        'Min': np.min(r_values),
        'Max': np.max(r_values),
        'Std': np.std(r_values)
    }

def sigmoid_y(y):
    y_abs=abs(y)
    y_direction=np.sign(y)
    return y_direction*(8 / (1 + np.exp(-y_abs * 0.05)) - 4)

def labeled_y(y):
    if y > 10:
        return 1
    elif y < -10:
        return -1
    else:
        return 0
        
def analyze_xy(df, col_x, col_y, weighted_y=None):
    df = df[[col_x, col_y]].copy().dropna()

    results = {}
    methods = {
        'Pearson': pearsonr,
        'Spearman': spearmanr,
        'Kendall': kendalltau
    }

    if weighted_y:
        df['weighted_y'] = df[col_y].apply(weighted_y)
        target_col = 'weighted_y'
    else:
        target_col = col_y

    for name, func in methods.items():
        corr, p_value = func(df[col_x], df[target_col])
        label = ' (Weighted)' if weighted_y else ''
        results[f'{name}{label} Corr'] = corr
        results[f'{name}{label} P-value'] = p_value
        
        loocv_results = loocv_corr(df, col_x, target_col, name.lower())
        for stat, value in loocv_results.items():
            results[f'LOOCV{label} {name} Corr {stat}'] = value

    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    return results_df


class KDEEstimator:
    def __init__(self, df, x_col=None, y_col=None, bandwidth=None, use_log_pdf=True):
        if x_col is not None and y_col is not None:
            self.df = df[[x_col, y_col]]
        else:
            self.df = df
        self.columns = self.df.columns
        self.use_log_pdf = use_log_pdf
        self.scaled_data, self.scaler = self.standardize_data(self.df)
        if bandwidth is None:
            self.best_bandwidth = self.select_bandwidth(self.scaled_data)
        else:
            self.best_bandwidth = bandwidth
        self.kde = KernelDensity(bandwidth=self.best_bandwidth)
        self.kde.fit(self.scaled_data)
        
        self.kde_x = KernelDensity(bandwidth=self.best_bandwidth)
        self.kde_x.fit(self.scaled_data[:, [0]])
        
        self.xx, self.yy, self.pdf = self.kde_joint_pdf(self.scaled_data, self.best_bandwidth)
        self.xx_orig = self.scaler.inverse_transform(np.vstack([self.xx.ravel(), self.yy.ravel()]).T)[:, 0].reshape(self.xx.shape)
        self.yy_orig = self.scaler.inverse_transform(np.vstack([self.xx.ravel(), self.yy.ravel()]).T)[:, 1].reshape(self.yy.shape)
        
    def standardize_data(self, df):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        self.means = scaler.mean_
        self.scales = scaler.scale_
        return scaled_data, scaler
    def custom_scorer(self, estimator, X):
        log_density = estimator.score_samples(X)
        #y = X[:, 1]
        density= np.exp(log_density)
        return (density ).mean()
    
    def select_bandwidth(self, data):
        params = {'bandwidth': np.linspace(0.01, 1, 30)}
        loo = LeaveOneOut()
        if self.use_log_pdf:
            grid = GridSearchCV(KernelDensity(), params, cv=loo)
        else:
            grid = GridSearchCV(KernelDensity(), params, cv=loo, scoring=self.custom_scorer)
        grid.fit(data)
        return grid.best_estimator_.bandwidth