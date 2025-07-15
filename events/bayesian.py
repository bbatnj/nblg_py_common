import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_event(event, ret_df, target = 'mid_last_ret_600_n'):
    event.index = pd.to_datetime(event.index).tz_convert(None)
    event.columns = event.columns.map(lambda x: f"{x[0]}_{x[1]}")

    event_merged = pd.merge_asof(event, ret_df, left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta('1m'))
    event_merged = event_merged.dropna(subset = ret_df.columns)
    event_merged.iloc[0] = event_merged.iloc[0]
    event_merged[event.columns.to_list()[:2]] = event_merged[event.columns.to_list()[:2]].fillna(method='ffill')
    z = pd.DataFrame(index=event_merged.index)

    # 复制前两列
    # 处理 shift(1) 并 ffill 后填充第一个值为 0
    z[event.columns.to_list()] = event_merged[event.columns.to_list()].ffill()  
    # 添加后四列
    z[ret_df.columns.to_list()] = event_merged[ret_df.columns.to_list()].shift(1)
    z = z.fillna(0)
    # 重命名列名，就按照z.columns的顺序来,分别叫做z0,z1,.....
    z.columns = [f"z{i}" for i in range(z.shape[1])]

    event_merged = pd.concat([event_merged, z], axis=1)
    event_merged = event_merged.dropna(how="all", subset=event.columns)
    Z = event_merged[z.columns].values
    y_binary = (event_merged[target] > 0).astype(int)
    y = event_merged[target].values
    X_obs = event_merged[event.columns].values
    #i就是X_obs如果对应位置不是NaN就是1，否则是0
    i = ~np.isnan(X_obs).astype(int)+2
    return {'Z': Z, 'y': y, 'X_obs': X_obs, 'i': i, 'y_binary':y_binary}, event_merged


def fit_pymc_model(Z, X_obs, y, i, draws=1500, tune=500, cores=2, plot=True):
    """
    PyMC 贝叶斯模型：X 作为 (n_samples, 2*d_x) 的变量, 统一建模。

    参数：
    - Z: 预测 X 的变量 (n_samples, d_z)
    - X_obs: X 观测值 (含 NaN，形状 (n_samples, 2*d_x))
    - y: 目标变量 (binary 或 {-1, 0, 1})
    - i: 缺失指示变量 (0=缺失, 1=观测) (n_samples, 2*d_x)

    返回：
    - trace: PyMC 采样结果
    """
    d_x = X_obs.shape[1]  # 现在 X 维度是 2*d_x
    d_z = Z.shape[1]  # Z 的维度

    with pm.Model() as model:
        # 先验分布
        beta = pm.Normal("beta", mu=0, sigma=1, shape=d_x)
        
        # 估计 X 的权重 (Z -> X)
        true_w = pm.Normal("true_w", mu=0, sigma=1, shape=(d_z, d_x))

        # 用 Z 估计 X
        mu_x = pm.Normal("mu_x", mu=Z @ true_w, sigma=0.2, shape=(Z.shape[0], d_x))

        # 观测 X
        X_imputed = pm.Normal("X_imputed", mu=mu_x, sigma=0.2, observed=X_obs)

        # 预测模型
        y_hat = pm.math.sum(beta * (i * X_imputed + (1 - i) * mu_x), axis=1)

        # 观测 y
        if set(np.unique(y)).issubset({0, 1}):
            y_obs = pm.Bernoulli("y_obs", p=pm.math.sigmoid(y_hat), observed=y)
        elif set(np.unique(y)).issubset({-1, 0, 1}):
            y_obs = pm.Categorical("y_obs", pm.math.softmax(y_hat), observed=y + 1)
        else:
            sigma_y = pm.HalfCauchy("sigma_y", beta=1)
            y_obs = pm.Normal("y_obs", mu=y_hat, sigma=sigma_y, observed=y)

        # 采样
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True, cores=cores)
    
    if plot:
        az.plot_posterior(trace, var_names=["beta", "true_w"])
        plt.show()

    return trace

def predict_pymc_model(trace, X_obs, Z_new, i_new):
    """
    使用拟合的 PyMC 模型进行预测，基于后验分布进行不确定性估计。

    参数：
    - trace: PyMC 采样结果 (InferenceData)
    - X_obs: 观测到的 X (n_samples, 2*d_x)
    - Z_new: 新的 Z 数据 (n_samples_new, d_z)
    - i_new: 缺失指示变量 (n_samples_new, 2*d_x)

    返回：
    - y_pred_mean: 预测的 y 的均值
    - y_pred_samples: 预测的 y 的所有采样值 (n_samples_new, n_draws * n_chains)
    """
    posterior = trace.posterior
    
    # 获取 posterior 的采样值
    beta_samples = posterior["beta"].values  # 形状 (n_chains, n_draws, d_x)
    true_w_samples = posterior["true_w"].values  # 形状 (n_chains, n_draws, d_z, d_x)

    # 获取采样维度
    n_chains, n_draws, d_x = beta_samples.shape
    n_samples_new, d_z = Z_new.shape

    # 处理 NaN
    X_obs[np.isnan(X_obs)] = 0

    # 初始化存储所有采样 y_pred 的数组
    y_pred_samples = np.zeros((n_samples_new, n_chains * n_draws))

    # 遍历每个采样
    idx = 0
    for chain in range(n_chains):
        for draw in range(n_draws):
            beta = beta_samples[chain, draw]  # (d_x,)
            true_w = true_w_samples[chain, draw]  # (d_z, d_x)

            # 预测 X 的均值
            mu_x_new = Z_new @ true_w  # (n_samples_new, d_x)

            # 计算 y 预测值
            y_pred_samples[:, idx] = np.sum(beta * (i_new * X_obs + (1 - i_new) * mu_x_new), axis=1)
            idx += 1

    # 计算均值
    y_pred_mean = np.mean(1/(1+np.exp(-y_pred_samples)), axis=1)

    return y_pred_mean, y_pred_samples




# import numpy as np

# # 设定随机种子，确保可复现性
# np.random.seed(42)

# n_samples = 100  # 样本数
# dim_z1, dim_z2 = 2, 3  # Z 的维度 (三维)
# dim_x1, dim_x2 = 2, 2  # X1 和 X2 的维度 (二维)

# # 生成 Z (独立变量, 三维)
# Z = np.random.normal(0, 1, (n_samples, dim_z1, dim_z2))

# # 生成 X1, X2 (与 Z 相关)
# true_w1 = np.random.normal(0, 1, (dim_z1, dim_z2, dim_x1))  # X1 的真实权重
# true_w2 = np.random.normal(0, 1, (dim_z1, dim_z2, dim_x2))  # X2 的真实权重

# # 计算 X1 和 X2
# X1 = np.einsum("ijk,jkl->il", Z, true_w1) + np.random.normal(0, 0.2, (n_samples, dim_x1))
# X2 = np.einsum("ijk,jkl->il", Z, true_w2) + np.random.normal(0, 0.2, (n_samples, dim_x2))

# # 生成缺失指示变量 i1, i2
# i1 = np.random.choice([0, 1], size=(n_samples, dim_x1), p=[0.2, 0.8])  # 20% 缺失
# i2 = np.random.choice([0, 1], size=(n_samples, dim_x2), p=[0.3, 0.7])  # 30% 缺失

# # 生成 y (考虑 X1 和 X2 多个维度)
# true_beta1 = np.random.normal(0, 1, (dim_x1,))
# true_beta2 = np.random.normal(0, 1, (dim_x2,))
# true_beta3 = np.random.normal(0, 1, (dim_x1,))
# true_beta4 = np.random.normal(0, 1, (dim_x2,))

# # 计算 y 值
# mu_x1 = np.einsum("ijk,jkl->il", Z, true_w1)
# mu_x2 = np.einsum("ijk,jkl->il", Z, true_w2)

# y = np.sum(
#     true_beta1 * i1 * X1 + 
#     true_beta2 * i2 * X2 +
#     true_beta3 * (1 - i1) * mu_x1 +
#     true_beta4 * (1 - i2) * mu_x2, 
#     axis=1  # 对多个维度求和，得到 (n_samples,)
# )

# # 生成二分类目标变量 y (根据 y 作为概率生成 0, 1)
# binary_y = np.random.binomial(1, 1 / (1 + np.exp(-y)))

# # 处理缺失值
# X1_obs = np.where(i1 == 1, X1, np.nan)  # 仅在 i1=1 时保留 X1 观测值
# X2_obs = np.where(i2 == 1, X2, np.nan)  # 仅在 i2=1 时保留 X2 观测值

# # 输出数据形状检查
# trace = fit_pymc_model(Z, X1_obs, X2_obs, binary_y, i1, i2, draws=1000, cores=20)