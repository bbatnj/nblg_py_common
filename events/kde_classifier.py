from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np  
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss 
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

class KDERegressor:
    def __init__(self, bandwidth=None, kernel='gaussian'):
        """
        使用核密度估计 (KDE) 进行非参数回归
        
        参数:
        - bandwidth: KDE 的带宽，控制平滑程度；如果为 None，则自动优化
        - kernel: 核函数类型，默认是 'gaussian'
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_xy = None  # 联合密度估计模型
        self.x1_train = None
        self.y_train = None

    def _kde_bandwidth_optimization(self, X, y, method="cv", bw_range=np.logspace(-1, 1, 20)):
        """
        通过交叉验证 (CV) 或留一法 (LOO) 选择最优 bandwidth
        
        参数:
        - X: 训练数据 (N, 1)
        - y: 目标变量 (N,)
        - method: 选择 "cv" (交叉验证) 或 "loo" (留一法)
        - bw_range: 供搜索的 bandwidth 值
        
        返回:
        - best_bandwidth: 选择的最优 bandwidth
        """
        xy_train = np.hstack([X, y.reshape(-1, 1)])

        if method == "cv":
            # 使用 GridSearchCV 找到最优 bandwidth
            grid = GridSearchCV(KernelDensity(kernel=self.kernel), 
                                {'bandwidth': bw_range}, 
                                cv=5)  # 5 折交叉验证
            grid.fit(xy_train)
            return grid.best_params_['bandwidth']
        
        elif method == "loo":
            # 使用 Leave-One-Out (LOO) 方法
            loo = LeaveOneOut()
            best_bw = None
            best_score = -np.inf  # 对数似然最大化

            for bw in bw_range:
                scores = []
                for train_idx, test_idx in loo.split(xy_train):
                    kde = KernelDensity(kernel=self.kernel, bandwidth=bw)
                    kde.fit(xy_train[train_idx])
                    score = kde.score(xy_train[test_idx])
                    scores.append(score)
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_bw = bw

            return best_bw

        else:
            raise ValueError("method 必须是 'cv' 或 'loo'")

    def fit(self, X, y, optimize_bandwidth=True, method="loo"):
        """
        拟合 KDE 回归模型，估计 P(X, Y)
        
        参数:
        - X: 训练特征 (N, 1) 形状的 NumPy 数组
        - y: 目标变量 (N,) 形状的 NumPy 数组
        - optimize_bandwidth: 是否自动优化 bandwidth
        - method: 选择 "cv" (交叉验证) 或 "loo" (留一法)
        """
        self.x1_train = X
        self.y_train = y

        # 自动选择 bandwidth
        if self.bandwidth is None and optimize_bandwidth:
            self.bandwidth = self._kde_bandwidth_optimization(X, y, method=method)

        xy_train = np.hstack([X, y.reshape(-1, 1)])
        self.kde_xy = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde_xy.fit(xy_train)

    def predict(self, X_test, y_grid=None):
        """
        计算 E[Y | X] ≈ ∑ y * P(y | x)
        
        参数:
        - X_test: (M, 1) 形状的 NumPy 数组，表示要预测的 X 值
        - y_grid: (G, 1) 形状的 NumPy 数组，表示用于积分的 Y 取值范围 (可选)
        
        返回:
        - y_pred: (M,) 形状的 NumPy 数组，表示预测的 y 值
        """
        if self.kde_xy is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        if y_grid is None:
            y_grid = np.linspace(self.y_train.min() - 1, self.y_train.max() + 1, 100).reshape(-1, 1)

        xy_vals = np.array(np.meshgrid(X_test.ravel(), y_grid.ravel())).T.reshape(-1, 2)
        log_prob_xy = self.kde_xy.score_samples(xy_vals)
        prob_xy = np.exp(log_prob_xy).reshape(len(X_test), len(y_grid))

        # 避免除 0
        eps = 1e-9
        prob_x = prob_xy.sum(axis=1)
        prob_x[prob_x < eps] = eps
        cond_exp_y = (prob_xy * y_grid.T).sum(axis=1) / prob_x

        return cond_exp_y
    
class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    一个加权融合的分类器，使用 LOO 计算 KDE、GNB、LR 的评分，并融合预测概率。
    """

    def __init__(self, lr_only = False):
        self.lr_only = lr_only
        self.models = {}  # 存储 {'KDE': model, 'GNB': model, 'LR': model}
        self.weights = None  # 存储 {'KDE': w1, 'GNB': w2, 'LR': w3}
        self.loo_scores = {}  # 存储 LOO 评分
        self.not_useful = False
        
    def fit(self, X, y):
        """
        训练 KDE, GaussianNB 和 Logistic Regression，并计算 LOO 评分
        
        :param X: 特征矩阵
        :param y: 目标变量
        """
        #如果X和y传入得是df,我们需要转换成np.array
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if len(y) < 5:
            raise ValueError("❌ 数据量不足，至少需要 5 个样本！")

        # 训练初始模型
        kde = KDEClassifier()
        gnb = GaussianNB()
        lr = LogisticRegression()

        # 计算 LOO 评分
        loo = LeaveOneOut()
        y_list = []
        predicts = {'KDE': [], 'GNB': [], 'LR': []} if not self.lr_only else {  'LR': []}
        probs = {'KDE': [], 'GNB': [], 'LR': []} if not self.lr_only else { 'LR': []}

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx] 

            # 确保训练集中有 0 和 1
            if len(np.unique(y_train)) < 2:
                continue  # 跳过该次 LOO
            y_list.append(y_test[0])
            # 训练模型
            kde.fit(X_train, y_train) if not  self.lr_only else None
            gnb.fit(X_train, y_train) if not  self.lr_only else None
            lr.fit(X_train, y_train)

            # 计算概率
            kde_prob = kde.predict_proba(X_test)[:, 1] if not self.lr_only else None
            gnb_prob = gnb.predict_proba(X_test)[:, 1] if not  self.lr_only else None
            lr_prob = lr.predict_proba(X_test)[:, 1]

            # 记录概率
            probs['KDE'].append(kde_prob) if not self.lr_only else None
            probs['GNB'].append(gnb_prob) if not self.lr_only else None
            probs['LR'].append(lr_prob)

            # 计算准确率
            kde_pred = (kde_prob > 0.5).astype(int) if not self.lr_only else None
            gnb_pred = (gnb_prob > 0.5).astype(int) if not self.lr_only else None
            lr_pred = (lr_prob > 0.5).astype(int)

            predicts['KDE'].append(kde_pred) if not self.lr_only else None
            predicts['GNB'].append(gnb_pred) if not self.lr_only else None
            predicts['LR'].append(lr_pred)

        # 计算 LOO 评分
        self.loo_scores = {}
        for model in ['KDE', 'GNB', 'LR'] if not self.lr_only else [ 'LR']:
            predicts[model] = np.array(predicts[model]).flatten()
            probs[model] = np.array(probs[model]).flatten()
            # #检测一下y_list, predicts[model], probs[model]的长度是否一致，不一致的话print
            # if len(y_list) != len(predicts[model]) or len(y_list) != len(probs[model]):
            #     print(f"❌ {model} 预测结果长度不一致！")
            #     print(y_list)
            #     print(predicts[model])
            #     print(probs[model])
            self.loo_scores[model] = {
                "accuracy": accuracy_score(y_list, predicts[model]),
                "log_loss": log_loss(y_list, probs[model]) if len(np.unique(y_list)) > 1 else -np.log(0.5)
            }

        # 计算权重
        acc_scores = np.array([self.loo_scores["KDE"]["accuracy"], 
                               self.loo_scores["GNB"]["accuracy"],
                               self.loo_scores["LR"]["accuracy"]])   if not  self.lr_only else np.array( [ self.loo_scores["LR"]["accuracy"]])

        log_loss_scores = np.array([self.loo_scores["KDE"]["log_loss"],
                                    self.loo_scores["GNB"]["log_loss"],
                                    self.loo_scores["LR"]["log_loss"]]) if not  self.lr_only else np.array([self.loo_scores["LR"]["log_loss"]])

        weighted_scores = (acc_scores+0.0001) / (1 + log_loss_scores)
        self.weights = weighted_scores / weighted_scores.sum()

        ##我们检测一下acc_scores, 如果acc_scores里面全是小于0.4的，我们就认为这个模型不可用，直接输出每次都是0.5
        if np.all(acc_scores < 0.4):
            self.not_useful = True
            print(f" ❌ 所有模型准确率均小于 0.4, 不可用！")
            return 
        elif  np.all(acc_scores < 0.5) and np.all(log_loss_scores > np.log(2)):
            self.not_useful = True
            print(f" ❌ 所有模型准确率均小于 0.5, 且所有模型 log_loss 均大于 ln(2)，不可用！")
            return
        
        # 重新训练最终模型
        kde.fit(X, y) if not  self.lr_only else None
        gnb.fit(X, y) if not  self.lr_only else None
        lr.fit(X, y)

        self.models = {"KDE": kde, "GNB": gnb, "LR": lr} if not self.lr_only else { "LR": lr}
        #如果weight含有NaN，说明loo_scores中有NaN，需要检查loo_scores
        if np.isnan(self.weights).any():
            print("❌ 加权系数含有 NaN，请检查 LOO 计算结果！")
            print(acc_scores)
            print(log_loss_scores)
            print(weighted_scores)
            print(self.weights)
        else:
            print(f"✅ LOO 计算完成！最终权重: KDE={self.weights[0]:.2f}, GNB={self.weights[1]:.2f}, LR={self.weights[2]:.2f}") if not self.lr_only else print(f"✅ LOO 计算完成！ ")

    def predict_proba(self, X, method='weighted'):
        """
        计算加权融合后的概率
        
        :param X: 输入特征矩阵
        :return: (N, 2) 形式的预测概率
        """
        if self.not_useful:
            return np.array([[0.5, 0.5]]*len(X))
        
        if not self.models:
            raise ValueError("❌ 模型未训练，请先调用 fit() 方法！")
        
        if self.weights is None:
            raise ValueError("❌ 加权系数未计算，请检查 fit() 是否成功执行！")

        # 计算每个模型的概率
        kde_probs = self.models["KDE"].predict_proba(X)[:, 1]  if not self.lr_only else None
        gnb_probs = self.models["GNB"].predict_proba(X)[:, 1] if not self.lr_only else None
        lr_probs = self.models["LR"].predict_proba(X)[:, 1]

        if method == 'KDE':
            return np.vstack([1 - kde_probs, kde_probs]).T
        elif method == 'GNB':
            return np.vstack([1 - gnb_probs, gnb_probs]).T
        elif method == 'LR':
            return np.vstack([1 - lr_probs, lr_probs]).T
        
        final_prob = (self.weights[0] * kde_probs +
                      self.weights[1] * gnb_probs +
                      self.weights[2] * lr_probs) if not self.lr_only else   lr_probs

        # 返回二分类概率
        return np.vstack([1 - final_prob, final_prob]).T

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    KDE 分类器，类似 sklearn 风格的 fit/predict_proba 方法。
    适用于 P(y=1 | x) 估计。
    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth  # 预设 bandwidth，若 None 则自动优化
        self.kde_1 = None  # P(y=1 | x)
        self.kde_0 = None  # P(y=0 | x)
        self.bw_1 = None
        self.bw_0 = None
        self.only_class = None  # 记录是否只有单个类别

    def fit(self, X, y):
        """
        训练 KDE 估计器，分别估计 y=1 和 y=0 的概率密度
        """
        X_1 = X[y == 1]
        X_0 = X[y == 0]

        if len(X_1) == 0:
            self.only_class = 0  # 只有 y=0 类别
            return
        elif len(X_0) == 0:
            self.only_class = 1  # 只有 y=1 类别
            return

        self.only_class = None  # 说明两个类别都有

        # 自动选择 bandwidth
        self.bw_1 = self.bandwidth if self.bandwidth else self._kde_bandwidth_optimization(X_1)
        self.bw_0 = self.bandwidth if self.bandwidth else self._kde_bandwidth_optimization(X_0)

        # 训练 KDE
        self.kde_1 = KernelDensity(bandwidth=self.bw_1, kernel='gaussian').fit(X_1)
        self.kde_0 = KernelDensity(bandwidth=self.bw_0, kernel='gaussian').fit(X_0)

    def predict_proba(self, X):
        """
        计算 P(y=1 | x) 概率
        """
        X = np.atleast_2d(X)  # 确保 X 是二维

        # 处理只有单一类别的情况
        if self.only_class == 0:
            return np.hstack([np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1))])  # 全部预测为 y=0
        elif self.only_class == 1:
            return np.hstack([np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))])  # 全部预测为 y=1

        # 计算对数概率密度
        log_prob_1 = self.kde_1.score_samples(X)
        log_prob_0 = self.kde_0.score_samples(X)

        # 防止 underflow
        log_prob_1 = np.clip(log_prob_1, -700, None)
        log_prob_0 = np.clip(log_prob_0, -700, None)

        prob_1 = np.exp(log_prob_1)
        prob_0 = np.exp(log_prob_0)

        # 归一化
        denominator = prob_1 + prob_0
        prob_y1 = np.where(denominator == 0, 0.5, prob_1 / denominator)

        return np.vstack([1 - prob_y1, prob_y1]).T  # 返回 (N, 2) 形状的概率

    def _kde_bandwidth_optimization(self, X, bandwidth_range=np.logspace(-2, 1, 20)):
        """
        选择最佳 bandwidth，但避免交叉验证崩溃
        """
        if len(X) < 2:  # 数据点太少，直接返回默认 bandwidth
            return 0.5
        cv_folds = min(3, len(X))  # 不能超过数据点数量
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidth_range},
                            cv=cv_folds)  # 自动适应数据量
        grid.fit(X)
        return grid.best_params_['bandwidth']
