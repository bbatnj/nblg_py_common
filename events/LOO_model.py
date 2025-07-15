import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from common.events.kde_classifier import KDERegressor 


class MultiEventLOORegressor:
    def __init__(self, df_dict, ret_df, **kwargs):
        self.event_dict = df_dict
        self.ret_df = ret_df
        self.target_col = kwargs.get('target_col', 'y')
        self.predict_point = kwargs.get('predict_point', None)
        self.single_models = {}
        self.multi_event_points = {}
        
    def single_event_train(self, event_name, event_df=None, lr_only = False):
        if event_df is None:
            event_df = self.event_dict[event_name]
        if event_name in self.single_models:
            return None
        event_df = pd.merge_asof(event_df, self.ret_df[[self.target_col]], left_index=True, right_index=True,
                              direction='backward', tolerance=pd.Timedelta('1m')).dropna(axis=0).dropna(how='all')
        if event_df.shape[0] < 5:
            # print(f"Event {event_name} has less than 5 data points, skip")
            self.single_models[event_name] = 'Not enough data'
            return None
        
        X = event_df.drop(columns=[self.target_col]).values
        y = event_df[self.target_col].values
        model = LOOModelEvaluator(lr_only)
        model.fit(X, y)
        model_scores = model.score()
        self.single_models[event_name] = model
        return None
    
    
    def single_event_predict(self, event_name, event_df=None):
        if self.single_models[event_name] == 'Not enough data' or self.single_models[event_name] == 'Failed to fit':
            return 'fail', False
        else: 
            scores = self.single_models[event_name].scores
            max_accuracy = max(model["win_rate"] for model in scores.values())
            candidates = {name: model for name, model in scores.items() if model["win_rate"] == max_accuracy}
            best = max(candidates.items(), key=lambda x: x[1]["corr"]) 
            return self.single_models[event_name].lr_model.predict(event_df.values), True
        
    def predict(self, event_df_dict=None):
        if event_df_dict is None:
            print('No event data provided')
        seperate_pred = {}
        available_events = {}
        predict_event = {}
        save_true_y = None
        for event_name, event_df in event_df_dict.items():
            # print(event_name)
            self.single_event_train(event_name)
            pred, attend = self.single_event_predict(event_name, event_df)
            seperate_pred[event_name] = pred
            if attend:
                # print('true')
                available_events[event_name] = self.event_dict[event_name]  
                
                predict_event[event_name] = event_df
                event_df = pd.merge_asof(event_df, self.ret_df[[self.target_col]], left_index=True, right_index=True,
                            direction='backward', tolerance=pd.Timedelta('1m')).dropna(axis=0).dropna(how='all')
                save_true_y = event_df[self.target_col]
                
        event_name = 'combined'
        if len(available_events) < 2:
            seperate_pred[event_name] = np.nan
        else:   
            #重命名列，用key作为列名前缀
            all_event_df = pd.concat(available_events.values(), keys=available_events.keys(), axis=1)
            all_event_df.columns = [f"{key}_{col}" for key, col in all_event_df.columns]
            event_name = 'combined'
            self.single_event_train(event_name, all_event_df, True)
            if self.single_models[event_name] == 'Not enough data':
                seperate_pred[event_name] = np.nan
            
            pred_event_df = pd.concat(predict_event.values(), keys=predict_event.keys(), axis=1)
            pred, attend = self.single_event_predict(event_name, pred_event_df) 
            if pred =='fail':
                seperate_pred[event_name] = np.nan
            else:
                seperate_pred[event_name] = pred
        
        seperate_pred['save_true_y'] = save_true_y
        return seperate_pred
 
class LOOModelEvaluator:
    def __init__(self, lr_only=False):
        """
        kde_regressor: 一个符合 sklearn API 的 KDE 回归器，需要实现 fit 和 predict 方法。
        """ 
        self.kde_model = None
        self.lr_model = None
        self.scores = {}
        self.lr_only = lr_only
    
    def fit(self, X, y):
        """
        使用 Leave-One-Out 交叉验证评估模型，同时存储最终的全数据训练模型。
        """
        loo = LeaveOneOut()
        y_list = []
        predicts = {'KDE': [], 'LR': []} if not self.lr_only else {'LR': []}
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_list.append(y_test[0])
            
            # 训练单次交叉验证模型
            kde = KDERegressor()
            kde.fit(X_train, y_train)
            lr = LinearRegression().fit(X_train, y_train)
            
            # 计算预测值
            predicts['KDE'].append(kde.predict(X_test)[0]) if not self.lr_only else None
            predicts['LR'].append(lr.predict(X_test)[0])
        
        # 计算评估分数
        final_scores = {}
        for model in ['KDE', 'LR'] if not self.lr_only else ['LR']:
            predicts[model] = np.array(predicts[model]).flatten()
            temp_df = pd.DataFrame({'y': y_list, 'predict': predicts[model]})
            final_scores[model] = {
                'win_rate': np.mean((temp_df['y'] * temp_df['predict']) > 0),
                'corr': temp_df['y'].corr(temp_df['predict']),
                'spearman': temp_df['y'].corr(temp_df['predict'], method='spearman'),
                '(y-yhat)/y': np.mean(abs(temp_df['y'] - temp_df['predict']) / abs(temp_df['y']))
            }
        
        self.scores = final_scores
        
        # 训练最终模型
        self.kde_model = KDERegressor() if not self.lr_only else None
        self.kde_model.fit(X, y) if not self.lr_only else None
        self.lr_model = LinearRegression().fit(X, y)
    
    def score(self):
        """
        返回 Leave-One-Out 交叉验证的评分结果。
        """
        return self.scores