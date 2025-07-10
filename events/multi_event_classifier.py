from common.events.kde_classifier import WeightedEnsembleClassifier
import numpy as np
import pandas as pd


class MultiEventClassifier:
    def __init__(self, df_dict, ret_df, **kwargs):
        self.event_dict = df_dict
        self.ret_df = ret_df
        self.target_col = kwargs.get('target_col', 'y')
        # self.ret_df['y'] = self.ret_df[self.target_col]
        self.predict_point = kwargs.get('predict_point', None)
        self.single_models = {}
        
        
    def single_event_train(self, event_name, event_df=None, lr_only = False):
        if event_df is None:
            event_df = self.event_dict[event_name]
        event_df = pd.merge_asof(event_df, self.ret_df[[self.target_col]], left_index=True, right_index=True,
                              direction='backward', tolerance=pd.Timedelta('1m')).dropna(axis=0).dropna(how='all')
        if event_df.shape[0] < 5:
            print(f"Event {event_name} has less than 5 data points, skip")
            self.single_models[event_name] = 'Not enough data'
            return None
        #if event_df[self.target_col]里面1的数量小于2,或者0的数量小于2,那么就不进行预测
        if event_df[self.target_col].sum() < 2:
            print(f"Event {event_name} has less than 2 positive data points, skip")
            self.single_models[event_name] = 'Pred all as negative'
            return None
        elif event_df[self.target_col].sum() > event_df.shape[0] - 2:
            print(f"Event {event_name} has less than 2 negative data points, skip")
            self.single_models[event_name] = 'Pred all as positive'
            return None
        
        Weighted_EnsembleClassifier = WeightedEnsembleClassifier(lr_only = lr_only)
        Weighted_EnsembleClassifier.fit(event_df.drop(columns=[self.target_col]), event_df[self.target_col])
        if Weighted_EnsembleClassifier.not_useful:
            print(f"Event {event_name} failed to fit")
            self.single_models[event_name] = 'Failed to fit'
            return None
        print(Weighted_EnsembleClassifier.loo_scores)
        self.single_models[event_name] = Weighted_EnsembleClassifier
        # 计算预测概率
        pred_proba = Weighted_EnsembleClassifier.predict_proba(event_df.drop(columns=[self.target_col]))[:, 1]
        
        # 计算 log-odds
        odds = np.log(np.clip(pred_proba / (1 - pred_proba), 1e-6, 1e6))  # 避免 0 和 1 的溢出问题
        
        # 返回 log-odds DataFrame
        return pd.DataFrame({'odds': odds}, index=event_df.index)
    
    
    def single_event_predict(self, event_name, event_df=None):
        if self.single_models[event_name] == 'Not enough data' or self.single_models[event_name] == 'Failed to fit':
            return 'fail', False
        elif self.single_models[event_name] == 'Pred all as negative': 
            return np.zeros(event_df.shape[0]), False
        elif self.single_models[event_name] == 'Pred all as positive': 
            return np.ones(event_df.shape[0]), False
        else: 
            scores = self.single_models[event_name].loo_scores
            max_accuracy = max(model["accuracy"] for model in scores.values())
            candidates = {name: model for name, model in scores.items() if model["accuracy"] == max_accuracy}
            best = min(candidates.items(), key=lambda x: x[1]["log_loss"]) 
            return self.single_models[event_name].predict_proba(event_df, best[0])[:, 1], True
        
    def predict(self, event_df_dict=None):
        if event_df_dict is None:
            print('No event data provided')
        seperate_prob_predict = {}
        available_events = {}
        available_odds = {}
        predict_event = {}
        predict_odds = {}
        save_true_y = None
        for event_name, event_df in event_df_dict.items():
            print(event_name)
            odds = self.single_event_train(event_name)
            prob, attend = self.single_event_predict(event_name, event_df)
            seperate_prob_predict[event_name] = prob
            if attend:
                print('true')
                available_events[event_name] = self.event_dict[event_name]  
                available_odds[event_name] = odds
                predict_odds[event_name] = pd.DataFrame({'odds': np.log(np.clip(prob / (1 - prob), 1e-6, 1e6))}, index=event_df.index)
                
                predict_event[event_name] = event_df
                event_df = pd.merge_asof(event_df, self.ret_df[[self.target_col]], left_index=True, right_index=True,
                            direction='backward', tolerance=pd.Timedelta('1m')).dropna(axis=0).dropna(how='all')
                save_true_y = event_df[self.target_col]
        try:
            #####接下来考虑odds里面的数据
            combined_event_name = '_'.join(available_odds.keys())+'odds'
            predict_odds_df = pd.concat(predict_odds, axis=1).dropna(axis=0)
            concat_event_df = pd.concat(available_odds , axis=1).dropna(axis=0)
            predict_event_df = pd.concat(predict_event, axis=1).dropna(axis=0)
            
            concat_event_df.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in concat_event_df.columns] 
            self.single_event_train(combined_event_name, concat_event_df, lr_only = True)
            
            pred, attend = self.single_event_predict(combined_event_name, predict_odds_df)
            seperate_prob_predict[combined_event_name] = pred

            #####把predict_event_df里面的值每行逐行平均作为新的预测值
            pred = 1/(1+np.exp(-predict_event_df.mean(axis=1)))
            seperate_prob_predict['mean'] = pred

            #####接下来考虑available_events里面的数据
            combined_event_name = '_'.join(available_events.keys())+'combined'
            concat_event_df = pd.concat(available_events , axis=1).dropna(axis=0)
            predict_event_df = pd.concat(predict_event, axis=1).dropna(axis=0)
            
            concat_event_df.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in concat_event_df.columns] 
            self.single_event_train(combined_event_name, concat_event_df)
            
            pred, attend = self.single_event_predict(combined_event_name, predict_event_df)
            seperate_prob_predict[combined_event_name] = pred
        except:
            pass
        return {**seperate_prob_predict, 'true_y': save_true_y.values} if save_true_y is not None else {**seperate_prob_predict, 'true_y': False}
