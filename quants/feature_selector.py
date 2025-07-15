from common.quants.stats import analyze_evaluation_results
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from common.quants.time_series_cv import run_tscv, obtain_cv_coef
from common.miscs.html import pph

from common.quants.time_series_cv import parse_tscv_res

model_params_grid = {'loss':'epsilon_insensitive',
    'penalty': 'elasticnet',
   'alpha': [0.05, 0.01, 0.005, 0.001],
    'l1_ratio': [0.05,  0.01, 0.005, 0.001],
    'max_iter': 1000,  
    'tol': 1e-3,      
    'random_state': 0,
    'fit_intercept': False
}
class FeatureSelector:
    def __init__(self, **kwargs):
        
        self.df_in = kwargs.pop('df_in')
        self.model = kwargs.pop('model', SGDRegressor)
        self.param_dict = kwargs.pop('param_dict', model_params_grid)
        self.target_col = kwargs.pop('target_col', 'y')
        self.test_ratio_each = kwargs.pop('test_ratio_each', 2/52)
        self.num_folds = kwargs.pop('num_folds', 10)
        self.n_process = kwargs.pop('n_process', 1)

        self.judgement = kwargs.pop('judgement', 'mape')
        self.max_iter = kwargs.pop('max_iter', 1)
        self.p_value_criteria = kwargs.pop('p_value_criteria', 0.01)
        self.p_criteria_decay = kwargs.pop('p_criteria_decay', 0.2)
        self.run_tscv_kwargs = kwargs

        self.start_features = [col for col in self.df_in.columns.tolist() if col not in [self.target_col]]
        self.features = {0: self.start_features}
        self.results  = {0: None}
        self.scores   = {0: None}
        self.best_params = {0: None}

    def get_best_x_features(self, verbose=False):
        if self.results[0] == None:
            self.run(verbose)
        best_score_in_different_round \
            = [np.max(self.scores[i]['median']) for i in range(min(len(self.scores), self.max_iter))]
        best_round = np.argmax(best_score_in_different_round)
        print(f'Best Round Selected {best_round}')
        return self.features[best_round]
    
    def obtain_grid_result(self, i, verbose=False):
        df = self.df_in[self.features[i]+[self.target_col]].copy()
        if verbose:
            print('start the {}th round'.format(i))
        all_result = run_tscv(self.model, self.param_dict, df, self.target_col, num_parallel=self.n_process,  test_ratio_each=self.test_ratio_each, num_folds=self.num_folds, print_result=False, save_y_pred=False, **self.run_tscv_kwargs, show_progress=True)
        self.results[i] = all_result
    
    def get_score(self, i, judgement=None, verbose=False):
        result = self.results[i]
        _ = parse_tscv_res(result, verbose=verbose)

        df_metric_test = result['metrics_test']
        index_columns = [col for col in df_metric_test.index.names if col != 'fold']
        # print('index_columns:', index_columns)
        # pph(df_metric_test.groupby(index_columns)[['r2','mape','corr','rank','win']].describe().T, f'{i}th round', show_result=verbose)

        judgement = judgement or self.judgement # when judgement is None, use self.judgement
        grouped = df_metric_test.groupby(index_columns)[judgement].median() # calc mape median
        param_list = [dict(zip(index_columns, values)) for values in grouped.index]
        median_list = grouped.values.tolist()
        self.scores[i] =  {'param': param_list, 'median': median_list}
    
    def get_best_features_for_iter(self, i):
        score = self.scores[i]
        best_index = np.argmax(score['median'])
        best_param = score['param'][best_index]
        self.best_params[i] = best_param
        temp_res = analyze_evaluation_results(obtain_cv_coef(self.results[i], self.features[i], best_param))
        current_p_criteria = self.p_value_criteria*(self.p_criteria_decay**i)
        cols=temp_res[temp_res['p_value']<current_p_criteria].sort_values('p_value', ascending=False ).index.tolist()
        self.features[i+1] = cols
        
    def run(self, verbose):
        for i in range(self.max_iter):
            if len(self.features[i]) <= 1:
                break

            self.obtain_grid_result(i)
            self.get_score(i, verbose=verbose)
            self.get_best_features_for_iter(i)

            if verbose:
                print('The {}th round is finished'.format(i))

        return self.features, self.results, self.scores