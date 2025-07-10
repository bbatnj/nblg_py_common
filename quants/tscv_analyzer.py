import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from . import model_analyzer
import warnings


class TSCV_evaluator:
    def __init__(self, **kwargs):
        self.model_class = kwargs.get('model_class', None)
        self.param_dict = kwargs.get('param_dict', None)
        self.df_in = kwargs.get('df_in', None)
        self.target_column = kwargs.get('target_column', None)
        self.num_parallel = kwargs.get('num_parallel', 1)
        self.test_ratio_over_all_df = kwargs.get('test_ratio_each', 0.05)
        self.num_folds = kwargs.get('num_folds', 5)
        self.print_result = kwargs.get('print_result', True)
        self.always_fixed_params = kwargs.get('always_fixed_params', None)
        self.win_threshold = kwargs.get('win_threshold', 0.5)
        self.trade_threshold = kwargs.get('trade_threshold', 0.5)
        self.save_y_pred = kwargs.get('save_y_pred', [])
        self.save_model_analyzer = kwargs.get('save_model_analyzer', False)
        self.type = kwargs.get('type', 'regression')
        self.zero_threshold = kwargs.get('zero_threshold', None)
        self.show_progress = kwargs.get('show_progress', True)
        self.divide_base_col = kwargs.get('divide_base_col', None)
        
        self.model_dict = {}
        self.model_analyzer_dict = {}
        self.param_comb_dict = {}
        self.metrics_dict = {}
        self.data_dict = {}
        self.index_dict = {}
        self.evaluate_result = None
    
    ##### TODO: obtain other params to calc test_ratio_over_all_df and num_folds
    def __get_data_splits(self):
        num_rows = self.df_in.shape[0]
        if self.divide_base_col:
            self.df_in = self.df_in.sort_values(self.divide_base_col)
            unique_values = self.df_in[self.divide_base_col].unique()
            num_rows = self.df_in[self.divide_base_col].nunique()
        fold_ratio = 1 - self.test_ratio_over_all_df * (self.num_folds - 1)
        self.test_ratio_per_fold = self.test_ratio_over_all_df / fold_ratio
        start_idx_s = [int(i * self.test_ratio_per_fold * num_rows) for i in range(self.num_folds)]
        end_idx_s = [int((i + 1) * self.test_ratio_per_fold * num_rows) for i in range(self.num_folds)]
        if self.divide_base_col:
            new_start_idx_s = []
            for start_idx in start_idx_s:
                start_value = unique_values[start_idx]
                row_idx = self.df_in.index.get_loc(self.df_in[self.df_in[self.divide_base_col] == start_value].index[0])
                new_start_idx_s.append(row_idx)

            start_idx_s = new_start_idx_s

            new_end_idx_s = []
            for end_idx in end_idx_s:
                end_value = unique_values[end_idx]
                row_idx = self.df_in.index.get_loc(self.df_in[self.df_in[self.divide_base_col] == end_value].index[0])
                new_end_idx_s.append(row_idx)

            end_idx_s = new_end_idx_s
            # self.df_in = self.df_in.drop(columns=self.divide_base_col)
        return start_idx_s, end_idx_s
    
    def __get_model_params(self):
        
        param_lists = {param: values for param, values in self.param_dict.items() if isinstance(values, list) and param not in self.always_fixed_params}
        fixed_params = {param: values for param, values in self.param_dict.items() if not isinstance(values, list) or param in self.always_fixed_params}

        param_comb_s = [dict(zip(param_lists.keys(), values)) for values in itertools.product(*param_lists.values())]
        
        args_list = []
        start_idx_s, end_idx_s = self.__get_data_splits()
        
        for comb_idx, param_comb in enumerate(param_comb_s):
            model_params = {**fixed_params, **param_comb}
            self.model_dict[f'comb_{comb_idx}'] = {}
            self.model_analyzer_dict[f'comb_{comb_idx}'] = {}
            self.index_dict[f'comb_{comb_idx}'] = {}
            self.param_dict[f'comb_{comb_idx}'] = {}
            self.param_comb_dict[f'comb_{comb_idx}'] = param_comb
            for fold, (start_index, end_index) in enumerate(zip(start_idx_s, end_idx_s)):
                self.model_dict[f'comb_{comb_idx}'][f'fold_{fold}'] = self.model_class(**model_params)
                self.index_dict[f'comb_{comb_idx}'][f'fold_{fold}'] = (start_index, end_index)
                self.param_dict[f'comb_{comb_idx}'][f'fold_{fold}'] = param_comb
                args_list.append(
                    (
                        comb_idx,
                        fold,
                    )
                )
        self.args_list = args_list
        return args_list
    
    def __get_model_params_multiprocess(self):
        args_list = self.__get_model_params()
        if self.num_parallel > 1:
            with ProcessPoolExecutor(self.num_parallel) as executor:
                futures = [executor.submit(self._train, *args) for args in args_list]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing") if self.show_progress else futures:
                    pass
            results = [future.result() for future in futures]
        else:
            results = [self._train(*args) for args in tqdm(args_list, desc="Processing")] if self.show_progress else [self._train(*args) for args in args_list]
        return results
    
    def _train(self, comb_idx, fold):
        #    start_index, end_index, model, param_comb, fold):
        start_index, end_index = self.index_dict[f'comb_{comb_idx}'][f'fold_{fold}']
        model = self.model_dict[f'comb_{comb_idx}'][f'fold_{fold}']
        param_comb = self.param_dict[f'comb_{comb_idx}'][f'fold_{fold}']
        training = model_analyzer.ModelAnalyzer(self.df_in.iloc[start_index:end_index], self.target_column, model, self.test_ratio_per_fold, divide_base_col = self.divide_base_col)
        training.train()
        if self.save_model_analyzer:
            self.model_analyzer_dict[f'comb_{comb_idx}'][f'fold_{fold}'] = training
        y_test = training.y_test
        y_train = training.y_train
        y_pred_test = training.test_predictions
        y_pred_train = training.train_predictions
            
        
        metrics_train = self.obtain_metrics(y_train, y_pred_train, fold, param_comb )
        metrics_test = self.obtain_metrics(y_test, y_pred_test, fold, param_comb )
        
        if self.print_result:
            print(f"Fold {fold}: {param_comb}")
            print(f"Test: {metrics_test}")
        
        result = {
            "metrics_train": metrics_train,
            "metrics_test": metrics_test,
            "model": training.model,
            "scaler": training.scaler,
            "fold": fold,
            "params": param_comb
        }
        
        if self.save_y_pred:
            result["y_test"] = pd.DataFrame({ 'y_pred_test': y_pred_test, 'y_test': y_test}, index=y_test.index)
            result["y_train"] = pd.DataFrame({ 'y_pred_train': y_pred_train, 'y_train': y_train}, index=y_train.index)
        
        return result  
    
    def obtain_metrics(self, y_true, y_pred, fold, param_comb):
        warnings.filterwarnings('ignore')
        df_temp = pd.DataFrame({ 'y_pred': y_pred, 'y_true': y_true}, index=y_true.index)
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mape': 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true)),
            'corr': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 1,
            'rank': spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 1,
            'se': mean_absolute_error(y_true, y_pred) * len(y_true),
            'win': np.mean((y_true * y_pred > self.win_threshold)),
            'pred_larger': np.mean((y_pred > y_true)),
        }        
        }        
        df_temp = pd.DataFrame({ 'y_pred': y_pred, 'y_true': y_true}, index=y_true.index)
        }
        df_temp = pd.DataFrame({ 'y_pred': y_pred, 'y_true': y_true}, index=y_true.index)
        # successful prediction
        df_success = df_temp[(df_temp['y_pred'] > self.trade_threshold) ]
        if len(df_success) > 0:
            success_metrics = {
                'success_r2': r2_score(df_success['y_true'], df_success['y_pred']),
                'success_mape': 1 - mean_absolute_error(df_success['y_true'], df_success['y_pred']) / np.mean(np.abs(df_success['y_true'])),
                'success_corr': pearsonr(df_success['y_true'], df_success['y_pred'])[0] if len(df_success) > 1 else 1,
                'success_rank': spearmanr(df_success['y_true'], df_success['y_pred'])[0] if len(df_success) > 1 else 1,
                'success_win': np.mean((df_success['y_true'] * df_success['y_pred'] > self.win_threshold)),
                'return': sum(df_success['y_true'])
            }
        else:
            success_metrics = {**metrics, 'success_r2': np.nan, 'success_mape': np.nan, 'success_corr': np.nan, 'success_rank': np.nan, 'success_win': np.nan, 'return': np.nan}
        metrics = {**metrics, **success_metrics}
        warnings.resetwarnings()

        param_keys = tuple(param_comb.keys())

        def list2tuple(params):
            return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}

        param_values = tuple(list2tuple(param_comb).values())
        index = pd.MultiIndex.from_tuples([(fold, *param_values)], names=['fold'] + list(param_keys))
        
        all_metrics = {**metrics}

        return pd.DataFrame([all_metrics], index=index)
    
    def evaluate(self):
        if self.evaluate_result is not None:
            return self.evaluate_result
        results = self.__get_model_params_multiprocess()
        tr_metric_s = []
        tst_metric_s = []
        models = []
        scalers = []
        folds = []
        params_list = []
        if self.save_y_pred:
            y_test = []
            y_train = []
            
        for result in results:
            tr_metric_s.append(result["metrics_train"])
            tst_metric_s.append(result["metrics_test"])
            models.append(result["model"])
            scalers.append(result["scaler"])
            folds.append(result["fold"])
            params_list.append(result["params"])
            if self.save_y_pred:
                y_test.append(result["y_test"])
                y_train.append(result['y_train'])
        tr_mtr_df = pd.concat(tr_metric_s)
        tst_mtr_df = pd.concat(tst_metric_s)

        self.evaluate_result = {
            "metrics_train": tr_mtr_df,
            "metrics_test": tst_mtr_df,
            "models": models,
            "scalers": scalers,
            "folds": folds,
            "params": params_list
        }

        if self.save_y_pred:
            self.evaluate_result["y_test"] = y_test
            self.evaluate_result["y_train"] = y_train

        return self.evaluate_result


####### example usage
# tscv_dict = {
#     'model_class': CustomRegression,
#     'param_dict': {},   
#     'df_in': used_df,
#     'target_column': target,
#     'num_parallel': 15,
#     'test_ratio_each': 7/240,
#     'num_folds': 15,
#     'print_result': True,
#     'save_y_pred': True
# }
# tscv = TSCV_evaluator(**tscv_dict)

# tscv.evaluate()