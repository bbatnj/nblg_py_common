import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from common.miscs.html import pph
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from . import model_analyzer
from common.miscs.basics import format_df
import warnings

from common.quants.corr import calc_auto_corr

from .stats import analyze_evaluation_results

def create_shared_df(df):    
    original_data = df.to_numpy()
    shm = shared_memory.SharedMemory(create=True, size=original_data.nbytes)
    shared_array = np.ndarray(original_data.shape, dtype=original_data.dtype, buffer=shm.buf)
    shared_array[:] = original_data[:]
    return shm, shared_array, df.columns, df.index 

def get_df_slice(start_idx, end_idx, shared_array, columns, index):
    slice_data = shared_array[start_idx:end_idx].copy()
    slice_index = index[start_idx:end_idx]
    return pd.DataFrame(slice_data, index=slice_index, columns=columns)


def calculate_metrics(y_true, y_pred, fold, params, win_threshold=0.0, auto_corr_lags=[30, 120, 600, 1800], freq='1s', type='regression', zero_threshold=1):
    warnings.filterwarnings("ignore")
    if type == 'regression':
        metrics = {        
            'r2': r2_score(y_true, y_pred),
            'mape': 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true)),
            'corr': pearsonr(y_true, y_pred)[0],
            'rank': spearmanr(y_true, y_pred)[0],
            'win': np.mean((y_true * y_pred > win_threshold)),
            'se': mean_absolute_error(y_true, y_pred) * len(y_true),
            # 'pred_larger': np.mean((y_pred > y_true)),
            # 'Y_true_zero_ratio': np.mean(y_true == 0),
            # 'zero_accuracy': accuracy_score(y_true == 0, y_pred_zero == 0),
            # 'zero_precision': precision_score(y_true == 0, y_pred_zero == 0, zero_division=0),
            # 'zero_recall': recall_score(y_true == 0, y_pred_zero == 0, zero_division=0),
            # 'zero_f1': f1_score(y_true == 0, y_pred_zero == 0, zero_division=0)
        }
        df_temp = pd.DataFrame({ 'y_pred': y_pred, 'y_true': y_true}, index=y_true.index)
        # successful prediction
        df_success = df_temp[(df_temp['y_pred'] > zero_threshold) ]
        if len(df_success) > 0:
            success_metrics = {
                'success_r2': r2_score(df_success['y_true'], df_success['y_pred']),
                'success_mape': 1 - mean_absolute_error(df_success['y_true'], df_success['y_pred']) / np.mean(np.abs(df_success['y_true'])),
                'success_corr': pearsonr(df_success['y_true'], df_success['y_pred'])[0] if len(df_success) > 1 else 1,
                'success_rank': spearmanr(df_success['y_true'], df_success['y_pred'])[0] if len(df_success) > 1 else 1,
                'success_win': np.mean((df_success['y_true'] * df_success['y_pred'] > win_threshold)),
                'return': sum(df_success['y_true'])
            }
        else:
            success_metrics = {**metrics, 'success_r2': np.nan, 'success_mape': np.nan, 'success_corr': np.nan, 'success_rank': np.nan, 'success_win': np.nan, 'return': np.nan}
        metrics = {**metrics, **success_metrics}
        
        
    elif type == 'sign':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'zero_accuracy': accuracy_score(y_true == 0, y_pred == 0),
            'zero_precision': precision_score(y_true == 0, y_pred == 0, zero_division=0),
            'zero_recall': recall_score(y_true == 0, y_pred == 0, zero_division=0),
            'zero_f1': f1_score(y_true == 0, y_pred == 0, zero_division=0)
        }
    
    elif type == 'true-false':
        y_true_binary = (y_true > win_threshold).astype(int)
        y_pred_binary = (y_pred > win_threshold).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }

    elif type == 'classifier':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
    warnings.resetwarnings()

    param_keys = tuple(params.keys())

    def list2tuple(params):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}

    param_values = tuple(list2tuple(params).values())
    index = pd.MultiIndex.from_tuples([(fold, *param_values)], names=['fold'] + list(param_keys))
    
    df_temp = pd.DataFrame({ 'y_pred': y_pred}, index=y_true.index)
    
    # Calculate auto-correlation
    auto_corr_df = calc_auto_corr(df_temp, 'y_pred', auto_corr_lags, freq)
    auto_corr_metrics = auto_corr_df.iloc[0].to_dict() 

    all_metrics = {**metrics, **auto_corr_metrics}

    return pd.DataFrame([all_metrics], index=index)

def train(test_ratio, shared_mem_name, shape, dtype, df_columns, df_index, start_index, end_index,
          target_column, model_input, params=None, data_test=None, print_result=False, fold=0,
          win_threshold=0.0, save_y_pred=False, use_type='regression',zero_threshold=0.1, test_col=None):
    try:
        os.nice(19)  
    except Exception as e:
        print(f"Failed to set low priority: {e}")

    pass
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    df = get_df_slice(start_index, end_index, shared_array, df_columns, df_index)

    df = df.copy()
    shm.close()
    if data_test is not None:
        training = model_analyzer.ModelAnalyzer(df, target_column, model_input, test_ratio, data_test=data_test)
    else:
        training = model_analyzer.ModelAnalyzer(df, target_column, model_input, test_ratio)
    
    training.train()
    y_test = training.y_test
    y_train = training.y_train
    y_pred_test = training.test_predictions
    y_pred_train = training.train_predictions
    
    ### only interested in the df that test_col=1
    if test_col is not None:
        # use df to help find the index that test_col=1
        df_test_col = df[test_col]
        test_col_index = df_test_col[df_test_col == 1].index
        
        
        # y_test index inside test_col_index, they are all time series data
        y_test_index = y_test.index.intersection(test_col_index)
        # turn y_test_index into the number index
        num_y_test_index = [y_test.index.get_loc(i) for i in y_test_index]
        y_test = y_test.loc[y_test_index]
        # but y_pred_test is np.array, so we need new y_test index to filter y_pred_test, it doesn't have the "loc" method
        y_pred_test =  y_pred_test[num_y_test_index]
        
        
        # y_train index inside test_col_index, they are all time series data
        y_train_index = y_train.index.intersection(test_col_index)
        # turn y_train_index into the number index
        num_y_train_index = [y_train.index.get_loc(i) for i in y_train_index]
        y_train = y_train.loc[y_train_index]
        # but y_pred_test is np.array, so we need new y_train index to filter y_pred_test, it doesn't have the "loc" method
        y_pred_train =  y_pred_train[num_y_train_index]
        
        
    if isinstance(y_pred_test, np.ndarray):
        y_pred_test = pd.Series(y_pred_test.reshape(-1), index=y_test.index)
    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = pd.Series(y_pred_train.reshape(-1), index=y_train.index)
    y_pred_test.index = y_test.index
    y_pred_train.index = y_train.index  
    metrics_train = calculate_metrics(y_train, y_pred_train, fold, params, win_threshold=0.0, type=use_type,zero_threshold=zero_threshold)
    metrics_test = calculate_metrics(y_test, y_pred_test, fold, params, win_threshold, type=use_type, zero_threshold=zero_threshold)
    
    if print_result:
        print(f"Fold {fold}: {params}")
        print(f"Train: {metrics_train}")
        print(f"Test: {metrics_test}")
    
    result = {
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "model": training.model,
        "scaler": training.scaler,
        "fold": fold,
        "params": params
    }
    
    if save_y_pred:
        result["y_test"] = pd.DataFrame({ 'y_pred_test': y_pred_test, 'y_test': y_test}, index=y_test.index)
        result["y_train"] = pd.DataFrame({ 'y_pred_train': y_pred_train, 'y_train': y_train}, index=y_train.index)
    
    return result



def run_tscv(model_class, param_dict, df_in, target_column, num_parallel=8,
            test_ratio_each=0.05, num_folds=5, data_test=None, print_result=False, always_fixed_params=[],
             win_threshold=0.0, save_y_pred=False, use_type='regression', zero_threshold=0.1, test_col=None,
             show_progress=True):
    """
    Perform time series cross-validation and return model evaluation results.
    
    Parameters:
    model_class (class): The model class used to instantiate models.
    param_dict (dict): A dictionary containing model parameters. Values can be a single value or a list of candidate values.
    df_in (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    num_parallel (int): Number of parallel processes to run. Default is 8.
    test_ratio_each (float): The number of test data in each fold divided by total data. Default is 0.05.
    num_folds (int): Number of folds for cross-validation. Default is 5.
    data_test (DataFrame): Optional, the test dataset. If provided, testing will be based on this dataset.
    print_result (bool): Whether to print results. Default is False.
    always_fixed_params (list): List of parameter names that are always fixed.

    Returns:
    dict: A dictionary containing training and testing evaluation metrics, models, and scalers.
    """

    print(df_in.shape)
    shm, shared_array, df_columns, df_index = create_shared_df(df_in)
    shared_mem_name = shm.name
    shape = shared_array.shape
    dtype = shared_array.dtype
    param_lists = {param: values for param, values in param_dict.items() if isinstance(values, list) and param not in always_fixed_params}
    fixed_params = {param: values for param, values in param_dict.items() if not isinstance(values, list) or param in always_fixed_params}

    param_comb_s = [dict(zip(param_lists.keys(), values)) for values in itertools.product(*param_lists.values())]
    
    total_length = len(df_in)
    fold_ratio = 1 - test_ratio_each * (num_folds - 1)
    test_ratio_per_fold = test_ratio_each / fold_ratio
    fold_length = int(total_length * fold_ratio)
    step_interval = int(total_length * test_ratio_each)
    start_idx_s = [i * step_interval for i in range(num_folds)]
    end_idx_s = [fold_length + i * step_interval for i in range(num_folds)]
    args_list = []
    results_list = []

    for param_comb in param_comb_s:
        model_params = {**fixed_params, **param_comb}
        for rep, (start_index, end_index) in enumerate(zip(start_idx_s, end_idx_s)):
            test_data = data_test.iloc[start_index:end_index] if data_test is not None else None
            args_list.append(
                (
                    test_ratio_per_fold,
                    shared_mem_name,
                    shape,
                    dtype,
                    df_columns,
                    df_index,
                    start_index,
                    end_index,
                    target_column,
                    model_class(**model_params),
                    param_comb,
                    test_data,
                    print_result,
                    rep,
                    win_threshold,
                    save_y_pred,
                    use_type,
                    zero_threshold,
                    test_col
                )
            )
            results_list.append({
                "fold": rep,
                **param_comb
            })

    if num_parallel > 1:
        with ProcessPoolExecutor(num_parallel) as executor:
            futures = [executor.submit(train, *args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing") if show_progress else futures:
                pass
        results = [future.result() for future in futures]
    else:
        results = [train(*args) for args in tqdm(args_list, desc="Processing")] if show_progress else [train(*args) for args in args_list]

    #print('analyzing')

    tr_metric_s = []
    tst_metric_s = []
    models = []
    scalers = []
    folds = []
    params_list = []
    if save_y_pred:
        y_test = []
        y_train = []
        
    for result in results:
        tr_metric_s.append(result["metrics_train"])
        tst_metric_s.append(result["metrics_test"])
        models.append(result["model"])
        scalers.append(result["scaler"])
        folds.append(result["fold"])
        params_list.append(result["params"])
        if save_y_pred:
            y_test.append(result["y_test"])
            y_train.append(result['y_train'])
    tr_mtr_df = pd.concat(tr_metric_s)
    tst_mtr_df = pd.concat(tst_metric_s)

    output = {
        "metrics_train": tr_mtr_df,
        "metrics_test": tst_mtr_df,
        "models": models,
        "scalers": scalers,
        "folds": folds,
        "params": params_list
    }

    if save_y_pred:
        output["y_test"] = y_test
        output["y_train"] = y_train
        
    shm.close()
    shm.unlink()
    return output

def parse_tscv_res(tscv_result, hyper_params=None, verbose=True):
    aggs = ['std', 'mean', 'median', 'min']
    cols = ['corr', 'r2', 'mape',  'rank', 'win']


    if hyper_params is None:
        df_train = tscv_result['metrics_train'][cols].agg(aggs)
        df_test = tscv_result['metrics_test'][cols].agg(aggs)
    else:
        df_train = tscv_result['metrics_train'].groupby(hyper_params)[cols].agg(aggs)
        df_test = tscv_result['metrics_test'].groupby(hyper_params)[cols].agg(aggs)

    df_train = format_df(df_train)
    df_test = format_df(df_test)

    if verbose:
        from IPython.display import display, HTML
        title_html = '<h3>TSCV Train</h3>'
        display(HTML(title_html))
        display(df_train)

        title_html = '<h3>TSCV Test</h3>'
        display(HTML(title_html))
        display(df_test)

    return df_train, df_test

def obtain_cv_coef(cv_result, col_name_list=None, params=None):
    coef = []
    if not params:
        for result in cv_result['models']:
            coef.append(result.coef_)
    else:
        idx = [i for i, p in enumerate(cv_result['params']) if p == params]
        for i in idx:
            coef.append(cv_result['models'][i].coef_)
    coef_df = pd.DataFrame(coef, index=[f'coef_{i}' for i in range(len(coef))])
    if col_name_list:
        coef_df.columns = col_name_list
    return coef_df

def tscv_feature_selection(tscv_result, x_cols, p_thd, params=None):
    result = analyze_evaluation_results(obtain_cv_coef(tscv_result, x_cols, params=params))
    result = result.sort_values('p_value', ascending=False, key=np.abs)
    pph(result, '')
    return list(result.query('p_value < @p_thd').index)

def obtain_best_params(tscv_result, metric='mape'):
    metric= 'mape'
    metrics_test = tscv_result['metrics_test']
    params = list(tscv_result['params'][0].keys())
    best_params = metrics_test.groupby(params)[metric].median().idxmax()
    best_params_dict = dict(zip(params, best_params))

    conditions = [(metrics_test.index.get_level_values(param) == value) for param, value in zip(params, best_params)]
    conditions = [pd.Series(cond, index=metrics_test.index) for cond in conditions]
    mask = pd.concat(conditions, axis=1).all(axis=1)
    filtered_metrics_test = metrics_test[mask]
    filtered_metrics_test


    metrics_train = tscv_result['metrics_train']
    filtered_metrics_train = metrics_train[mask]
    filtered_metrics_train
 
    # result_dict best_params_dict, filtered_metrics_test.describe(), filtered_metrics_train.describe()
    result_dict = {'best_params': best_params_dict,
                     'metrics_test': filtered_metrics_test.describe(),
                     'metrics_train': filtered_metrics_train.describe()}
    return result_dict

######### use of the function #########
# from common.quants.time_series_cv import obtain_cv_coef
# from common.quants.stats import analyze_evaluation_results
# analyze_evaluation_results(obtain_cv_coef(cv_result, cols))