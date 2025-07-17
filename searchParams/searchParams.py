import json
import pathlib
import os
import time
from copy import deepcopy
import itertools
from jinja2 import Template
import numpy as np
import pandas as pd
import glob
import cma
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from common.log_parser import analyze_log
from common.run_sim import run_sim, save_dict_to_json, analyze_simulation_metrics
from common.quants.pnl import sharpe, win_ratio
from common.quants.pta import run_pta

def replace_param(libcfg_default_path, combination):
    with open(libcfg_default_path, 'r') as file:
        json_str = file.read()
    for key, value in combination.items():
        placeholder = f"@{key}@"
        json_str = json_str.replace(placeholder, str(value))
    return json.loads(json_str)

def generate_config_path_df(combination, output_config_path, save_path, name):

    os.system(f'mkdir -p {save_path}')

    def list2tuple(params):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    param_keys = tuple(combination.keys())
    param_values = tuple(list2tuple(combination).values())
    index = pd.MultiIndex.from_tuples([param_values], names=list(param_keys))
    results_df = pd.DataFrame([output_config_path], index=index, columns=["outputConfigPath"])
    
    result_name = os.path.join(save_path, f'{name}.csv')

    if os.path.exists(result_name):
        results_df.to_csv(result_name, mode='a', header=False)
    else:
        results_df.to_csv(result_name)

    return results_df


def _run_sim_with_param(i, combination, config_template_path, grid_temp_save, save_config_path, title, prefix='grid_search', return_config=False):
    time.sleep(i)  # Pause different moment to ensure all logs are written
    with open(config_template_path) as f:
        config = json.load(f)
        libcfg_default_path = config['nblgJsonPath']

    update_dict = replace_param(libcfg_default_path, combination)
    SimLibName = f"SimLibCfg+{'+'.join([f'{k}_{v}' for k, v in combination.items()])}.json"
    SimLibCfg_path = save_dict_to_json(update_dict, grid_temp_save, SimLibName)

    config['nblgJsonPath'] = SimLibCfg_path
    SimCfgName = f"SimCfg+{'+'.join([f'{k}_{v}' for k, v in combination.items()])}.json"
    SimCfg_file_path = save_dict_to_json(config, grid_temp_save, SimCfgName)

    new_title = f"{title + '/' if title else ''}{'/'.join([f'{k}_{v}' for k, v in combination.items()])}"

    result = run_sim(SimCfg_file_path, new_title, config_save_path=save_config_path, name=SimCfgName)
    output_config_path = result['output_config_path']
    if return_config:
        return result['config']
    return generate_config_path_df(combination, output_config_path, save_config_path, f'{prefix}_{i}')

def run_grid_search(grid_dict, config_template_fn,  title=None, params_combine_rule='Cartesian', num_workers=10):

    def get_grid_combinations(grid_params, method='Cartesian'):

        if method == 'Cartesian':
            keys, values = zip(*grid_params.items())
            return [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        else:
            lengths = [len(v) for v in grid_params.values() if len(v) > 1]

            if len(lengths) > 0 and len(set(lengths)) != 1:
                raise ValueError("All vectors with length greater than 1 must have the same length.")

            max_length = lengths[0] if lengths else 1
            expanded_params = {
                k: v if len(v) > 1 else v * max_length
                for k, v in grid_params.items()
            }
            keys, values = zip(*expanded_params.items())
            return [dict(zip(keys, combination)) for combination in zip(*values)]

    combinations = get_grid_combinations(grid_dict, method=params_combine_rule)
    print('parameter combinations:', combinations)

    working_dir = '/'.join(config_template_fn.split('/')[0:-1])
    ###### Alternatively, we use working_dir+title as the save path
    working_dir = os.path.join(working_dir, title)
    
    #if output dir already exist, we wont run it, simply print an cmd on how to del the out put and exit
    if os.path.exists(os.path.join(working_dir, 'grid_save_config_path')):
        print(f"output dir already exist, please run the following cmd to delete the output dir and rerun this script")
        print(f"rm -rf {os.path.join(working_dir, 'grid_save_config_path')}")
        print(f"rm -rf {os.path.join(working_dir, 'grid_temp_save')}")
        return
    grid_temp_save = os.path.join(working_dir, 'grid_temp_save')

    save_config_path = os.path.join(working_dir, 'grid_save_config_path')

    pool_args = [(i, combination, config_template_fn, grid_temp_save, save_config_path, title)
                 for i, combination in enumerate(combinations) ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results_df_list = list(executor.map(_run_sim_with_param, *zip(*pool_args)))
    else:
        results_df_list = [_run_sim_with_param(*args) for args in pool_args]

    result = pd.concat(results_df_list)
    result_name = os.path.join(save_config_path, 'grid_result.csv')
    result.to_csv(result_name)
    
    return result

def calc_trade_stats(metric_df, dict_df_detail, instr_name=None):
    if instr_name is None:
        metric_sum_df = metric_df[['gross_book_size', 'net_book_size']].sum().to_frame().T   #'total_pnl', 'notional', 
    else:
        metric_sum_df = metric_df.query('instr == @instr_name').copy()
    
    new_df = metric_sum_df
    new_df['pnl_daily_avg'] = metric_for_search({'dict_df_detail': dict_df_detail}, instr_name=instr_name, pnl_type='total_pnl', metric_type='daily_mean_pnl', principal=1000)
    new_df['ntl_vlm_daily_avg'] = metric_for_search({'dict_df_detail': dict_df_detail}, instr_name=instr_name, pnl_type='total_pnl', metric_type='daily_mean_notional_vol', principal=1000)
    new_df['win_ratio'] = metric_for_search({'dict_df_detail': dict_df_detail}, instr_name=instr_name, pnl_type='total_pnl', metric_type='win_ratio', principal=1000)
    new_df['sharpe'] = metric_for_search({'dict_df_detail': dict_df_detail}, instr_name=instr_name, pnl_type='total_pnl', metric_type='sharpe', principal=1000)
    
    return new_df

def process_row(i, grid_search_path_df, fee_rate, pnl_hzs, instr_name):
    config_path = grid_search_path_df.iloc[i][-1]
    with open(config_path) as f:
        config = json.load(f)
    # print(config)
    stat, metric_df, dict_df_detail, parser = analyze_log(config, fee_rate, pnl_hzs, False)
    return i, stat, metric_df, dict_df_detail, parser

def parse_grid_search(grid_search_path_df, fee_rate, pnl_hzs, instr_name=None):
    df_template = grid_search_path_df.copy()
    df_template = df_template.drop('outputConfigPath', axis=1)
    order_stats = df_template.copy()
    trade_stats = df_template.copy()

    parsers = []

    with ProcessPoolExecutor(16) as executor:
        futures = {executor.submit(process_row, i, grid_search_path_df, fee_rate, pnl_hzs, instr_name): i for i in range(grid_search_path_df.shape[0])}
        
        for future in as_completed(futures):
            i, stats, metric_df, dict_df_detail, parser = future.result()
            parsers.append(parser)
            # order_stats 
            temp_df = pd.DataFrame([stats])
            for col in temp_df.columns:
                order_stats.at[i, col] = temp_df[col].values[0]

            # trade_stats 
            temp_df = calc_trade_stats(metric_df, dict_df_detail, instr_name)    
            for col in temp_df.columns:
                trade_stats.at[i, col] = temp_df[col].values[0]

    print('order_stats:', order_stats)
    print('trade_stats:', trade_stats)
    return order_stats, trade_stats, parsers

def metric_for_search(result, instr_name=None, pnl_type='total_pnl', metric_type='sharpe', principal=1000):
    df_dicts = result.get('dict_df_detail', {})
    def aggregate_data(df_dicts, key_filter, column_name):
        df = pd.DataFrame()
        for str, value in df_dicts.items():
            for key, key_value in value.items():
                if key_filter not in key:
                    continue
                if df.empty:
                    df = key_value[[column_name]].copy()
                else:
                    df[column_name] += key_value[column_name]
        return df
    if instr_name is None:
        pnl = aggregate_data(df_dicts, 'pnl', pnl_type)
    else:
        pnl = df_dicts[instr_name]['pnl'].copy()
        if pnl.empty:
            raise ValueError(f"No data found for instrument: {instr_name}")
    if pnl.empty:
        return 0
    pnl['date'] = pnl.index.date
    
    if metric_type == 'daily_mean_notional_vol':
        if instr_name is None:
            notion_vol_dict = {}
            for key, value in df_dicts.items():
                notion_vol = value["fills"][['notional','date']].copy()
                notion_vol['date'] = notion_vol.index.date
                notion_vol_dict[key] = notion_vol
            if not notion_vol_dict:
                raise ValueError(f"No data found for notion_vol")
            notion_val = []
            notion_val.append(notion_vol_dict[key].groupby('date')['notional'].sum().mean()) 
            return pd.Series(notion_val).mean()
        else:
            notion_vol = df_dicts[instr_name]["fills"].copy()
            if notion_vol.empty:
                raise ValueError(f"No data found for instrument: {instr_name}")
        if notion_vol.empty:
            return 0
        notion_vol['date'] = notion_vol.index.date
        return notion_vol.groupby('date')['notional'].sum().mean()

    result_df = pnl.groupby('date')[pnl_type].last() - pnl.groupby('date')[pnl_type].first()

    if result_df.empty:
        return 0
    metric_functions = {
        'daily_mean_pnl': result_df.mean,
        'sharpe': lambda: sharpe(result_df),
        'win_ratio': lambda: win_ratio(result_df),
    }
    if metric_type in metric_functions:
        return metric_functions[metric_type]()
    raise ValueError(f"Unsupported metric_type: {metric_type}")

def func_for_cma_search(i, cma_vector, cma_dict_template, config_template_path, title, fee_rate= -0.5e-4, capital = 1e3, cfg_temp_path=None, save_config_path=None):
    print(f"this CMA input is {cma_vector}")
    with open(config_template_path) as f:
        config = json.load(f)
    libcfg_default_path = config['libConfigPath']
    working_dir = '/'.join(config_template_path.split('/')[0:-1])
    save_config_path = os.path.join(working_dir, 'cma_save_config_path')   
    template = Template(cma_dict_template)
    updated_dict_content = template.render(cma_vector=cma_vector)
    combination = json.loads(updated_dict_content)
    cma_temp_save = os.path.join(working_dir, 'cma_temp_save')
    title = 'test'
    print("Running CMA optimization")
    config = _run_sim_with_param(i, combination, config_template_path, cma_temp_save, save_config_path, title, prefix='cma', return_config = True) 
    log_dir = config['logDir']
    # sdate = '2024-11-01'
    # edate = '2024-11-30'
    # fns = glob.glob('/nas-1/ShareFolder/bb/grid_search_output/okx_fwd1m_me0x54_with_Events_FIX_WITH_BNF_FEATURES_ONLY_HIGHER_BTCUSDC_MINEDGE_B_II/DRC_60/posAdjMul_0.5/sprd2EgMul_0.125/BTCUSDC_MinEdgeB_1.05_2025-01-31_17-18-52/*.txt.gz')

    sdate = config['startDate']
    edate = config['endDate']
    fns = glob.glob(f'{log_dir}/*.txt.gz')
    z = run_pta(fns, sdate, edate, fee_rate=fee_rate, capital=capital,show_result=False, show_daily_result = False)
    value = z['PnL_Summary'].loc["Total", ("daily_pnl", "sharpe")]
    print(f"this CMA output is {value}")
    return -value #becase we want to maximize the sharpe ratio, so we need to minimize the negative sharpe ratio

def parallel_fun(xs, cma_dict_template, config_template_fn, title, fee_rate , capital ):
    with ProcessPoolExecutor() as executor:
        fitnesses = list(executor.map(func_for_cma_search, range(len(xs)), xs, itertools.repeat(cma_dict_template), itertools.repeat(config_template_fn), itertools.repeat(title), itertools.repeat(fee_rate), itertools.repeat(capital))  )
    return fitnesses

def CMA_search(init_dict, config_template_fn , title='cma', CMA_stds={'sprd2EgMul': 0.1,'BTCUSDC_MinEdgeB': 0.1}, POPSIZE=10, log_dir=None, fee_rate= -0.5e-4, capital = 1e3):
    # 提取向量并构建模板
    cma_vector = []
    template_dict = init_dict.copy()

    CMA_std_array = []
    index = 0  # 用于跟踪cma_vector索引

    for key, value in CMA_stds.items():
        CMA_std_array.append(value)
        cma_vector.append(template_dict[key]) 
        template_dict[key] = f"{{{{cma_vector[{index}]}}}}"  # 变成模板占位符
        index += 1

    # 生成 JSON 格式的模板字符串
    cma_dict_template = json.dumps(template_dict, indent=4)

    if log_dir == None:
        working_dir =  '/'.join(config_template_fn.split('/')[0:-1])
        log_file_path = os.path.join(working_dir, 'log.txt')
    print(f'log file path: {log_file_path}')
    with open(log_file_path, 'w') as log_file:
      sys.stdout = log_file
      sys.stderr = log_file

      options = {'CMA_stds': np.array(CMA_std_array), 
          'popsize': POPSIZE}  # 生成POPSIZE个候选解

      sigma0 = 1  # 初始标准差用于采样新解
      # CMA-ES 实例
      es = cma.CMAEvolutionStrategy(cma_vector, sigma0, options)
      # 优化过程，手动并行化目标函数的计算
      while not es.stop():
          print('start optimization')
          solutions = es.ask()  # 获取候选解
          # 并行计算每个候选解的目标函数值
          fitnesses = parallel_fun(solutions , cma_dict_template, config_template_fn, title, fee_rate= fee_rate, capital = capital)
          es.tell(solutions, fitnesses)  # 将适应度反馈给CMA
          es.disp()
          print('end optimization')
      # 打印最优解，让fitness最小
      xopt = es.result.xbest
      print('优化结果:', xopt)
