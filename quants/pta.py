import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from common.log_parser import LogParser

from common.miscs.basics import group_and_concat, merge_by_index
from common.miscs.html import pph
from common.quants.corr import pair_corr, calc_auto_corr, bucket_analyze_generic
from common.quants.pnl import sharpe, win_ratio, calc_trade_markout, n_count
import matplotlib.pyplot as plt

beta_dict = {
    'BTCUSDT.BNF': 1.0,
    'BTCUSDC.BNF': 1.0,
    'ETHUSDT.BNF': 1.4,
    'SOLUSDT.BNF': 1.6,

    'BTCUSDT.OKF': 1.0,
    'BTCUSDC.OKF': 1.0,
    'ETHUSDT.OKF': 1.4,
    'SOLUSDT.OKF': 1.6
}

pnl_hzs = [10, 60, 300, 1800, 7200, 1e9]


def process_directory(root, base_path, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel_inner, suffix):
    relative_path = os.path.relpath(root, base_path)
    files = [f for f in sorted(os.listdir(root)) if f.endswith(suffix)]
    if not files:
        return None, None
    files = [f'{root}/{f}' for f in files if f.endswith(suffix)]
    try:
        df_res = run_pta_multi_session(files, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel=num_parallel_inner, show_result=False)['trade_total']
        return relative_path, df_res
    except Exception as e:
        print(f'Error processing {relative_path} due to: {e}')
        return None, None



def run_pta_group_old(base_path, sdate, edate, fee_rate, capital, suffix='.txt.gz', pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, num_parallel=32):
    results = {}

    for root, _, files in os.walk(base_path):
        # Extract relative path from parent_dir
        relative_path = os.path.relpath(root, base_path)

        # Skip parent_dir itself if empty
        if relative_path == ".":
            continue

        files = [f for f in sorted(files) if f.endswith(suffix)]

        if not files:
            continue

        files = [f'{root}/{f}' for f in files]

        # Process each file in the directory
        try:
            df_res = run_pta_multi_session(files, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel=num_parallel, show_result=False)['trade_total']
            results[relative_path] = df_res
        except Exception as e:
            print(f'Error processing {relative_path} due to : {e}')

    result_df = pd.concat(results, axis=1, names=['stats'])
    return result_df.T




    
def run_pta_group(base_path, sdate, edate, fee_rate, capital, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, suffix='.txt.gz', num_parallel=32, n_process = 4):
    results = {}
    directories = [root for root, _, files in os.walk(base_path) if files]
    num_parallel_inner = max(1, num_parallel // n_process)  # 控制内部并行度

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        future_to_path = {executor.submit(process_directory, root, base_path, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel_inner, suffix): root for root in directories}
        
        for future in as_completed(future_to_path):
            relative_path, df_res = future.result()
            if relative_path and df_res is not None:
                results[relative_path] = df_res
    if results:
        result_df = pd.concat(results, axis=1, names=['stats'])
        return result_df.T
    return pd.DataFrame()


def obtain_pos_corr(df_panel, instr2dfs, beta_dict, pnl_hzs):
    df_panel['fm_bias'] = df_panel.eval('(bfm - sfm) / (bfm + sfm)')

    cols = 'fm_bias rv fv f bfm sfm pos maxLong maxShort'.split()

    df_panel['maxShort'] = -1 * df_panel['maxShort']
    df_panel = df_panel.sort_index()
    df_merged = group_and_concat(df_panel, 'instr', cols)
    # if can't calculate , pass
    df_merged['Total@pos'] = df_merged[[c for c in df_merged if '@pos' in c]].sum(axis=1)
    # beta_dict is the weight of each instrument, it is the weighted sum of the beta of each instrument
    df_merged['Total@beta_pos'] = sum(
        beta_dict[instr] * df_merged[f'{instr}@pos'] 
        for instr in [col.split('@')[0] for col in df_merged if '@pos' in col and 'Total' not in col]
    )
    #Calc Trading Pnls
    trading_pnl_cols = [f'trading_pnl_{hz}s' for hz in pnl_hzs if hz < 1e8] + ['total_pnl']

    for instr in instr2dfs:
        df_pnl = instr2dfs[instr]['pnl']
        df_pnl = df_pnl[trading_pnl_cols]
        df_pnl.columns = [f'{instr}@{c}' for c in df_pnl]
        df_merged = merge_by_index(df_merged, df_pnl)

    for c in trading_pnl_cols:
        df_merged[f'Total@{c}'] = df_merged[[f'{instr}@{c}' for instr in instr2dfs]].sum(axis=1)
    ###TODO: should we use mean pos corr?
    df_pos_corr = calc_auto_corr(df_merged, [c for c in df_merged if 'pos' in c ], [30, 60, 300, 900, 1800, 3600], freq='2s', missing_handle='ffill')
    return df_panel, df_merged,trading_pnl_cols, df_pos_corr


def pph_pta(output):
    show_result = True
    pph(output['Daily_Summary'], 'Daily Summary', show_result=show_result)
    plt = output['daily_return']
    plt.show()
    pph(output['PnL_Summary'], 'PnL Summary', show_result=show_result)
    pph(output['Trade_and_Book_Stats'], 'Trade and Book Stats', show_result=show_result)
    pph(None, 'Pos', show_result=show_result)
    pph(output['Mean_Pos_Corr'], 'Mean Pos-Corr', show_result=show_result)
    pph(output['Order_Stats'], 'Order Stats', show_result=show_result)
    pph(None, 'F', show_result=show_result)
    pph(output['F_Corr'], 'F-Corr', level=4, show_result=show_result)
    pph(output['F_Stats'], 'F-Stats', level=4, show_result=show_result)
    pph(None, 'Markout Analysis', show_result=show_result)
    pph(output['Markout_Analysis'].T, 'Markout Analysis', show_result=show_result)
    pph(output['Markout_by_Kind'], 'Markout by Kind', show_result=show_result)
    pph(output['Gap_Analysis'], 'Gap on fills' , show_result=show_result)
    for instr in output['bu_markout']:
        per_instr_bin_markout = output['bu_markout'][instr]
        n = per_instr_bin_markout[0].shape[0]
        pph(None, f'{instr} BU markout count : {n}', show_result=show_result)
        for i, df_bin_markout in enumerate(per_instr_bin_markout):
            pph(df_bin_markout.T, f'{instr} BU Markout {i}', show_result=show_result)




def run_pta(in_fns, sdate, edate, fee_rate, capital, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, show_result=True, show_daily_result=True):
    output = gen_pta_output(in_fns, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel=1, show_result=show_result, show_daily_result=show_daily_result)
    if show_result:
        pph_pta(output)
    return output


def run_pta_multi_session(in_fns, sdate, edate, fee_rate, capital, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, num_parallel=4, show_result=True, show_daily_result=True):
    output = gen_pta_output(in_fns, sdate, edate, fee_rate, capital, pnl_hzs, account_name, beta_dict, num_parallel=num_parallel, show_result=show_result, show_daily_result=show_daily_result)
    if show_result:
        pph_pta(output)

    # def parallel_update(df_merged):
    #     df_merged = df_merged.copy()
    #     if not isinstance(df_merged.index, pd.DatetimeIndex):
    #         raise ValueError("Index must be a DatetimeIndex.")
    #
    #     grouped = df_merged.groupby(df_merged.index.date)
    #
    #     result = []
    #     prev_last_row = None
    #
    #     for day, group in grouped:
    #         if prev_last_row is not None:
    #             group = group.add(prev_last_row, axis=1)
    #
    #         result.append(group)
    #
    #         prev_last_row = group.iloc[-1]
    #
    #     return pd.concat(result)

    # df_merged = output['df_merged']
    #
    # cols_all = []
    # for instr in output['instr_list']+['Total']:
    #     cols =  [c for c in df_merged if f'{instr}@trading_pnl_' in c] + [f'{instr}@total_pnl']
    #     cols_all += cols
    #
    # df_merged[cols_all]= parallel_update(df_merged[cols_all])
    #plot_pta(output)
    return output

def gen_pta_output(in_fns, sdate, edate, fee_rate, input_capital=1000, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, num_parallel=1, show_result=True, show_daily_result=True):
    from collections import defaultdict as DD
    output = DD(lambda : {})
    final_result_list = parallel_gen_results(in_fns, sdate, edate, fee_rate, pnl_hzs, account_name, beta_dict, num_parallel=num_parallel)
    instr2dfs = {}
    final_result_list = [e for e in final_result_list if e is not None]

    for final_result in final_result_list:
        for instr, kind2df in final_result['instr2dfs'].items():
            if instr not in instr2dfs:
                instr2dfs[instr] = {}
                instr2dfs[instr]['fills'] = []#pd.DataFrame()
                instr2dfs[instr]['pnl'] = []#pd.DataFrame()
                instr2dfs[instr]['auto_corr'] = []#pd.DataFrame()
                
            # instr2dfs[instr]['pnl'] = pd.concat([instr2dfs[instr]['pnl'], dfs['pnl']])
            # ######################### TODO: shall we simply concat these auto_corr? #########################
            # instr2dfs[instr]['auto_corr'] = pd.concat([instr2dfs[instr]['auto_corr'], dfs['auto_corr']])
            # instr2dfs[instr]['fills'] = pd.concat([instr2dfs[instr]['fills'], dfs['fills']])

            instr2dfs[instr]['pnl'].append(kind2df['pnl'])
            ######################### TODO: shall we simply concat these auto_corr? #########################
            instr2dfs[instr]['auto_corr'].append(kind2df['auto_corr'])
            instr2dfs[instr]['fills'].append(kind2df['fills'])

    for instr, kind2df in instr2dfs.items():
        instr2dfs[instr]['pnl'] = pd.concat(kind2df['pnl'], axis=0)
        instr2dfs[instr]['auto_corr'] = pd.concat(kind2df['auto_corr'], axis=0)
        instr2dfs[instr]['fills'] = pd.concat(kind2df['fills'], axis=0)

    instr_list = []
    for final_result in final_result_list:
        for instr in final_result['instr2dfs']:
            instr_list.append(instr)
            instr2dfs[instr]['pnl'] = instr2dfs[instr]['pnl'].sort_index()
            instr2dfs[instr]['auto_corr'] = instr2dfs[instr]['auto_corr'].sort_index()
            instr2dfs[instr]['fills'] = instr2dfs[instr]['fills'].sort_index()
    instr_list = list(set(instr_list))
    output['instr_list'] = instr_list

    trade_metric = pd.concat([final_result['trade_metric'] for final_result in final_result_list])
    
    df_trade_metric = trade_metric.set_index(['instr', 'date'])
    total_by_date = df_trade_metric.groupby('date').sum()
    total_by_date['instr'] = 'Total'
    total_by_date = total_by_date.reset_index().set_index(['instr', 'date'])
    df_with_total = pd.concat([df_trade_metric, total_by_date])
    df_trade_metric = df_with_total.sort_index(level=[0,1], ascending=[True, False])

    book_stats_cols = [c for c in df_trade_metric if any(sub in c for sub in 'gross_book  net_book'.split())]
    metric_cols = 'daily_pnl total_pnl notional'.split() + book_stats_cols

    output['Daily_Summary'] = df_trade_metric[metric_cols]

    ######################### Daily Returns #########################
    if isinstance(input_capital, (int, float)):
        df_temp = df_trade_metric[metric_cols].copy()
        df_temp['daily_return'] = df_temp['total_pnl'] / input_capital * 100
        metric_summary = df_trade_metric.groupby('instr')['total_pnl'].agg([sharpe, win_ratio]).reset_index()
        df_temp = df_temp.reset_index()
        plt.figure(figsize=(12, 6))

        for instr in df_temp['instr'].unique():
            instr_data = df_temp[df_temp['instr'] == instr]
            sharpe_val = metric_summary[metric_summary['instr'] == instr]['sharpe'].values[0]
            win_ratio_val = metric_summary[metric_summary['instr'] == instr]['win_ratio'].values[0]

            plt.plot(instr_data['date'][::-1],
                instr_data['daily_return'][::-1].cumsum(),
                linestyle='-', marker='o', alpha=0.7,
                label=f"{instr} | Sharpe: {sharpe_val:.2f}, Win: {win_ratio_val:.2%}"
            )

        plt.title('Daily Returns by Instrument')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend(title='Instrument')
        target_folder=os.path.basename(os.path.dirname(in_fns[0]))
        plt.figtext(0.02, 0.02, f"target_folder: {target_folder}", fontsize=8, wrap=True, ha="left")
        plt.grid(True)
        output['daily_return'] = plt.gcf()
        if show_result or show_daily_result:
            plt.show()
    #####################################################################
    basic_agg = ['sum', 'mean', 'median', 'std', 'min', 'max']
    extra_agg = [sharpe, win_ratio]+basic_agg

    pnl_part = df_trade_metric[[ 'total_pnl']].groupby('instr').agg(extra_agg)
    trade_part = df_trade_metric[['notional', 'gross_book_mean','net_book_mean']].groupby('instr').agg(basic_agg)
    concat_df = pd.concat([pnl_part, trade_part], axis=1)
    output['trade_total'] = concat_df.loc['Total']

    #'prod_pnl',
    output['PnL_Summary'] = df_trade_metric[['daily_pnl',  'total_pnl']].groupby('instr').agg(extra_agg)
    #'prod_notional',
    #basic_agg = ['mean', 'median', 'std', 'min', 'max']
    basic_agg = ['mean', 'std']
    grp_trade_metric = df_trade_metric.groupby('instr')
    output['Trade_and_Book_Stats'] = grp_trade_metric[['notional'] + book_stats_cols].agg(basic_agg)
    #trade_summary = pd.concat([trade_summary, grp_trade_metric[book_stats_cols].agg(basic_agg)], axis=1)

############################################################################
#Pos
############################################################################        
    df_pos_corr_list = []
    for final_result in final_result_list:
        df_panel_temp = final_result['df_panel']
        if df_panel_temp.empty:
            continue
        instr2dfs_temp = final_result['instr2dfs']
        df_panel_temp, df_merged_temp, trading_pnl_cols_temp, df_pos_corr_temp = obtain_pos_corr(df_panel_temp, instr2dfs_temp, beta_dict, pnl_hzs)
        day = df_panel_temp.index[0].strftime('%Y-%m-%d')
        # pph(df_pos_corr_temp, f'Pos-Corr at {day}', show_result=show_result)
        df_pos_corr_list.append(df_pos_corr_temp)
    
    df_mean_df_pos_corr = pd.concat(df_pos_corr_list).groupby(level=0).mean()
    output['Mean_Pos_Corr'] = df_mean_df_pos_corr
    
    
    df_panel = pd.concat([final_result['df_panel'] for final_result in final_result_list]).sort_index()
    df_panel, df_merged,trading_pnl_cols, df_pos_corr = obtain_pos_corr(df_panel, instr2dfs, beta_dict, pnl_hzs)
    # pph(df_pos_corr.sort_index(), 'Pos-Corr', show_result=show_result)
    
############################################################################
#Latency
############################################################################   
    df_order = pd.concat([final_result['df_order'] for final_result in final_result_list]).sort_index()
    time_cols = ['send_ack_time_logger', 'send_time_lws', 'order_end_time_logger', 'cancel_time_lws']
    for col in time_cols:
        df_order[col] = pd.to_datetime(df_order[col], errors='coerce')
    print('\n[external] lws receive exchange send ack (uds) - lws write send')
    stat = ((df_order['send_ack_time_logger'] - df_order['send_time_lws']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    print(stat)

    print('\n[external] lws receive exchange order end msg (uds) - lws write cancel')
    stat = ((df_order['order_end_time_logger'] - df_order['cancel_time_lws']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    print(stat)



############################################################################
# Order Stats
############################################################################
    order_stat_list = []
    for final_result in final_result_list:
        order_stats = final_result['order_stats']
        for date, instr_stats in order_stats.items():
            for instr, stats in instr_stats.items():
                row = {'date': date, 'instr': instr, **stats}
                order_stat_list.append(row)
    order_stat_df = pd.DataFrame(order_stat_list).set_index(['instr', 'date'])
    # show only some columns
    order_stat_df = order_stat_df[
        ['reject_ratio', 'cancel_ratio', 'fill_ratio', 'late_ratio', 'late_cancel_cnt/trade_cnt', 'mean_lifespan',
         'sum_lifespan']]
    final_result['order_stats'] = order_stats
    output['Order_Stats'] = order_stat_df.sort_index(level=[0, 1], ascending=[True, False])
    if not show_result: ########################### following code is very slow, so we skip it if show_result is False
        return output
############################################################################
#F
############################################################################
    corr_ret_hzs = [10, 60, 300, 1800]
    for instr in instr2dfs:
        for hz in corr_ret_hzs:
            c = f'{instr}@fv_ret_{hz}'
            fv_c = f'{instr}@fv'
            df_merged[c] = df_merged[fv_c].shift(-hz)
            df_merged[c] = (df_merged[c] / df_merged[fv_c]).apply(np.log)
    
    df_y_hat_corr = []
    corr_ret_hzs = [10, 60, 300, 1800]
    corr_order =  ['10s', '60s', '300s', '1800s']
    for instr in instr_list:
        x_cols = [f'{instr}@f']
        y_cols = [c for c in df_merged if f'{instr}@fv_ret_' in c]
        try:
            df_y_hat_corr.append(pair_corr(df_merged, x_cols, y_cols))
        except Exception as e:
            pass

    df_y_hat_corr = pd.concat(df_y_hat_corr, axis=0)['y corr win_ratio beta'.split()]

    df_y_hat_corr['sym'] = df_y_hat_corr['y'].apply(lambda x: x.split('@')[0])
    df_y_hat_corr['ret'] = df_y_hat_corr['y'].apply(lambda x: x.split('@')[1])
    
    df_y_hat_corr = group_and_concat(df_y_hat_corr.drop(['y'], axis=1), 'sym', ['corr', 'beta', 'win_ratio'])
    df_y_hat_corr.index = [f'{c}s' for c in corr_ret_hzs]
    df_y_hat_corr = df_y_hat_corr.T
    df_y_hat_corr.index = pd.MultiIndex.from_tuples(
        [col.split("@") for col in df_y_hat_corr.index], names=["symbol", "metric"])

    df_y_hat_corr = df_y_hat_corr.unstack(level="metric")
    df_y_hat_corr = df_y_hat_corr.reorder_levels(["metric", None], axis=1).sort_index(axis=1)
    sorted_columns = sorted(
    df_y_hat_corr.columns,
        key=lambda x: (x[0], corr_order.index(x[1]))
    )

    # reorder columns
    output['F_Corr'] = df_y_hat_corr[sorted_columns]
    f_cols = [c for c in df_merged if c.endswith('@f')]
    output['F_Stats'] = df_merged[f_cols].agg(['median', 'mean', 'std', 'skew', 'kurt'])
    

    instr2df_markout = {'Total' : []}

    for instr in instr2dfs.keys():
        df_mkt_px = df_merged[[f'{instr}@{c}' for c in 'f fv bfm sfm rv '.split()]].copy()
        df_mkt_px.columns = [c.replace(f'{instr}@', '') for c in df_mkt_px.columns]
        df_mkt_px = df_mkt_px.rename({f'fv': 'mid'}, axis=1)

        markout_hzs = pnl_hzs
        df_fills = instr2dfs[instr]['fills']
        df_markout = calc_trade_markout(df_fills, df_mkt_px, markout_hzs)
        df_markout['instr'] = instr
        df_markout['cancel_time'] = df_markout['cancel_time'].apply(pd.Timestamp)

        instr2df_markout[instr] = df_markout
        instr2df_markout['Total'].append(df_markout)

    instr2df_markout['Total'] = pd.concat(instr2df_markout['Total'], axis=0)

    markout_dict = {}
    df_markout_by_kind = {}
    gap_stats_dict = {}
    for instr in instr2df_markout:
        # pph(None, instr + ' Markout Analysis', level=4, show_result=show_result)
        df_markout = instr2df_markout[instr]

        df_markout = df_markout.replace([np.inf, -np.inf], np.nan)\
            .query('f == f and rv == rv and bfm == bfm and sfm == sfm').copy()


        df_markout['rv_bin'] = pd.qcut(df_markout['rv'], 5)
        df_markout['bfm_bin'] = pd.qcut(df_markout['bfm'], 5)
        df_markout['sfm_bin'] = pd.qcut(df_markout['sfm'], 5)
        df_markout['f_bin'] = pd.qcut(df_markout['f'].abs(), 5)
        df_markout['late_fill'] = df_markout.eval('cancel_time == cancel_time')
        bu_df = df_markout.query('status==4').copy()
        df_markout['bu0_bin'] = pd.qcut(bu_df['bu0'] / bu_df['rv'], 3)
        df_markout['bu1_bin'] = pd.qcut(bu_df['bu1'] / bu_df['rv'], 3)
        df_markout['bu2_bin'] = pd.qcut(bu_df['bu2'] / bu_df['rv'], 3)
        df_markout['bu3_bin'] = pd.qcut(bu_df['bu3'] / bu_df['rv'], 3)

        if not bu_df.empty and {'mid', 'price', 'side'}.issubset(bu_df.columns):
            gap = ((bu_df['mid'] - bu_df['price']) / bu_df['mid'] * bu_df['side']).dropna()
            if not gap.empty:
                stats = gap.agg(['mean', 'median']).to_dict()
                stats['10%'] = gap.quantile(0.1)
                stats['20%'] = gap.quantile(0.2)
                stats['80%'] = gap.quantile(0.8)
                stats['90%'] = gap.quantile(0.9)
                gap_stats_dict[instr] = stats

        markout_res_cols = [c for c in df_markout if 'markout' in c and 'adj' not in c]
        agg_funcs = [win_ratio, 'median', 'mean', 'std', n_count]
        markout_dict[instr] = df_markout[markout_res_cols].agg(agg_funcs).unstack(level=0)
        # pph(df_markout[markout_res_cols].agg(agg_funcs), f'{instr} markout all', level=5, show_result=show_result)

        from common.quants.corr import bucket_analyze_generic
        bu0_bin_markout = bucket_analyze_generic(df_markout, markout_res_cols, 'bu0_bin', agg_funcs)
        bu1_bin_markout = bucket_analyze_generic(df_markout, markout_res_cols, 'bu1_bin', agg_funcs)
        bu2_bin_markout = bucket_analyze_generic(df_markout, markout_res_cols, 'bu2_bin', agg_funcs)
        bu3_bin_markout = bucket_analyze_generic(df_markout, markout_res_cols, 'bu3_bin', agg_funcs)

        #from common.quants.corr import bucket_analyze_generic

        #pph(bucket_analyze_generic(df_markout, markout_res_cols, 'rv_bin', agg_funcs),
        #    f' {instr} Markout by rv', level=5, show_result=show_result)

        #pph(bucket_analyze_generic(df_markout, markout_res_cols, 'f_bin', agg_funcs),
        #    f' {instr} Markout by f', level=5, show_result=show_result)

        # pph(bucket_analyze_generic(df_markout, markout_res_cols, 'bfm_bin', [win_ratio, 'median']),
        #     f' {instr} Markout by bfm', show_result=show_result)

        # pph(bucket_analyze_generic(df_markout, markout_res_cols, 'sfm_bin', [win_ratio, 'median']),
        #     f' {instr} Markout by sfm', show_result=show_result)

        df_markout_by_kind[instr] = bucket_analyze_generic(df_markout, markout_res_cols,
                                                           'late_fill', agg_funcs + ['count']).T

        output['bu_markout'][instr] = [bu0_bin_markout, bu1_bin_markout, bu2_bin_markout, bu3_bin_markout]

        # print('================================================================')

    gap_df = (pd.DataFrame(gap_stats_dict).T[['mean', 'median', '10%', '20%', '80%', '90%']].sort_index())
    output['Gap_Analysis'] = gap_df
    output['Markout_Analysis'] = pd.concat(markout_dict, axis=1).T
    output['Markout_by_Kind'] = pd.concat(df_markout_by_kind, axis=1)

    output['df_merged'] = df_merged
    output['instr2dfs'] = instr2dfs
    return  output



def parallel_gen_results(in_fns, sdate, edate, fee_rate, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict, num_parallel=4):
    if isinstance(in_fns, str):
        in_fns = [in_fns]

    if num_parallel == 1 or len(in_fns) == 1:
        results = [gen_result(in_fns, sdate, edate, fee_rate, pnl_hzs, account_name, beta_dict)]
    else:
        in_fns_list = [[in_fn] for in_fn in in_fns]
        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            results = list(executor.map(gen_result, in_fns_list, [sdate]*len(in_fns_list), [edate]*len(in_fns_list), [fee_rate]*len(in_fns_list), [pnl_hzs]*len(in_fns_list), [account_name]*len(in_fns_list), [beta_dict]*len(in_fns_list )))
    return results

def gen_result(in_fns, sdate, edate, fee_rate, pnl_hzs=pnl_hzs, account_name='bin1', beta_dict=beta_dict):
    final_result = {}
    # if in_fns is string then convert it to list

    try:
        in_fns = [file for file in in_fns if os.path.exists(file)]
        in_fns = sorted(in_fns, key=os.path.getmtime)

        if in_fns == []:
            print('No log files found')
            print('in_fn:', in_fns)

        save_details = False

        parser = LogParser(in_fns, analyze_latency=True)
        order_stats, trade_metric, instr2dfs = parser.analyze(fee_rate, pnl_hzs, save_details, sdate, edate, account_name= account_name, only_total_stats = False)

        final_result['order_stats'] = order_stats
        final_result['trade_metric'] = trade_metric
        final_result['instr2dfs'] = instr2dfs

        df_order, df_panel = parser.get_order_and_panel(sdate, edate)
        final_result['df_order'] = df_order
        final_result['df_panel'] = df_panel
        return final_result
    except Exception as e:
        print(f'error parsing for {in_fns} due to {e}')
        return None


def analyze_slurm_sim(sdate, edate, sim_name, fee_rate=-0.3e-4, capital=1e6,
                      parent_folder='/mnt/sda/NAS/ShareFolder/bb/sim_slurm/',
                      suffix='.log', num_parallel=256, output_folder=f'/mnt/sda/NAS/ShareFolder/bb/sim_slurm/'):
    folder = parent_folder + sim_name

    df_res = run_pta_group(folder, sdate, edate, fee_rate, capital, suffix=suffix, num_parallel=num_parallel, n_process=2)

    df_res = df_res.sort_index()
    df_res.columns = ['_'.join(col) for col in df_res.columns]
    cols = 'total_pnl_sharpe total_pnl_win_ratio total_pnl_sum total_pnl_mean total_pnl_std total_pnl_min notional_mean gross_book_mean_mean net_book_mean_mean'.split()
    df_res['te'] = df_res.eval('notional_mean / gross_book_mean_mean')
    df_res['score'] = df_res.eval('total_pnl_sharpe * (2 * total_pnl_win_ratio - 1) * sqrt(1+te)') #score calc
    #df_res['score'] = df_res.eval('total_pnl_sharpe * (2 * total_pnl_win_ratio - 1) * log(1+te)') #score calc
    df_res = df_res.sort_values('score', ascending=False)[['score'] + cols].copy() #.query('total_pnl_win_ratio >= 0.6')
    output_folder = output_folder + f'{sim_name}'
    os.system(f'mkdir -p {output_folder}')
    df_res.to_parquet(f'{output_folder}/df_res.parquet')
    return df_res



if __name__ == '__main__':

    # single session
    # in_fns = [f'/nas-1/ShareFolder/bb/prod_logs/okx1/okx1_latest.log']
    # edate = (pd.Timestamp.now() + pd.Timedelta('1d')).strftime("%Y-%m-%d")
    # sdate = '2024-12-11'
    # fee_rate = -0.5e-4
    # pnl_hzs = [20, 180, 1800, 1e9]
    #
    # res = run_pta(in_fns, sdate, edate, fee_rate)
    # df_merged, instr2dfs = res['df_merged'], res['instr2dfs']

    # multi sessions
    # in_fns = sorted(glob.glob(
    #     '/nas-1/ShareFolder/bb/grid_search_output/okx_fwd1m_me0x54_II/DRC_60/posAdjMul_0.0/sprd2EgMul_0.25_2025-01-24_08-09-21/*txt.gz'))
    #
    # sdate, edate = '2024-11-01', '2024-12-01'
    #
    # fee_rate = -0.5e-4
    # pnl_hzs = [10, 60, 300, 1800, 1e9]
    # capital = 1e3
    #
    # res = run_pta_multi_session(in_fns, sdate, edate, fee_rate, capital, pnl_hzs, num_parallel=16)

    #pta group
    # sdate = '2024-11-01'
    # edate = '2024-11-30'
    # fee_rate = -0.5e-4
    # capital = 1e3
    # df_res = run_pta_group('/nas-1/ShareFolder/bb/grid_search_output/okx_fwd1m_me0x54/',
    #                        sdate, edate, fee_rate, capital, num_parallel=8)

    # in_fns = ['/nas-1/ShareFolder/bb/prod_logs/bin4/binance4_20250324101350.2.log',
    #           '/nas-1/ShareFolder/bb/prod_logs/bin4/binance4_20250324101350.1.log',
    #           '/nas-1/ShareFolder/bb/prod_logs/bin4/binance4_20250324101350.log']

    #in_fns = ['/nas-1/ShareFolder/bb/prod_logs/bin4/binance4_20250324101350.2.log']
    sdate, edate = '2025-06-24', '2025-06-26'
    #in_fns = ['/home/bb/sim/BTCUSDT.BNF_20241205_20241205.log']
    in_fns = ['/nas-1/ShareFolder/bb/prod_logs/bin4/binance4_latest.log']
    fee_rate = -3e-5
    capital = 1e6
    pnl_hzs = [30, 180, 1800, 1e9]
    capital = 1e4

    res = run_pta(in_fns, sdate, edate, fee_rate, capital)
    df_merged, instr2dfs = res['df_merged'], res['instr2dfs']
