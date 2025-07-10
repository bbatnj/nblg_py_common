import json

# 初始化字典
init_dict = {
  'DRC': 60,
  'posAdjMul': 0.5,
  'sprd2EgMul': [0.3],
  'BTCUSDC_MinEdgeB': 0.65,
  'BTCUSDC_sprd2EgMul': 0.2,
  'BTCUSDT_MinEdgeB': 0.3,
  'SOLUSDT_MinEdgeB': 0.55
}

config_template_fn = '/nas-1/ShareFolder/bb/cma_sim_test/SimCfgRel_OKF_Me0X54_1Min_with_events.json'

from common.searchParams.searchParams import CMA_search

CMA_search(init_dict, config_template_fn , title='cma', sigma0=1, POPSIZE=2, log_dir=None, fee_rate= -0.5e-4, capital=1e3)

pass