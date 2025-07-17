import json
import re
import os
import gzip
import glob
from functools import cache

import pandas as pd
import numpy as np
from common.searchParams.utils import calc_trading_metrics, calc_order_stat

ORDER_STATUS = {
    'invalid':0,
    'pending':1,
    'new':2,
    'partial_fill':3,
    'fill':4,
    'cancel':5,
    'reject':6,
}

TRIGGER_ORDER = re.compile(r'(Core order|onUdsMsg|newOrder(?!RespType)|cancelOrder|Lws write|onApiMsg)')
TRIGGER_PANEL = 'F-core panel:'
TRIGGER_ORDER_UPDATE = 'OnRecvOrderUpdate'

ORDER_KEY_MAP = {
    'h': 'handle',
    't': 'action',
    'st': 'status',
    'p': 'px',
    'q': 'qty',
    'n': 'time',
    'ec': 'engine',
    'ot': 'order_type',
    'tif': 'time_inforce',
    'i': 'instr',
    'c': 'counter',
    'fill': 'traded_qty',
    'tg': 'trigger',
    's': 'side', # -1 is short, 1 is long
    'err': 'error',
    'r': 'r',
    'f': 'f',
    'bu0': 'bu0',
    'bu1': 'bu1',
    'bu2': 'bu2',
    'bu3': 'bu3',
}

ORDER_COLUMNS = [
    'handle', 'orig_handle', 'status', 'px', 'qty', 'time', 'engine', 'order_type', 'time_inforce',
    'instr', 'traded_qty', 'trigger', 'side', 'step', 'traded_qty_when_cancelling',
    'first_fill_time', 'all_fill_time',
    'send_time', 'send_time_logger', 'send_time_gateway', 'send_time_lws', 'send_ack_time_logger',
    'cancel_time', 'cancel_time_logger', 'cancel_time_gateway', 'cancel_time_lws', 'order_end_time_logger', 'late_cancel',
    'bu0', 'bu1', 'bu2', 'bu3']

FEE_TABLE = {
    "BTCUSDT.BNF": {
        "GTX": 0.0,
        "IOC": 0.0,
    },
    "BTCUSDC.BNF": {
        "GTX": 0.0,
        "IOC": 0.0,
    },
    "ETHUSDT.BNF": {
        "GTX": 0.0,
        "IOC": 0.0,
    },
    "SOLUSDT.BNF": {
        "GTX": 0.0,
        "IOC": 0.0,
    },
}

class OrderParser():
    def __init__(self, exchange):
        self.exchange = exchange
        self.order_dict = {}
        self.order_cancel_dict = {}
        self.order_update_dict = {}
        self.to_check_order_update = set()
        self.df = pd.DataFrame(columns=ORDER_COLUMNS)
        self.handle_offset = 0
        self.handle_max = 0
        self.done_get_df = False
        self.req_id_to_handle = {}
        self.instr_id = {
        }  #inject instr id here if necesary

    def update_instr_id(self, line):
        # [2024-12-08 13:59:35.562] [Sim] [info] [2376550:2376550] Init instrUniv 3=SOLUSDT.BNF
        line = line.upper()
        pattern = 'INIT INSTRUNIV'
        txt = line[line.find(pattern) + len(pattern):]
        try:
            key, value = txt.split('=')
            key = int(key.strip())  
            value = value.strip()  
            self.instr_id[key] = value
            # print('update instr_id', key, value)
        except ValueError:
            print(f"Error parsing instr update line: {line}")

    def parse_order(self, line):
        kw = 'Core order'
        try:
            #logger_time = pd.Timestamp(line.split(']')[0].strip('['))
            logger_time = pd.Timestamp(line.split('[')[0].strip(']'))
        except Exception:
            logger_time = pd.Timestamp(line.split(' INFO')[0])
            #logger_time = pd.Timestamp(line.split('[')[0].strip(']'))

        txt = line[line.find(kw) + len(kw):]
        txt = self.json_hack(txt)
        try:
            data = json.loads(txt)
        except Exception:
            pass
            # print('Error: Failed to parse core order : ', txt)
            # raise


        # order update json has 'ts' field
        if 'n' not in data:
            if isinstance(data['ts'], (int, float, str)):
                data['n'] = pd.Timestamp(data['ts'])
            elif isinstance(data['ts'], pd.Series):
                data['n'] = data['ts'].apply(lambda ts: pd.Timestamp(ts))
            else:
                raise TypeError(f"Unexpected type for 'ts': {type(data['ts'])}. Expected int, float, str, or pd.Series.")
        else:
            data['n'] = pd.Timestamp(data['n'])

        order = {mapped_k: data[k] for k, mapped_k in ORDER_KEY_MAP.items() if k in data}
        if order['action'] == 'U4C':  # warning information, ignore
            return
        # panel instr is string BTCUSDT.BNF, while order instr is index 0/1, here we unify it
        if 'instr' in order:  # cancel order log does not include instr
            assert order['instr'] in self.instr_id, f"Instrument {order['instr']} not found in instr_id dictionary."
            order['instr'] = self.instr_id[order['instr']]

        order.pop('counter', None)
        order['orig_handle'] = order['handle']
        order['handle'] += self.handle_offset
        self.handle_max = max(self.handle_max, order['handle'])
        handle = order.pop('handle')
        if handle in self.order_dict:
            # careful, dict.update is good since here we have only one layer dict
            self.order_dict[handle].update(order)
        else:
            self.order_dict[handle] = order

        # modify the value manually
        if order.get('error') == 'NOT CURRENT ORDER' and order['status'] < 4: # Do we still need this?
            self.check_order_update(handle)
        if order['action'] == 'send':
            self.order_dict[handle]['send_time'] = order['time']
            self.order_dict[handle]['send_time_logger'] = logger_time
        elif order['action'] == 'cancel':
            self.order_dict[handle]['cancel_time'] = order['time']
            self.order_dict[handle]['cancel_time_logger'] = logger_time
            self.order_dict[handle]['traded_qty_when_cancelling'] = self.order_dict[handle].get('traded_qty', 0)
        elif order['action'] == 'update':
            if (order['status'] == 3 or order['status'] == 4 ) and 'first_fill_time' not in self.order_dict[handle]:
                self.order_dict[handle]['first_fill_time'] = order['time']
            if order['status'] == 4:
                self.order_dict[handle]['all_fill_time'] = order['time']

    def parse_uds_resp(self, line):
        try:
            if self.exchange == 'binance':
                self.parse_uds_binance(line)
            elif self.exchange == 'okx':
                self.parse_uds_okx(line)
        except Exception:
            print('error parsing', line)

    def parse_uds_binance(self, line):
        logger_time = pd.Timestamp(line[:21])
        idx = line.find('onUdsMsg') + 9
        d = json.loads(line[idx:])
        if d['e'] != 'ORDER_TRADE_UPDATE':
            return
        status = d['o']['X']
        handle = int(d['o']['c']) + self.handle_offset
        if status == 'NEW':
            if handle not in self.order_dict:
                print('unexpected uds resp for handle', handle, line)
                return
            self.order_dict[handle]['send_ack_time_logger'] = logger_time
            return
        # possible status: canceled, filled, partially_filled
        if handle not in self.order_cancel_dict: # no cancel, directly filled
            return
        # assert cancelOrder appears before ORDER_TRADE_UPDATE in log
        # best to keep log sequence ordered by timestamp
        self.order_cancel_dict[handle]['order_end_time_logger'] = logger_time
        self.order_cancel_dict[handle]['late_cancel'] = status != 'CANCELED'

    def parse_uds_okx(self, line):
        logger_time = pd.Timestamp(line[:21])
        idx = line.find('onUdsMsg') + 9
        d = json.loads(line[idx:])
        if d.get('arg', {}).get('channel') != 'orders':
            return
        status = d['data'][0]['state']
        handle = int(d['data'][0]['clOrdId']) + self.handle_offset
        if status == 'live':
            if handle not in self.order_dict:
                print('unexpected uds resp for handle', handle, line)
                return
            self.order_dict[handle]['send_ack_time_logger'] = logger_time
            return
        # possible status: canceled, filled, partially_filled
        if handle not in self.order_cancel_dict: # no cancel, directly filled
            return
        # assert cancelOrder appears before ORDER_TRADE_UPDATE in log
        # best to keep log sequence ordered by timestamp
        self.order_cancel_dict[handle]['order_end_time_logger'] = logger_time
        self.order_cancel_dict[handle]['late_cancel'] = status != 'canceled'

    def parse_api_resp(self, line):
        ''' those rejected order won't get uds msg '''
        try:
            if self.exchange == 'binance':
                self.parse_api_binance(line)
            elif self.exchange == 'okx':
                self.parse_api_okx(line)
        except Exception:
            print('error parsing api resp', line)

    def parse_api_binance(self, line):
        logger_time = pd.Timestamp(line[:21])
        idx = line.find('onApiMsg') + 9
        d = json.loads(line[idx:])
        if d['status'] != 400 or d['error']['code'] != -5022:
            return
        # only -5022 left, which means GTX reject
        assert d['id'] in self.req_id_to_handle
        handle = self.req_id_to_handle[d['id']]
        self.order_dict[handle]['send_ack_time_logger'] = logger_time # send_rej_time_logger actually

    def parse_api_okx(self, line):
        # TODO
        return

    def parse_cancel_req(self, line):
        idx = line.find('cancelOrder') + 12
        logger_time = pd.Timestamp(line[:21])
        d = json.loads(line[idx:])
        if self.exchange == 'binance':
            handle = int(d['params']['origClientOrderId']) + self.handle_offset
            # only record the first cancel request RT
            if handle not in self.order_cancel_dict:
                self.order_cancel_dict[handle] = {'id': d['id']}
            if self.order_cancel_dict[handle]['id'] == d['id']:
                self.order_cancel_dict[handle]['cancel_time_gateway'] = pd.Timestamp(int(d['params']['timestamp']) * 1000000)
        elif self.exchange == 'okx':
            handle = int(d['args'][0]['clOrdId']) + self.handle_offset
            # only record the first cancel request RT
            if handle not in self.order_cancel_dict:
                self.order_cancel_dict[handle] = {'id': d['id']}
            if self.order_cancel_dict[handle]['id'] == d['id']:
                self.order_cancel_dict[handle]['cancel_time_gateway'] = logger_time

    def parse_new_req(self, line):
        idx = line.find('newOrder') + 9
        logger_time = pd.Timestamp(line[:21])
        d = json.loads(line[idx:])
        if self.exchange == 'binance':
            handle = int(d['params']['newClientOrderId']) + self.handle_offset
            if handle not in self.order_dict:
                self.order_dict[handle] = {}
            self.order_dict[handle]['send_time_gateway'] = pd.Timestamp(int(d['params']['timestamp']) * 1000000)
            self.req_id_to_handle[d['id']] = handle
        elif self.exchange == 'okx':
            handle = int(d['args'][0]['clOrdId']) + self.handle_offset
            if handle not in self.order_dict:
                self.order_dict[handle] = {}
            self.order_dict[handle]['send_time_gateway'] = logger_time # okx no ts in query_string

    def parse_lws(self, line):
        if self.exchange == 'binance':
            self.parse_lws_binance(line)
        elif self.exchange == 'okx':
            self.parse_lws_okx(line)

    def parse_lws_binance(self, line):
        idx = line.find('Lws write') + 10
        logger_time = pd.Timestamp(line[:21])
        d = json.loads(line[idx:])
        if d['method'] == 'order.place':
            handle = int(d['params']['newClientOrderId']) + self.handle_offset
            if handle not in self.order_dict:
                self.order_dict[handle] = {}
            self.order_dict[handle]['send_time_lws'] = logger_time
        elif d['method'] == 'order.cancel':
            handle = int(d['params']['origClientOrderId']) + self.handle_offset
            if handle not in self.order_cancel_dict:
                self.order_cancel_dict[handle] = {'id': d['id']}
            if self.order_cancel_dict[handle]['id'] == d['id']:
                self.order_cancel_dict[handle]['cancel_time_lws'] = logger_time

    def parse_lws_okx(self, line):
        idx = line.find('Lws write') + 10
        if line[idx] != '{': # not json, like 20241231 11:56:57.498 [info] Lws write ping
            return
        logger_time = pd.Timestamp(line[:21])
        d = json.loads(line[idx:])
        # some don't have op, some are subscribe
        if d.get('op') == 'order':
            handle = int(d['args'][0]['clOrdId']) + self.handle_offset
            if handle not in self.order_dict:
                self.order_dict[handle] = {}
            self.order_dict[handle]['send_time_lws'] = logger_time
        elif d.get('op') == 'cancel-order':
            handle = int(d['args'][0]['clOrdId']) + self.handle_offset
            if handle not in self.order_cancel_dict:
                self.order_cancel_dict[handle] = {'id': d['id']}
            if self.order_cancel_dict[handle]['id'] == d['id']:
                self.order_cancel_dict[handle]['cancel_time_lws'] = logger_time

    def fill_cancel_resp_time(self):
        for handle, v in self.order_cancel_dict.items():
            if 'order_end_time_logger' not in v:
                continue
            v.pop('id', None)
            self.order_dict[handle].update(v)

    def convert_type(self, df):
        for tc in ('time', 'first_fill_time', 'all_fill_time', 'send_time', 'cancel_time',
                   'send_time_logger', 'send_time_gateway', 'send_time_lws', 'send_ack_time_logger',
                   'cancel_time_logger', 'cancel_time_gateway', 'cancel_time_lws', 'order_end_time_logger'):
            if tc not in df.columns:
                continue
            df[tc] = pd.to_datetime(df[tc], format="%Y-%m-%dT%H:%M:%S.%f", errors='coerce')
        for k in ('qty', 'px', 'traded_qty', 'side'):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors='coerce')

    def consume(self):
        ''' combine df_increment into df, clear self.orders '''
        if not self.order_dict:
            return
        df_increment = pd.DataFrame.from_dict(self.order_dict, orient='index')
        df_increment['handle'] = df_increment.index
        self.convert_type(df_increment)
        self.df = pd.concat([self.df, df_increment], ignore_index=True)

    @cache
    def get_df(self):
        ''' TODO: add check. this function should be called ONLY ONCE. '''
        if self.done_get_df:
            return self.df
        self.consume()

        df_order = self.df
        df_order.set_index('handle', inplace=True)

        if df_order.empty:
            print('No order found in sim_log')
            raise ValueError('No order found in log')

                ############################## Make Sure that df time cols are timestamp
        for col in ['send_time', 'cancel_time', 'first_fill_time', 'all_fill_time', 'send_time_gateway']:
            if type(df_order[col].iloc[0]) == float:
                df_order[col] = pd.to_datetime(df_order[col], errors='coerce')
        ##### Added lifespan column
        # pos = df_order[['all_fill_time', 'cancel_time']].applymap(lambda x: type(x) == float)
        # print(df_order[['all_fill_time', 'cancel_time']][pos.any(axis=1)])
        df_order['end_time'] = df_order[['all_fill_time', 'cancel_time']].min(axis=1)
        # print('type of df_order[end_time]', type(df_order['end_time']))
        # print(df_order['end_time'])
        # print('type of df_order[send_time]', type(df_order['send_time']))
        # print('df_order[send_time]', df_order['send_time'])
        df_order['send_time'] = pd.to_datetime(df_order['send_time'], errors='coerce')
        df_order['end_time'] = pd.to_datetime(df_order['end_time'], errors='coerce')
        df_order['lifespan'] = df_order['end_time'] - df_order['send_time']
        df_order.loc[df_order['lifespan'] < pd.Timedelta('0ns'), 'lifespan'] = pd.Timedelta(seconds=0) # df_order['lifespan'] = df_order['lifespan'].apply(lambda x: pd.Timedelta(seconds=0) if x < pd.Timedelta('0ns') else x)
        df_order.drop(columns=['end_time'], inplace=True)
        ### df_order side column should be 1 or -1 int
        df_order['side'] = df_order['side'].astype(int)

        #print('done df_order.')
        self.done_get_df = True
        fill_zero_columns = ['traded_qty', 'traded_qty_when_cancelling']
        df_order[fill_zero_columns] = df_order[fill_zero_columns].fillna(0)
        self.df = df_order.sort_values('time')
        return df_order.copy()

    def json_hack(self, txt):
        txt = txt.replace('-nan', 'null')
        txt = txt.replace('\'nan\'', 'null')
        txt = txt.replace('nan', 'null')
        if " ,'bq':" not in txt and ", 'bq':" not in txt  and " 'bq':" in txt:
            txt = txt.replace(" 'bq':", " ,'bq':")
        if 'cancel' in txt and txt.endswith(','):
            txt = txt + " 'st':4}"
        if "CANCELLED', " not in txt and "CANCELLED' "  in txt:
            txt = txt.replace("CANCELLED' ", "CANCELLED', ")
        if "err:" in txt:
            txt = txt.replace("err:", "'err':")
        txt = txt.strip().replace("'", '"')
        if "-inf," in txt:
            txt = txt.replace("-inf,", "\"-inf\",")
        if "inf," in txt:
            txt = txt.replace("inf,", "\"inf\",")
        return txt

    def check_order_update(self, id):
        if id in self.order_update_dict and self.order_update_dict[id]['status'] >= 4:
            self.order_dict[id]['traded_qty'] = self.order_update_dict[id]['filledQty']
            self.order_dict[id]['status'] = self.order_update_dict[id]['status']
            if self.order_dict[id]['traded_qty'] > 0:
                self.order_dict[id]['first_fill_time'] = self.order_update_dict[id]['time']
                self.order_dict[id]['all_fill_time'] = self.order_update_dict[id]['time']
        else:
            self.to_check_order_update.add(id)

    def parse_order_update(self, line):
        kw = TRIGGER_ORDER_UPDATE
        time_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]"
        match = re.search(time_pattern, line)
        time_str = pd.NaT
        if match:
            time_str = match.group(1)
        order_update_dict = {'time': pd.to_datetime(time_str)}
        line = line[line.find(kw) + len(kw):].strip()
        for k, v in re.findall(r'(\w+)=(\d+\.\d+|\d+)', line):
            order_update_dict[k] = float(v) if '.' in v else int(v)
        id = order_update_dict['orderId']

        self.order_update_dict[id] = order_update_dict

        if id in self.to_check_order_update and order_update_dict['status'] >= 4:
            self.order_dict[id]['traded_qty'] = order_update_dict['filledQty']
            self.order_dict[id]['status'] = order_update_dict['status']
            if self.order_dict[id]['traded_qty'] > 0:
                self.order_dict[id]['first_fill_time'] = order_update_dict['time']
                self.order_dict[id]['all_fill_time'] = order_update_dict['time']
            self.to_check_order_update.remove(id)

    def update_handle_offset(self ):
        # print('update handle offset' , self.handle_offset, self.handle_max)
        self.handle_offset = self.handle_max


class PanelParser():
    def __init__(self, exchange):
        self.exchange = exchange
        self.panels = [] # list of dict
        columns = ['tr', 'instr', 'n', 'c', 'a', 'u', 'fv', 'pos', 'ae', 'fm', 'rv', 'f']
        self.df = pd.DataFrame(columns=columns).set_index('n', inplace=False)
        self.done_get_df = False
        
    def parse(self, line):
        kw = TRIGGER_PANEL
        line = line[line.find(kw) + len(kw):]
        line = self.json_hack(line)

        try:
            data = json.loads(line)
        except Exception:
            print('err', line)
            return
            #raise

        if data.get('n', '').startswith('1970-'):
            return
        if data.get('ns', 1) <= 0:  # not valid log
            return
        data['n'] = pd.Timestamp(data['n'])
        data.update(data.pop('panel')) # dict flatten
        self.panels.append(data)

    def consume(self):
        if not self.panels:
            return
        df_increment = pd.DataFrame(self.panels)
        self.convert_type(df_increment)
        self.panels.clear()
        self.df = pd.concat([self.df.dropna(axis=1, how='all'), df_increment], ignore_index=True) 
        
    @cache
    def get_df(self):
        if self.done_get_df:
            return self.df
        self.consume()
        df_panel = self.df.copy()

        df_panel.set_index('n', inplace=True)
        df_panel = df_panel.sort_index()
        self.df = df_panel
        #print('done df_panel.')
        self.done_get_df = True
        return df_panel

    def convert_type(self, df):
        for k in ('fm', 'bfm', 'sfm', 'f', 'bp', 'ap', 'c', 'a', 'u', 'fv', 'pos', 'maxLong', 'maxShort'):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors='coerce')

    def json_hack(self, line):
        line = line.replace('-nan', 'null')
        line = line.replace("'nan'", 'null')
        line = line.replace('nan', 'null')
        line = line.strip().replace("'", '"')
        if '"instr": "' not in line and '"instr":"' not in line and '"instr":' in line:
            line = line.replace('"instr":', '"instr":"')
        if '", "n":' not in line and '" , "n":' not in line and ', "n":' in line:
            line = line.replace(', "n":', '", "n":')
        if "-inf," in line:
            line = line.replace("-inf,", "\"-inf\",")
        if "inf," in line:
            line = line.replace("inf,", "\"inf\",")
        return line


class LogParser:
    def __init__(self, params, analyze_latency=False, exchange='binance'):
        #print(f'LogParser : {params}')
        self.order_parser = OrderParser(exchange=exchange)
        self.panel_parser = PanelParser(exchange=exchange)
        self.order_status = ORDER_STATUS
        self.order_keymap = ORDER_KEY_MAP
        self.fee_table = FEE_TABLE
        self.analyze_latency = analyze_latency
        self.errors = []

        if type(params) is str:
            self.parse_pattern(params)
        elif type(params) is list:
            self.parse_file_list(params)
        else:
            raise ValueError('params should be either a pattern of str or a list of files')
        self.order_parser.fill_cancel_resp_time()
        self.order_parser.get_df()
        self.panel_parser.get_df()

    def parse_line(self, line):
        if 'resend' in line or 'last_attempt' in line: #temp remove me RH
            return
        if 'Init instrUniv' in line or 'init instrUniv' in line:
            self.order_parser.update_instr_id(line)
        
        if 'Init finished' in line:
            try:
                self.order_parser.update_handle_offset()
            except Exception as e:
                print('Error: Failed to update handle offset : ', line)
                raise
        match = TRIGGER_ORDER.search(line)
        if match:
            if match.group(1) == 'Core order':
                try:
                    self.order_parser.parse_order(line)
                except Exception as e:
                    print('Error: Failed to parse core order : ', line)
                    raise
            elif match.group(1) == 'onUdsMsg' and self.analyze_latency:
                self.order_parser.parse_uds_resp(line)
            elif match.group(1) == 'onApiMsg' and self.analyze_latency:
                self.order_parser.parse_api_resp(line)
            elif match.group(1) == 'cancelOrder' and self.analyze_latency:
                self.order_parser.parse_cancel_req(line)
            elif match.group(1) == 'newOrder' and self.analyze_latency:
                self.order_parser.parse_new_req(line)
            elif match.group(1) == 'Lws write' and self.analyze_latency:
                self.order_parser.parse_lws(line)

        # elif TRIGGER_ORDER_UPDATE in line:
        #     self.order_parser.parse_order_update(line)
        elif TRIGGER_PANEL in line:
            self.panel_parser.parse(line)

    def get_order_and_panel(self, sdate='1970-01-01', edate='2070-01-01'):
        df_order = self.order_parser.get_df().query('@sdate <= time <= @edate').copy()
        df_panel = self.panel_parser.get_df().query('@sdate <= n <= @edate').copy()
        return df_order, df_panel

    def parse_file(self, file):
        assert file

        if file.endswith('.gz'):
            with gzip.open(file, 'rt') as lines:
                for line in lines:
                    try:
                        self.parse_line(line)
                    except Exception as e:
                        error_message = f"Error: Failed to parse line: {line} | Exception: {e}"
                        print(error_message)
                        self.errors.append(error_message)  # 将错误存储到 self.errors 中
        else:
            with open(file, 'r', encoding='utf-8', errors='ignore') as lines: ################sometimes utf-8 is not enough
                for line in lines:
                    try:
                        self.parse_line(line)
                    except Exception as e:
                        error_message = f"Error: Failed to parse line: {line} | Exception: {e}"
                        print(error_message)
                        self.errors.append(error_message)  # 将错误存储到 self.errors 中
                        
    def parse_pattern(self, pattern):
        log_files = []
        pattern = pattern+'/*'
        print('pattern:', pattern)
        for file in glob.glob(pattern):
            # if log or txt or log.gz
            if file.endswith('.log') or file.endswith('.txt') or file.endswith('.log.gz'):
                log_files.append(file)
        log_files.sort()
        print('processing:', '\n    '.join(log_files))
        for file in sorted(log_files):
            self.parse_file(file)
        self.save_dir = '/'.join(pattern.split('/')[:-1])

    def parse_file_list(self, file_list):
        if all(file_list[i] <= file_list[i+1] for i in range(len(file_list)-1)):
            file_list = file_list[::-1]
        for file in file_list:
            self.parse_file(file)
        self.save_dir = '/'.join(file_list[0].split('/')[:-1])

    @cache
    def calc_order_stats(self, sdate='1970-01-01', edate='2070-01-01'):
        return calc_order_stat(self.order_parser.df, sdate, edate)

    def analyze(self, fee_rate, pnl_hzs, save_detail=False, sdate='1970-01-01', edate='2070-01-01', return_detail=False, account_name='bin1', only_total_stats =True):
        df_order, df_panel = self.get_order_and_panel(sdate, edate)
        df_order.set_index('time', inplace=True)

        if only_total_stats:
            order_stats = self.calc_order_stats(sdate, edate)
        else:
            order_stats = {}
            # order_stats['total'] = self.calc_order_stats(sdate, edate)
            date_list = np.unique(df_order.index.date)
            instr_list = list(df_order['instr'].unique())
            for date in date_list:
                order_stats[str(date)] = {}
                df_order_date = df_order.query('index.dt.date == @date')
                end_date = date + pd.Timedelta(days=1)
                order_stats[str(date)]['total'] = calc_order_stat(df_order_date,date, end_date)
                for instr in instr_list:
                    df_order_temp = df_order_date.query('instr == @instr')
                    order_stats[str(date)][instr] = calc_order_stat(df_order_temp, date, end_date)
                
        if save_detail:
            df_order.to_parquet(os.path.join(self.save_dir, "order.parquet"))
            df_panel.to_parquet(os.path.join(self.save_dir, "panel.parquet"))

        metric_df, dict_df_detail = calc_trading_metrics({'order':df_order, 'panel':df_panel}, fee_rate, pnl_hzs, account_name=account_name)
        if return_detail:
            return  order_stats, metric_df, dict_df_detail, {'order':df_order, 'panel':df_panel}
        return order_stats, metric_df, dict_df_detail 

    @classmethod
    def get_order_from_log(cls, logPath) -> dict:
        order_dict = {}
        req_order_dict = {}
        for line in open(logPath):
            if 'newOrder' in line or 'cancelOrder' in line:
                local_ts = pd.Timestamp(line[:line.find('[')].strip())
                idx = line.find('"id"')
                x = line[idx - 1 : line.find('}}', idx) + 2]
                d = json.loads(x)
                method = d['method'].split('.')[1]
                if method not in {'place', 'cancel'}:
                    continue
                orderId = d['params'].get('newClientOrderId') or d['params'].get('origClientOrderId')
                assert orderId
                if orderId not in order_dict:
                    order_dict[orderId] = {'uds': []}
                order_dict[orderId][d['id']] = {
                    'action': method,
                    'gen_local_ts': pd.Timestamp(int(d['params']['timestamp']) * 1000000),
                    'send_local_ts': local_ts,
                }
                req_order_dict[d['id']] = orderId
            elif 'onUdsMsg {' in line:
                local_ts = pd.Timestamp(line[:line.find('[')].strip())
                idx = line.find('onUdsMsg')
                x = line[idx + 9 :]
                d = json.loads(x)
                if d['e'] == 'ORDER_TRADE_UPDATE':
                    orderId = d['o']['c']
                    order_dict[orderId]['uds'].append({
                        's': d['o']['X'],
                        'recv_local_ts': local_ts,
                    })
                elif d['e'] == 'TRADE_LITE':
                    orderId = d['c']
                    order_dict[orderId]['uds'].append({
                        's': 'TRADE_LITE',
                        'recv_local_ts': local_ts,
                    })
            elif 'onApiMsg {' in line:
                local_ts = pd.Timestamp(line[:line.find('[')].strip())
                idx = line.find('onApiMsg')
                x = line[idx + 9 :]
                d = json.loads(x)
                if d['status'] != 200: # normally, cancel reject
                    orderId = req_order_dict[d['id']]
                    order_dict[orderId][d['id']].update({
                        'resp': d['status'],
                        'recv_local_ts': local_ts,
                    })
                else:
                    if any(k in d['result'] for k in ('apiKey', 'listenKey')):
                        continue
                    orderId = d['result']['clientOrderId']
                    order_dict[orderId][d['id']].update({
                        'resp': d['result']['status'],
                        'recv_local_ts': local_ts,
                    })
        return order_dict

    @classmethod
    def analyze_lws_latency(cls, logPath):
        order_dict = cls.get_order_from_log(logPath)
        latency_dict = {}
        for v in order_dict.values():
            req_place_id = None
            for kk, req in v.items():
                if kk == 'uds':
                    continue
                latency_dict[kk] = {
                    'action': req['action'],
                    'resp': str(req['resp']),
                    'send_ts': req['send_local_ts'],
                    'recv_ts': req['recv_local_ts'],
                }
                if req['action'] == 'place':
                    req_place_id = kk
            if req_place_id is not None:
                for item in v['uds']:
                    if item['s'] == 'NEW':
                        latency_dict[req_place_id]['uds_recv_ts'] = item['recv_local_ts']

        df_latency = pd.DataFrame.from_dict(latency_dict, orient='index')

        df_place = df_latency[df_latency['action'] == 'place']
        print('\nplace api latency')
        df_place['api_latency_ms'] = (df_place['recv_ts'] - df_place['send_ts']).dt.total_seconds() * 1000
        print(df_place['api_latency_ms'].describe(percentiles=[.5, .9, .99]))
        print('\nplace uds latency')
        df_place['uds_latency_ms'] = (df_place['uds_recv_ts'] - df_place['send_ts']).dt.total_seconds() * 1000
        print(df_place['uds_latency_ms'].describe(percentiles=[.5, .9, .99]))

        df_cancel = df_latency[df_latency['action'] == 'cancel']
        df_cancel['api_latency_ms'] = (df_cancel['recv_ts'] - df_cancel['send_ts']).dt.total_seconds() * 1000
        print('\ncancel api latency')
        print(df_cancel['api_latency_ms'].describe(percentiles=[.5, .9, .99]))
        return df_latency

    @classmethod
    def analyze_sn_latency(cls, logPath):
        '''
        we only measure from post_data -> get uds, which excludes SN internal processing time
        make sure (post_data == send out time) and (uds response == receive time)
        '''
        order_dict = {}
        mapped_id = {}
        logPath = '/home/hftops2/yr_system/yr_system/log/Crypto.log'
        for line in open(logPath):
            if 'post_data: ' in line:
                idx = line.find('post_data: ')
                x = line[idx + 11 :]
                d = {}
                for kv in x.split('&'):
                    k, v = kv.split('=')
                    d[k] = v
                d['timestamp'] = int(d['timestamp']) # convert to int, otherwise str * 1000000, terrible
                if 'newClientOrderId' in d:
                    order_dict[d['newClientOrderId']] = {
                        'place_local_ts': pd.Timestamp(line[1:line.find(']')]),
                    } # key is str
                elif 'orderId' in d:
                    # d['orderId'] is str
                    if d['orderId'] not in mapped_id:
                        print('err', d['orderId'], 'not in mapped_id')  # file truncated
                        continue
                    order_dict[mapped_id[d['orderId']]]['cancel_local_ts'] = pd.Timestamp(line[1:line.find(']')])
            elif 'OnMessage: {' in line:
                idx = line.find('OnMessage: {')
                jsonend = line.find('}}', idx)
                if jsonend == -1:
                    continue
                x = line[idx + 11 : jsonend + 2]
                d = json.loads(x)
                if d['e'] != 'ORDER_TRADE_UPDATE':
                    continue
                orderId = d['o']['c'] # str
                if orderId not in order_dict:
                    continue  # file truncated
                # d['o']['i'] is number
                mapped_id[str(d['o']['i'])] = orderId # map i -> c, orderId -> clientOrderId
                status = d['o']['X']
                if f'uds_{status}_local_ts' in order_dict[orderId]:
                    continue
                order_dict[orderId].update({
                    f'uds_{status}_local_ts': pd.Timestamp(line[1:line.find(']')]),
                })
        df = pd.DataFrame.from_dict(order_dict, orient='index')
        df = df.reset_index().rename(columns={'index': 'handle'})
        print('SN')
        print('place latency')
        (df['uds_NEW_local_ts'] - df['place_local_ts']).describe(percentiles=[.25, .5, .99])
        print('\ncancel latency')
        (df['uds_CANCELED_local_ts'] - df['cancel_local_ts']).describe(percentiles=[.25, .5, .99])
        return df

def analyze_log(config,fee_rate, pnl_hzs, save_df=False):
    print("logDir:", config["logDir"])
    parser = LogParser(config["logDir"])
    print('analyzing .......')
    order_stats, metric_df, dict_df_detail = parser.analyze(fee_rate, pnl_hzs)
    return order_stats, metric_df, dict_df_detail, parser


if __name__ == '__main__':
    # path = ""
    # parser = LogParser()
    # parser.parse_pattern(path)
    # dict_df = parser.get_dfs()
    # print(dict_df['order'])
    # print(dict_df['panel'])
    # stats = parser.calc_order_stats()
    # print(stats)
    parser = LogParser(
        ['/home/bb/sim_output/drc_duration_60+max_drc_ratio_0.05_2024-10-11_08-21-19/2024-07-21_2024-07-21.stdout.txt.gz'])