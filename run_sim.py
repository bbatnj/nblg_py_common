import json
import argparse
import pathlib
import subprocess
import os
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from copy import deepcopy
from shutil import copyfile

from common.log_parser import LogParser


def get_config(cfg_path, title, ts=None):
    assert os.path.isabs(cfg_path), f'use absolute path for {cfg_path}'

    with open(cfg_path) as f_config:
        config = json.load(f_config)
    print('logDir:',config['logDir'])
    log_dir = config['logDir']
    username = os.getlogin() or os.getenv('USER')

    if r'${USER}' in log_dir:
        log_dir = log_dir.replace(r'${USER}', username)
    assert not log_dir or os.path.isabs(log_dir), 'use absolute path for logDir'
    # use ts to differentiate backtest output, unless designated
    if ts is None:
        ts = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    latest_link_path = os.path.join(log_dir, 'latest')

    if os.path.exists(latest_link_path) or os.path.islink(latest_link_path):
        os.remove(latest_link_path)
    log_dir = os.path.join(log_dir, f'{title}_{ts}')
    config['logDir'] = log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('Sim output folder is located in', log_dir)

    os.system(f'rm -f /home/{username}/sim_latest')
    os.system(f'ln -s {log_dir} /home/{username}/sim_latest')

    # put config and log under this log_dir
    # simlibcfg copy
    assert os.path.isabs(config['nblgJsonPath']), 'use absolute path for nblgJsonPath'
    simLibCfgCopyPath = os.path.join(log_dir, os.path.basename(config['nblgJsonPath']))
    copyfile(config['nblgJsonPath'], simLibCfgCopyPath)
    copyfile(cfg_path, os.path.join(log_dir, os.path.basename(cfg_path)))
    config['nblgJsonPath'] = simLibCfgCopyPath

    return config


def prepare_sim_configs(config: dict):
    # Parse the start and end dates
    start_date = datetime.strptime(config['startDate'], "%Y-%m-%d")
    end_date = datetime.strptime(config['endDate'], "%Y-%m-%d")
    n = config.pop('multiprocessCount')

    # Calculate the total number of days
    total_days = (end_date - start_date).days + 1

    # Calculate the number of days per process
    days_per_process = total_days // n
    remainder = total_days % n

    # Split the date range
    config_list = []
    current_date = start_date
    for i in range(n):
        if current_date > end_date: # process count > total_days
            break
        if remainder > 0:
            process_days = days_per_process + 1
            remainder -= 1
        else:
            process_days = days_per_process

        next_date = current_date + timedelta(days=process_days - 1)
        cfg_copy = deepcopy(config)
        cfg_copy['startDate'] = current_date.strftime("%Y-%m-%d")
        cfg_copy['endDate'] = next_date.strftime("%Y-%m-%d")
        logDir = cfg_copy['logDir']
        cfg_copy['logFilePath'] \
            = f"{logDir}/{cfg_copy['startDate']}_{cfg_copy['endDate']}.log" if not cfg_copy['zipLogFile'] else ""
        cfg_copy['cfgFileName'] = f'cfg_{cfg_copy["startDate"]}_{cfg_copy["endDate"]}.json'
        cfg_copy['stdout'] = f'{cfg_copy["startDate"]}_{cfg_copy["endDate"]}.stdout.txt'
        config_list.append(cfg_copy)
        current_date = next_date + timedelta(days=1)
        print(f'process {i + 1}: {cfg_copy["startDate"]} to {cfg_copy["endDate"]}')

    return config_list


def run_in_one_process(config):
    executable_name = 'NblgCryptoSim'

    # NblgCryptoSim executable
    cwd = pathlib.Path(__file__).parent
    sim_dir = config['simDir']
    assert sim_dir, f'provide simDir in config to find {executable_name}'
    if not os.path.isabs(sim_dir):
        sim_dir = cwd / sim_dir
    assert os.path.exists(f'{sim_dir}/{executable_name}')

    # simcfg copy 
    config_path = os.path.join(config['logDir'], config['cfgFileName'])

    with open(config_path, 'w') as f_config:
        json.dump(config, f_config, indent=4)

    script_template = '''
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/sda/NAS/Release/external/boost/v1.64.0/lib

ulimit -c unlimited

'''

    if not config.get('zipLogFile'):
        script_template += 'nice {sim_dir}/{executable_name} {config_path} > /dev/null'
    else:
        gz_fn = f'{config["startDate"]}_{config["endDate"]}.stdout.txt.gz'
        script_template += 'nice {sim_dir}/{executable_name} {config_path} 2>&1 | nice gzip -9 -c > ' + f'{config["logDir"]}/{gz_fn}'
    #

    scripts_txt = script_template.format(
        sim_dir=sim_dir, 
        config_path=config_path,
        executable_name=executable_name,
    )
    commands = [line.strip() for line in scripts_txt.splitlines() if line.strip()]
    print(commands)
    with open(os.path.join(config['logDir'], 'debug.sh'), 'w') as f_debug:
        sh_commands = commands[:]
        sim_executable_path = os.path.realpath(f'{sim_dir}/{executable_name}')
        sh_commands[-1] = f'gdb --args {sim_executable_path} {config_path}'
        for cmd in sh_commands:
            f_debug.write(f'{cmd}\n')
    f = subprocess.run(' && '.join(commands), shell=True, cwd=cwd)
    f.check_returncode()


def compress_logs(wd):
    compress_script = '''find . \( -name "*.log" -o -name "*.txt" \) -print0 | xargs -0 -n 1 -P 0 gzip -9'''
    f = subprocess.run(compress_script, shell=True, cwd=wd)
    f.check_returncode()

def save_dict_to_json(dict, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    cfg_path = os.path.join(save_path, name)
    with open(cfg_path, 'w') as f:
        json.dump(dict, f, indent=4)
    print(f"save {name} in {save_path}")        
    return cfg_path

def run_sim(cfg_path, title, config_save_path=None, name=None):
    config = get_config(cfg_path, title)
    configs = prepare_sim_configs(config)

    try:
        p = Pool(processes=len(configs))

        for _ in p.map(run_in_one_process, configs):
            pass
    except Exception as e:
        print(f'error running sim due to {e}')

    time.sleep(1)  # Pause for a moment to ensure all logs are written
    compress_logs(config['logDir'])
    time.sleep(1)

    output_config_path = None
    if config_save_path is not None:
        output_config_path = save_dict_to_json(config, config_save_path, name)

    result = {'output_config_path': output_config_path, 'config': config}
    return result

def analyze_simulation_metrics(result):
    config = result['config']
    stat, metric_df, dict_df_detail = None, None, None

    log_parser = LogParser(config['logDir'] + '/*.stdout.txt.gz')
    stat, metric_df, dict_df_detail = log_parser.analyze()
    print(json.dumps(stat, indent=4))
    
    metric_file_path = os.path.join(config['logDir'], "metric.csv")
    metric_df.to_csv(metric_file_path)

    result['stat'] = stat
    result['metrics'] = metric_df
    result['dict_df_detail'] = dict_df_detail
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="runSim in cpp_nblg")
    parser.add_argument('title', type=str, help='To differentiate each sim experiment')
    parser.add_argument('cfg_path', type=str, help='Path to the configuration file')
    parser.add_argument(
        '--metric',
        action='store_true',
        default=False,
        help='calc metric')
    parser.add_argument(
        '--save_df',
        action='store_true',
        default=False,
        help='store dataframe as parquet file')
    args = parser.parse_args()

    simulation_result = run_sim(args.cfg_path, args.title, args.metric, args.save_df)
    metrics_result = analyze_simulation_metrics(simulation_result, calc_metric=True)
    print('all done.')
