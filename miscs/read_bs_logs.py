import re
import ast
import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial
import argparse
import os
import gzip
### Explaination:
# The script is used to extract logging data from a log file into DataFrames and save them as parquet files (named by the key in the logging data).

def read_and_concat_parquet_files(parquet_dir, key):
    parquet_files = [f for f in os.listdir(parquet_dir) if key in f and f.endswith('.parquet')]

    parquet_files.sort()

    df_list = []

    for parquet_file in tqdm.tqdm(parquet_files, desc='Reading Parquet Files', unit='file',total=len(parquet_files)):
        parquet_file_path = os.path.join(parquet_dir, parquet_file)
        
        df = pd.read_parquet(parquet_file_path)
        
        df_list.append(df)

    if df_list:
        combined_df = pd.concat(df_list)
        combined_df.sort_index(inplace=True)
    else:
        combined_df = pd.DataFrame()  

    return combined_df

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        if parent_key == 'basic':
            new_key = k
        else:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def process_line(line, pattern):
    match = pattern.search(line)
    if match:
        timestamp = match.group(1)
        key = match.group(2)
        value_str = match.group(3)
        try:
            value_dict = ast.literal_eval(value_str)
            flat_dict = flatten_dict(value_dict)
            flat_dict['timestamp'] = timestamp
            return key, flat_dict
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing dictionary in line: {line}")
            print(f"Error: {e}")
    return None
def process_lines_chunk(lines_chunk, pattern):
    chunk_logging_data = {}
    for line in lines_chunk:
        result = process_line(line, pattern)
        if result:
            key, flat_dict = result
            if key not in chunk_logging_data:
                chunk_logging_data[key] = []
            chunk_logging_data[key].append(flat_dict)
    return chunk_logging_data

def extract_logging_data_to_df(log_file_path, save_parquet_path=None, num_processes=50, conver_n=True):
    logging_data = {}
    pattern = re.compile(r"\[(.*?)\] \[Sim\] \[info\] \[\d+:\d+\] BS_Logging, (\S+), (.+)")


    if log_file_path.endswith('.gz'):
        with gzip.open(log_file_path, 'rt') as file:
            lines = file.readlines()
    else:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
    
    num_processes = num_processes or mp.cpu_count()
    chunk_size = len(lines) // num_processes

    with mp.Pool(processes=num_processes) as pool:
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        partial_process_lines = partial(process_lines_chunk, pattern=pattern)
        results = list(tqdm.tqdm(pool.imap(partial_process_lines, chunks), total=len(chunks)))

    for chunk_result in results:
        for key, flat_dicts in chunk_result.items():
            if key not in logging_data:
                logging_data[key] = []
            logging_data[key].extend(flat_dicts)


    df_logging_data = {key: pd.DataFrame(value).set_index('timestamp') for key, value in logging_data.items()}
    # conver n
    if conver_n:
        for key, df in df_logging_data.items():
            if 'n' in df.columns:
                df['n'] = pd.to_datetime(df['n'])
                df.set_index('n', inplace=True)
    if save_parquet_path:
        for key, df in df_logging_data.items():
            df.to_parquet(f'{save_parquet_path}/{key}.parquet')
    
    return df_logging_data
 
def process_log_files_in_directory(log_file_dir, save_parquet_dir, num_processes=50):
    if not os.path.exists(save_parquet_dir):
        os.makedirs(save_parquet_dir)

    log_files = [f for f in os.listdir(log_file_dir) if 'log' in f and f.endswith('.gz')]

    log_files.sort()

    for log_file in tqdm.tqdm(log_files, desc="Processing files",total=len(log_files)):
        log_file_path = os.path.join(log_file_dir, log_file)
        print(f"Processing file: {log_file_path}")

        df_logging_data = extract_logging_data_to_df(log_file_path, None, num_processes)

        for key, df in df_logging_data.items():
            df.sort_index(inplace=True)

            index_time_str = df.index[0].strftime('%Y%m%d_%H%M%S')

            parquet_file_path = os.path.join(save_parquet_dir, f'{index_time_str}_{key}.parquet')
            df.to_parquet(parquet_file_path)

    print("All files processed and saved successfully.")

# # Example usage:
# python /home/bb/Crypto/read_bs_logs.py --log_file_path '/mnt/sda/NAS/ShareFolder/bb/BBCrypto.log_2024-08-11_20-30.69' --save_parquet_path '/home/bb/Crypto/test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract logging data from a log file into DataFrames.')
    parser.add_argument('--log_file_dir', type=str, help='Path to the log file.')
    parser.add_argument('--save_parquet_dir', type=str,  help='Directory to save the parquet files.')

    parser.add_argument('--num_processes', default=50, type=int, help='Number of processes to use')
    args = parser.parse_args()
    
    log_file_dir = args.log_file_dir
    save_parquet_dir = args.save_parquet_dir
    num_processes = args.num_processes
    
    os.makedirs(save_parquet_dir, exist_ok=True)
    

    process_log_files_in_directory(log_file_dir, save_parquet_dir, num_processes=50)