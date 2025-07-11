import os
import sys
import json
import datetime
import itertools
import argparse
from turtledemo.sorting_animate import partition

from common.quants.time_series_cv import parse_tscv_res


def load_param_dict(param_config_path):
    """
    Load parameter dictionary from a JSON file.
    
    Args:
        param_config_path (str): Path to the JSON config file containing parameter search ranges
        
    Returns:
        dict: Dictionary of parameters with their search ranges
    """
    try:
        with open(param_config_path, 'r') as f:
            param_dict = json.load(f)
            
        # Validate that each parameter has a list of values
        for key, values in param_dict.items():
            if not isinstance(values, list):
                print(f"Warning: Parameter '{key}' doesn't have a list of values. Converting single value to list.")
                param_dict[key] = [values]
                
        return param_dict
    except FileNotFoundError:
        print(f"Error: Parameter config file '{param_config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{param_config_path}' as valid JSON.")
        sys.exit(1)


def generate_slurm_commands(args, param_dict):
    simCfg_path = args.simCfg
    nblgLibCfg_path = args.libCfg
    program_path = os.path.abspath(args.program)
    title_path = os.path.join(args.prefix, args.title)
    product_kind = args.product_kind

    # Load simCfg.json
    with open(simCfg_path, 'r') as f:
        simCfg = json.load(f)
        start_date = datetime.datetime.strptime(simCfg["startDate"], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(simCfg["endDate"], "%Y-%m-%d")
        if simCfg.get('logDir') and simCfg['logDir'] != args.prefix:
            simCfg['logDir'] = title_path
        print('note: log path', title_path)

    # Compute days between start_date and end_date
    day_count = (end_date - start_date).days + 1

    # Load nblgLibCfg.json as a template
    print(f'nblgLibCfg_path : {nblgLibCfg_path}')
    with open(nblgLibCfg_path, 'r') as f:
        nblgLibCfg_template = f.read()

    # Separate search keys from fixed keys
    search_keys = {k: v for k, v in param_dict.items() if len(v) > 1 and k[0] != '_'}
    fixed_keys = {k: v[0] for k, v in param_dict.items() if len(v) == 1 and k[0] != '_'}

    # Generate cartesian product of search parameters
    if product_kind.upper() == 'CARTESIAN':
        search_combinations = list(itertools.product(*search_keys.values()))
    elif product_kind.upper() == 'PAIRWISE':
        # For pairwise, handle parameters with dimension 1
        if search_keys:
            # Find the maximum length of parameter lists
            max_length = max(len(values) for values in search_keys.values())
            
            # Extend any parameter with dimension 1 to match max_length
            extended_param_values = []
            for key, values in search_keys.items():
                if len(values) == 1:
                    # If parameter has only one value, repeat it to match max_length
                    extended_param_values.append(values * max_length)
                elif len(values) == max_length:
                    # If parameter has exactly max_length values, use it as is
                    extended_param_values.append(values)
                else:
                    raise ValueError(f"Parameter '{key}' has {len(values)} values, but the length must be either 1 or {max_length} for pairwise mode.")
            
            # Now do the pairwise matching
            search_combinations = list(zip(*extended_param_values))
            print(f"Pairwise combinations after extending dim-1 parameters: {len(search_combinations)}")
        else:
            search_combinations = []
    else:
        raise ValueError(f"Unknown product kind: {product_kind}")


    # to store soft links to the actual config.json, for easier access
    config_dir = os.path.join(title_path, 'config')
    os.makedirs(config_dir, exist_ok=True)

    for idx, values in enumerate(search_combinations):
        param_dict = {}

        # Replace template placeholders
         
        param_content = nblgLibCfg_template
        for key, value in zip(search_keys.keys(), values):
            enclosed_key = f"@{key}@"
            assert enclosed_key in param_content, f'{enclosed_key} not found'
            param_content = param_content.replace(enclosed_key, str(value))
            param_dict[key] = value

        for key, value in fixed_keys.items():
            enclosed_key = f"@{key}@"
            assert enclosed_key in param_content, f'{enclosed_key} not found'
            param_content = param_content.replace(enclosed_key, str(value))

        expr_name = '_'.join(f'{k}@{v}' for k, v in sorted(param_dict.items()))
        expr_path = os.path.join(title_path, expr_name)
        os.makedirs(expr_path, exist_ok=True)

        param_file_path = os.path.join(expr_path, "nblgLibCfg.json")
        with open(param_file_path, 'w') as f:
            f.write(param_content)

        for day_offset in range(day_count):
            day = start_date + datetime.timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")

            # Copy and modify simCfg.json
            simCfg_copy = simCfg.copy()
            simCfg_copy["startDate"] = day_str
            simCfg_copy["endDate"] = day_str
            simCfg_copy["nblgJsonPath"] = param_file_path
            simCfg_copy["logFilePath"] = os.path.join(expr_path, f"{day_str}.log")

            simCfg_path_copy = os.path.join(expr_path, f"simCfg_{day_str}.json")
            with open(simCfg_path_copy, 'w') as f:
                json.dump(simCfg_copy, f, indent=4)

# Define Slurm task command
#            log_dir = os.path.join(expr_path, f"logslurm_{day_str}.out")
#            slurm_cmd = f"sbatch --partition=ECS --array=1 --cpus-per-task=1 --mem-per-cpu=6G -o {log_dir} --wrap=\"{program_path} {simCfg_path_copy}\""
#            tasks.append(slurm_cmd)

            # Create a soft link under title_path for easy access
            job_id = idx * day_count + day_offset + 1
            os.system(f'ln -s {simCfg_path_copy} {title_path}/config/slurm_{job_id}')



    log_dir = os.path.join(title_path, 'slurm_log')
    os.makedirs(log_dir, exist_ok=True)

    # total num restriction
    job_num = len(search_combinations) * day_count
    use_cpu_num = 2 * job_num
    # 32 nodes, each 2*64*2 ( 2 CPU * 64 core * 2 thread)
    #assert use_cpu_num < 4096, 'folei: no more than half of all resource'

    # Define Slurm task command
    cloud_partition = args.partition
    slurm_cmd = f"sbatch --partition={cloud_partition} --array=1-{job_num}%{args.max_concurrent_job} --cpus-per-task=2 --mem-per-cpu=3G -o {log_dir}/slurm_%a.out --wrap=\"{program_path} slurm {title_path}/config\""
#    slurm_cmd = f"sbatch --partition=ECS --array=1-{job_num}%{args.max_concurrent_job} --cpus-per-task=2 --mem-per-cpu=3G -o {log_dir}/slurm_%a.out --wrap=\"{program_path} slurm {title_path}/config\""
    cmds = [slurm_cmd]
    # Write run.sh
    run_sh_path = os.path.join(title_path, "run.sh")
    cmds.extend([
        '# squeue -u USERNAME',
        '# squeue -j JOBID',
        '# scontrol show job JOBID',
        '# scancel JOBID',
    ])
    with open(run_sh_path, 'w') as f:
        f.write("\n".join(cmds) + "\n")
    os.chmod(run_sh_path, 0o744)
    print(f"Slurm tasks written to {run_sh_path}")


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Slurm grid search experiment generator")
    #          CPU : RAM
    #AMD_G     1   : 4    AMD 382C , 1.5T
    #AMD_R     1   : 8    AMD 256C , 2.0T
    #GPU_4090  1   : 5.3, AMD 192C , 1.0T 8x4090 per machine
    #ECS       elastic capacity (not ready as of Apr  7, 2025)
  
    parser.add_argument("--partition", type=str, default="AMD_R", help="compute partition, can be AMD_G AMD_R GPU_4090 ECS")
    parser.add_argument("--product_kind", type=str, default="Cartesian",help="Cartesian or PAIRWISE")
    parser.add_argument("--simCfg", type=str, default="simCfg.json", help="Path to the simCfg.json configuration file")
    parser.add_argument("--libCfg", type=str, default="nblgLibCfg.json", help="Path to the nblgLibCfg.json template file")
    parser.add_argument("--program", type=str, default="NblgCryptoSim", help="Path to the executable program")
    parser.add_argument("--prefix", type=str, default="/mnt/sda/NAS/ShareFolder/bb/sim_slurm", help="Base directory for sim_slurm")
    parser.add_argument("--max_concurrent_job", type=int, default=2500, help="Control how many jobs can be run concurrently")
    parser.add_argument("--title", type=str, required=True, help="Experiment title name")
    parser.add_argument("--param_config", type=str, default="param_config.json", help="Path to strategy parameter JSON file")

    args = parser.parse_args()

    # Check if input files and program exist
    for filepath in [args.simCfg, args.libCfg, args.program, args.param_config]:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' does not exist.")
            sys.exit(1)

    # Check if prefix directory exists
    if not os.path.exists(args.prefix):
        print(f"Creating directory: {args.prefix}")
        os.makedirs(args.prefix, exist_ok=True)

    # Create experiment directory
    title_path = os.path.join(args.prefix, args.title)
    os.makedirs(title_path, exist_ok=False)

    # Load parameter dictionary from config file
    param_dict = load_param_dict(args.param_config)
    
    # Print parameter summary
    print("Parameter search space:")
    for key, values in param_dict.items():
        print(f"  {key}: {values}")
    
    # Generate Slurm commands
    generate_slurm_commands(args, param_dict)
