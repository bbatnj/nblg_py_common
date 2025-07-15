import os

from common.quants.pta import analyze_slurm_sim

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sdate       = '2024-12-03'
    edate       = '2025-12-06'
    sim_name    = 'sim'
    parent_dir = os.path.join(repo_root, "common", "data_sample")
    output_dir = os.path.join(repo_root, "output", "parquet")
    
    os.makedirs(output_dir, exist_ok=True)

    df_res = analyze_slurm_sim(
        sdate,
        edate,
        sim_name,
        parent_folder=parent_dir,
        suffix='.log.gz',
        num_parallel=4,
        output_folder=output_dir
    )
    print(df_res)

if __name__ == "__main__":
    main()
