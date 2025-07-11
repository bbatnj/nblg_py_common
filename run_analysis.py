from common.quants.pta import analyze_slurm_sim

def main():
    sdate       = '2024-12-03'
    edate       = '2025-12-06'
    sim_name    = 'sim'
    parent_dir  = '/home/shroy/Desktop/python1/nblg_py_common/common/data_sample/'
    output_dir  = '/home/shroy/Desktop/python1/nblg_py_common/output/'

    import os
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
