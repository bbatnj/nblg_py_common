import os
import gzip
import ast
import pandas as pd

def parse_panel_snapshots(log_path: str, out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    split_marker = " INFO F-core panel: "
    records = []

    with gzip.open(log_path, "rt") as f:
        for raw_line in f:
            line = raw_line.strip()
            if split_marker not in line:
                continue

            # Split into timestamp part and payload part
            timestamp_str, payload_str = line.split(split_marker, 1)

            # Parse the timestamp
            ts = pd.to_datetime(timestamp_str, format="%Y%m%d %H:%M:%S.%f")

            data = ast.literal_eval(payload_str)

            panel = data.pop("panel", {})
            
            record = {
                "instr":              data.get("instr"),
                "trades_since_last":  int(data.get("tr", 0)),
                "bfm":                float(data.get("bfm", 0)),
                "sfm":                float(data.get("sfm", 0)),
                "f":                  float(data.get("f", 0)),
                "rv":                 float(data.get("rv", 0)),
                "bid_px":             float(data.get("bp", 0)),
                "ask_px":             float(data.get("ap", 0)),
                "ts":                 ts,
            }
            # Merge in every key from the nested panel dict
            for key, value in panel.items():
                record[key] = float(value)

            records.append(record)

    # Create a DataFrame, index it by timestamp, sort, and write out
    df_panel = pd.DataFrame(records)
    df_panel = df_panel.set_index("ts").sort_index()
    out_file = os.path.join(out_dir, "panel.parquet")
    df_panel.to_parquet(out_file)
    print(f"Wrote {len(df_panel)} snapshots → {out_file}")


if __name__ == "__main__":
    # derive the repo root from this script’s location
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    log_path = os.path.join(
        repo_root,
        "common", "data_sample", "sim", "sim@1",
        "BTCUSDT.BNF_20241204.log.gz"
    )

    out_dir = os.path.join(
        repo_root,
        "output", "parquet", "sim",
        "BTCUSDT.BNF_20241204"
    )

    parse_panel_snapshots(log_path, out_dir)
