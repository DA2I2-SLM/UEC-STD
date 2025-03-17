import os
import re
import argparse
from collections import defaultdict

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Extract mse and mae using regex
    mse_match = re.search(r"mse:([0-9.]+)", content)
    mae_match = re.search(r"mae:([0-9.]+)", content)

    if mse_match and mae_match:
        mse = float(mse_match.group(1))
        mae = float(mae_match.group(1))
        return mse, mae
    return None, None

def aggregate_results(directory):
    results = defaultdict(lambda: {'mse': [], 'mae': []})

    for fname in os.listdir(directory):
        if 'metrics' in fname:
            parts = fname.split('-')
            if len(parts) == 3:
                _, i, j = parts
                j = j.strip()
                filepath = os.path.join(directory, fname)
                mse, mae = parse_file(filepath)
                if int(i)<=Lmax and int(i)>=Lmin:
                    if mse is not None and mae is not None:
                        results[j]['mse'].append(mse)
                        results[j]['mae'].append(mae)
    print(results)

    # Print average metrics per mode
    for j_mode, metrics in sorted(results.items()):
        mse_avg = sum(metrics['mse']) / len(metrics['mse']) if metrics['mse'] else float('nan')
        mae_avg = sum(metrics['mae']) / len(metrics['mae']) if metrics['mae'] else float('nan')
        print(f"Error model mode {j_mode}: avg MSE = {mse_avg:.6f}, avg MAE = {mae_avg:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average MSE and MAE from result files.")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing result files (default: current dir)")
    parser.add_argument("--Lrange", type=str, default="0,1000", help="range of L")
    print("READ RESULTS>>>>>")
    args = parser.parse_args()
    Lmin =  int(args.Lrange.split(",")[0])
    Lmax =  int(args.Lrange.split(",")[1])
    aggregate_results(args.dir)
