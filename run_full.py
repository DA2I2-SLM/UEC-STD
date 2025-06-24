import argparse
import os
from distutils.util import strtobool

def parse_arg_list(s):
    return [x.strip() for x in s.split(',') if x.strip()]

def str_to_bool(value):
    return bool(strtobool(value))

parser = argparse.ArgumentParser()
parser.add_argument('--pred_lengths_train', type=int, default=336,
    help='Training length for the ecm')
parser.add_argument('--pred_lengths', type=str, default="96,192,336,720",
    help='Comma-separated prediction lengths')
parser.add_argument('--error_flags', type=str, default="0,0.1,0.3,0.5,0.7,1",
    help='Comma-separated error model flags')
parser.add_argument('--data_type', type=str, required=True,
    help='Comma-separated dataset types, e.g., ETT,Weather,Traffic')
parser.add_argument('--data', type=str, required=True,
    help='Comma-separated dataset names, e.g., ETTh1,ETTh2')
parser.add_argument('--model', type=str, required=True,
    help='Comma-separated model names, e.g., TimeMixer,Informer')
parser.add_argument('--ecm', type=str, required=True,
    help='Comma-separated ECMs (e.g., linear,mlp)')
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--seasonal_trend', type=str_to_bool, 
                     help='Enable seasonal and trend decomposition', default="False")
parser.add_argument('--season_coef', type=float, default=0.5, help="weight of seasonal interpolation")
parser.add_argument('--trend_coef', type=float, default=0.5, help="weight of trend interpolation")

args = parser.parse_args()

pred_lengths = parse_arg_list(args.pred_lengths)
pred_lengths_train = args.pred_lengths_train
error_model = [eval(x) for x in parse_arg_list(args.error_flags)]
data_types = parse_arg_list(args.data_type)
datasets = parse_arg_list(args.data)
models = parse_arg_list(args.model)
ecms = parse_arg_list(args.ecm)

if not args.seasonal_trend:
    if len(data_types) == 1:
        data_types *= len(datasets)
    elif len(data_types) != len(datasets):
        raise ValueError("Length of --data_type must be 1 or match length of --data")

    # Align lengths: repeat data_type if only one is given
    if len(data_types) == 1:
        data_types *= len(datasets)
    elif len(data_types) != len(datasets):
        raise ValueError("Length of --data_type must be 1 or match length of --data")

    if args.step == 1:
        # === STEP 1: Run the "main" script first for each model-dataset pair ===
        print("Running initial scripts for each model-dataset pair...")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    pre_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}.sh"
                else:
                    pre_script = f"./scripts/long_term_forecast/{data_type}_script/{model}.sh"
                if os.path.isfile(pre_script):
                    print(f"[PRE-RUN] Running initial script: {pre_script}")
                    os.system(f"bash {pre_script}")
                else:
                    print(f"[WARN] Initial script not found: {pre_script}")
    elif args.step == 2:
        # === STEP 2: Train all ECM + pred_length + error_flag combinations ===
        print("Train ECM only.")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}_test.sh"
                else:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_test.sh"
                if not os.path.isfile(bash_script):
                    print(f"Warning: script not found: {bash_script}")
                    continue
                for ecm in ecms:
                    for pred_len in pred_lengths:
                        cmd = f"bash {bash_script} 3 1 {pred_lengths_train} 1 {ecm} {args.season_coef} {args.trend_coef}"
                        print(f"Running: {cmd}")
                        os.system(cmd)
    elif args.step == 3:
        # === STEP 3: Eval all ECM + pred_length + error_flag combinations ===
        print("Eval ECM + pred_length + error_flag combinations.")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}_test.sh"
                else:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_test.sh"
                if not os.path.isfile(bash_script):
                    print(f"Warning: script not found: {bash_script}")
                    continue
                for ecm in ecms:
                    for pred_len in pred_lengths:
                        for err_flag in error_model:
                            cmd = f"bash {bash_script} 4 1 {pred_len} {err_flag} {ecm} {args.season_coef} {args.trend_coef}"
                            print(f"Running: {cmd}")
                            os.system(cmd)

else:
    if len(data_types) == 1:
        data_types *= len(datasets)
    elif len(data_types) != len(datasets):
        raise ValueError("Length of --data_type must be 1 or match length of --data")

    # Align lengths: repeat data_type if only one is given
    if len(data_types) == 1:
        data_types *= len(datasets)
    elif len(data_types) != len(datasets):
        raise ValueError("Length of --data_type must be 1 or match length of --data")

    if args.step == 1:
        # === STEP 1: Run the "main" script first for each model-dataset pair ===
        print("Running initial scripts for each model-dataset pair...")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    pre_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}.sh"
                else:
                    pre_script = f"./scripts/long_term_forecast/{data_type}_script/{model}.sh"
                if os.path.isfile(pre_script):
                    print(f"[PRE-RUN] Running initial script: {pre_script}")
                    os.system(f"bash {pre_script}")
                else:
                    print(f"[WARN] Initial script not found: {pre_script}")
    elif args.step == 2:
        # === STEP 2: Train all ECM + pred_length + error_flag combinations ===
        print("Train ECM only.")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}_test.sh"
                else:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_test.sh"
                if not os.path.isfile(bash_script):
                    print(f"Warning: script not found: {bash_script}")
                    continue
                for ecm in ecms:
                    for pred_len in pred_lengths:
                        cmd = f"bash {bash_script} 5 1 {pred_lengths_train} 1 {ecm} {args.season_coef} {args.trend_coef}"
                        print(f"Running: {cmd}")
                        os.system(cmd)
    elif args.step == 3:
        # === STEP 3: Eval all ECM + pred_length + error_flag combinations ===
        print("Eval ECM + pred_length + error_flag combinations.")
        for model in models:
            for dataset, data_type in zip(datasets, data_types):
                if "ETT" in dataset:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_{dataset}_test.sh"
                else:
                    bash_script = f"./scripts/long_term_forecast/{data_type}_script/{model}_test.sh"
                if not os.path.isfile(bash_script):
                    print(f"Warning: script not found: {bash_script}")
                    continue
                for ecm in ecms:
                    for pred_len in pred_lengths:
                        for err_flag in error_model:
                            cmd = f"bash {bash_script} 6 1 {pred_len} {err_flag} {ecm} {args.season_coef} {args.trend_coef}"
                            print(f"Running: {cmd}")
                            os.system(cmd)