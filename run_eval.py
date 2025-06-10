import argparse
import os
from distutils.util import strtobool

def parse_arg_list(s):
    return [eval(x.strip()) for x in s.split(',') if x.strip()]

def str_to_bool(value):
    return bool(strtobool(value))
    
parser = argparse.ArgumentParser()
parser.add_argument('--pred_lengths', type=str, required=False,
help='Comma-separated prediction lengths', default="96,192,336,720")
parser.add_argument('--error_flags', type=str, required=False,
help='Comma-separated error model flags (0, 0.5, 1)', default="0,0.1,0.3,0.5,0.7,1")
parser.add_argument('--data', type=str, required=True, default="ETTh1")
parser.add_argument('--model', type=str, required=True, default="TimeMixer")
parser.add_argument('--ecm', type=str, required=True, default="linear")
parser.add_argument('--include_x0', type=str, required=False, default="True")

args = parser.parse_args()

pred_lengths = parse_arg_list(args.pred_lengths)
error_model = parse_arg_list(args.error_flags)

for j in pred_lengths:
    for i in error_model:
        if "ETT" in args.data:
            cmd = f"bash ./scripts/long_term_forecast/ETT_script/{args.model}_{args.data}_test.sh 4 1 {j} {i} {args.ecm}"
        else:
            cmd = f"bash ./scripts/long_term_forecast/{args.data}_script/{args.model}_test.sh 4 1 {j} {i} {args.ecm}"
 
        print(f"Running: {cmd}")
        os.system(cmd)