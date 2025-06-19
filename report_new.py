import os
import re
from collections import defaultdict

import csv

# Optional: use file_paths directly in your metric extraction script


# def extract_metrics(filepath):
#     with open(filepath, "r") as file:
#         content = file.read()
#         mse = re.search(r"mse[:=]\s*([0-9.]+)", content)
#         mae = re.search(r"mae[:=]\s*([0-9.]+)", content)
#         dtw = re.search(r"dtw[:=]\s*([-\d.]+)", content)
#         return {
#             "mse": float(mse.group(1)) if mse else None,
#             "mae": float(mae.group(1)) if mae else None,
#             "dtw": float(dtw.group(1)) if dtw else None
#         }

# def run(base_dir, base_folder):


#     model_names = ["linear", "logistic", "random_forest", "xgboost", "lstm", "GRU", "CNN", "TF"]
#     coefficients = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
#     horizons = [96, 192, 336, 720]

#     file_paths = []

#     for model in model_names:
#         model_folder = f"{base_folder}-{model}"
#         for coef in coefficients:
#             for length in horizons:
#                 file_name = f"metrics-{length}-{coef}.txt"
#                 path = os.path.join(base_dir, model_folder, file_name)
#                 if os.path.exists(path):  # ✅ Only include existing files
#                     file_paths.append(path)

#     # Print to verify
#     for path in file_paths:
#         print(path)

#         results = []

#         for path in file_paths:
#             filename = os.path.basename(path)  # e.g., metrics-96-0.1.txt
#             parent_folder = os.path.basename(os.path.dirname(path))  # e.g., ...-CNN or ...-linear

#             # Get method (CNN or linear)
#             method_match = re.search(r'-(CNN|linear|logistic|random_forest|xgboost|lstm|GRU|TF)$', parent_folder)
#             method = method_match.group(1) if method_match else "unknown"

#             # Get horizon and coefficient from file name
#             match = re.search(r"metrics-(\d+)-([0-9.]+)\.txt", filename)
#             if match:
#                 horizon = match.group(1)
#                 coef = match.group(2)
#             else:
#                 horizon, coef = "?", "?"

#             metrics = extract_metrics(path)
#             results.append((method, coef, horizon, metrics))

#     grouped = defaultdict(list)
#     for method, coef, horizon, metrics in results:
#         grouped[(method, coef)].append(metrics)

#     print("\n📊 Average metrics per (method, coef):\n")
#     for (method, coef), metrics_list in grouped.items():
#         mse_vals = [m['mse'] for m in metrics_list if m['mse'] is not None]
#         mae_vals = [m['mae'] for m in metrics_list if m['mae'] is not None]
#         dtw_vals = [m['dtw'] for m in metrics_list if m['dtw'] is not None]

#         avg_mse = sum(mse_vals) / len(mse_vals) if mse_vals else None
#         avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else None
#         avg_dtw = sum(dtw_vals) / len(dtw_vals) if dtw_vals else None

#         print(f"Method: {method}, Coef: {coef} → avg_mse: {avg_mse:.4f}, avg_mae: {avg_mae:.4f}, avg_dtw: {avg_dtw:.4f}")

def extract_metrics(filepath):
    with open(filepath, "r") as file:
        content = file.read()
        mse = re.search(r"mse[:=]\s*([0-9.]+)", content)
        mae = re.search(r"mae[:=]\s*([0-9.]+)", content)
        dtw = re.search(r"dtw[:=]\s*([-\d.]+)", content)
        return {
            "mse": float(mse.group(1)) if mse else None,
            "mae": float(mae.group(1)) if mae else None,
            "dtw": float(dtw.group(1)) if dtw else None
        }

def run(base_dir, base_folder):
    model_names = ["linear", "logistic", "random_forest", "xgboost", "lstm", "GRU", "CNN", "TF"]
    coefficients = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    horizons = [96, 192, 336, 720]

    file_paths = []
    results = []

    for model in model_names:
        model_folder = f"{base_folder}-{model}"
        for coef in coefficients:
            for length in horizons:
                file_name = f"metrics-{length}-{coef}.txt"
                path = os.path.join(base_dir, model_folder, file_name)
                if os.path.exists(path):
                    file_paths.append(path)

    for path in file_paths:
        filename = os.path.basename(path)
        parent_folder = os.path.basename(os.path.dirname(path))
        method_match = re.search(r'-(CNN|linear|logistic|random_forest|xgboost|lstm|GRU|TF)$', parent_folder)
        method = method_match.group(1) if method_match else "unknown"
        match = re.search(r"metrics-(\d+)-([0-9.]+)\.txt", filename)
        if match:
            horizon = match.group(1)
            coef = match.group(2)
        else:
            horizon, coef = "?", "?"

        metrics = extract_metrics(path)
        results.append((method, coef, horizon, metrics))

    # Group by method and coef
    grouped = defaultdict(list)
    for method, coef, horizon, metrics in results:
        grouped[(method, coef)].append(metrics)

    # Write to CSV
    csv_path = "average_metrics.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Coef", "Avg_MSE", "Avg_MAE", "Avg_DTW"])

        print("\n📊 Average metrics per (method, coef):\n")
        for (method, coef), metrics_list in grouped.items():
            mse_vals = [m['mse'] for m in metrics_list if m['mse'] is not None]
            mae_vals = [m['mae'] for m in metrics_list if m['mae'] is not None]
            dtw_vals = [m['dtw'] for m in metrics_list if m['dtw'] is not None]

            avg_mse = sum(mse_vals) / len(mse_vals) if mse_vals else None
            avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else None
            avg_dtw = sum(dtw_vals) / len(dtw_vals) if dtw_vals else None

            print(f"Method: {method}, Coef: {coef} → avg_mse: {avg_mse:.4f}, avg_mae: {avg_mae:.4f}, avg_dtw: {avg_dtw:.4f}")
            writer.writerow([method, coef, avg_mse, avg_mae, avg_dtw])

    print(f"\n✅ Saved average metrics to `{csv_path}`.")