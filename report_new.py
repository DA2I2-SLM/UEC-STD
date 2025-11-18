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

    for loss in ["MSE", "MAE", "DTW"]:
        for model in model_names:
            model_folder = f"{base_folder}-{model}"
            for coef in coefficients:
                for length in horizons:
                    file_name = f"metrics-{length}-{coef}.txt"
                    path = os.path.join(base_dir, model_folder, file_name)
                    if os.path.exists(path):
                        file_paths.append(path)
        for model in model_names:
            model_folder = f"{base_folder}-{model}-{loss}"
            for coef in coefficients:
                for length in horizons:
                    file_name = f"metrics-{length}-{coef}.txt"
                    path = os.path.join(base_dir, model_folder, file_name)
                    if os.path.exists(path):
                        file_paths.append(path)
    print(file_paths)
    for path in file_paths:
        filename = os.path.basename(path)
        parent_folder = os.path.basename(os.path.dirname(path))
        method_match = re.search(r'-(CNN|linear|logistic|random_forest|xgboost|lstm|GRU|TF|CNN-MSE|CNN-MAE|linear-MSE|linear-MAE|logistic-MSE|logistic-MAE|random_forest-MSE|random_forest-MAE|xgboost-MSE|xgboost-MAE|lstm-MSE|lstm-MAE|GRU-MSE|GRU-MAE|TF-MSE|TF-MAE)$', parent_folder)
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



def run_each_horizon(base_dir, base_folder, target_horizon):
    import os
    import re
    import csv

    model_names = ["linear", "logistic", "random_forest", "xgboost", "lstm", "GRU", "CNN", "TF"]

    # Store results
    results = []
    backbone_metrics = {}  # Store metrics for target_horizon if coef_val == 0.0

    # Iterate over model folders
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            for model in model_names:
                if d.startswith(base_folder) and model in d:
                    folder_path = os.path.join(root, d)
                    for fname in os.listdir(folder_path):
                        match = re.match(r"metrics-(\d+)-([0-9.]+)\.txt", fname)
                        if not match:
                            continue
                        horizon = int(match.group(1))
                        if horizon != target_horizon:
                            continue  # skip other horizons
                        coef_val = float(match.group(2))
                        file_path = os.path.join(folder_path, fname)

                        # Extract metrics from file (must return dict with 'mse' and 'mae')
                        metrics = extract_metrics(file_path)
                        results.append((model, coef_val, metrics['mse'], metrics['mae']))

    # Write results to CSV and print
    csv_path = f"metrics_horizon_{target_horizon}.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Coef", "MSE", "MAE"])
        print(f"\n📊 Metrics for horizon {target_horizon}:\n")
        print(f"{'Method':<15} {'Coef':<5} {'MSE':<10} {'MAE':<10}")
        print("-" * 50)
        for row in results:
            writer.writerow(row)
            method, coef, mse, mae = row
            print(f"{method:<15} {coef:<5} {mse:<10.3f} {mae:<10.3f}")

    print(f"\n✅ Saved metrics to `{csv_path}`")



def run_each_horizon_best(checkpoint_dir, base_dir, base_folder, target_horizon):
    import os
    import re

    model_names = ["linear", "logistic", "random_forest", "xgboost", "lstm", "GRU", "CNN", "TF"]

    # --- Step 1: Find best coefficients ---
    pattern = re.compile(r"checkpoint-modelerr-(\w+)-found-best-coeff-([\d.]+)-(MAE|MSE)")
    best_coeffs = {}  # model -> {"Best_MAE_Coef": val, "Best_MSE_Coef": val}

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            model_type = match.group(1)
            coeff_value = float(match.group(2))
            metric = match.group(3)

            if model_type not in best_coeffs:
                best_coeffs[model_type] = {}
            best_coeffs[model_type][f"Best_{metric}_Coef"] = coeff_value

    # --- Step 2: Collect best MSE and best MAE separately ---
    best_mse_results = {}  # model -> MSE
    best_mae_results = {}  # model -> MAE
    backbone_metrics = {}

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            for model in model_names:
                if d.startswith(base_folder) and model in d:
                    folder_path = os.path.join(root, d)
                    for fname in os.listdir(folder_path):
                        match = re.match(r"metrics-(\d+)-([0-9.]+)\.txt", fname)
                        if not match:
                            continue
                        horizon = int(match.group(1))
                        coef_val = float(match.group(2))
                        if horizon != target_horizon:
                            continue

                        file_path = os.path.join(folder_path, fname)
                        metrics = extract_metrics(file_path)

                        best_mse_coef = best_coeffs.get(model, {}).get("Best_MSE_Coef", None)
                        best_mae_coef = best_coeffs.get(model, {}).get("Best_MAE_Coef", None)

                        # Backbone
                        if coef_val == 0.0:
                            backbone_metrics[model] = metrics

                        # Only best MSE
                        if coef_val == best_mse_coef:
                            best_mse_results[model] = metrics['mse']

                        # Only best MAE
                        if coef_val == best_mae_coef:
                            best_mae_results[model] = metrics['mae']

    # --- Step 3: Optionally include backbone if no best found ---
    # for model, metrics in backbone_metrics.items():
    #     if model not in best_mse_results:
    #         best_mse_results[model] = metrics['mse']
    #     if model not in best_mae_results:
    #         best_mae_results[model] = metrics['mae']

    # --- Step 4: Print results ---
    print(f"\n📊 Best MSE for horizon {target_horizon}:")
    for model, mse_val in best_mse_results.items():
        print(f"{model:<15}: {mse_val:.4f}")

    print(f"\n📊 Best MAE for horizon {target_horizon}:")
    for model, mae_val in best_mae_results.items():
        print(f"{model:<15}: {mae_val:.4f}")

    return best_mse_results, best_mae_results




def extract_dataset_and_model(folder_name):
    # First extract dataset and full model part (like before)
    match = re.search(r"^long_term_forecast_(\w+)_\d+_\d+_([\w]+)", folder_name, re.IGNORECASE)
    if match:
        dataset = match.group(1)
        full_model = match.group(2)  # e.g. TimesNet_custom_ftM...
        # Extract just the primary model (first segment before '_')
        primary_model = full_model.split('_')[0]
        return dataset, primary_model
    return None, None


def run_new(base_dir, base_folder, season_trend="None_None"):
    import os, re, csv
    from collections import defaultdict

    model_names = ["linear", "logistic", "random_forest", "xgboost", "lstm", "GRU", "CNN", "TF"]
    coefficients = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    horizons = [96, 192, 336, 720]
    losses = ["MSE", "MAE", "DTW"]

    file_paths = []
    results = []

    # Collect all metric file paths
    for loss in losses:
        for model in model_names:
            model_folder_base = f"{base_folder}-{model}"
            for suffix in ["", f"-{loss}"]:
                model_folder = model_folder_base + suffix
                for coef in coefficients:
                    for length in horizons:
                        file_name = f"metrics-{length}-{coef}.txt"
                        path = os.path.join(base_dir, model_folder, file_name)
                        if os.path.exists(path):
                            file_paths.append(path)

    # Parse metrics from files
    for path in file_paths:
        filename = os.path.basename(path)
        parent_folder = os.path.basename(os.path.dirname(path))

        method_match = re.search(
            r'-(CNN|linear|logistic|random_forest|xgboost|lstm|GRU|TF)(?:-(MSE|MAE|DTW))?$', parent_folder
        )
        if method_match:
            method = method_match.group(1)
            loss = method_match.group(2) or "default"
        else:
            method = "unknown"
            loss = "default"

        match = re.search(r"metrics-(\d+)-([0-9.]+)\.txt", filename)
        if match:
            horizon = match.group(1)
            coef = match.group(2)
        else:
            horizon, coef = "?", "?"

        metrics = extract_metrics(path)
        results.append((method, loss, coef, horizon, metrics))

    # Group by method, loss, coef
    grouped = defaultdict(list)
    for method, loss, coef, horizon, metrics in results:
        grouped[(method, loss, coef)].append(metrics)

    # Average per group
    averaged = []
    for (method, loss, coef), metrics_list in grouped.items():
        mse_vals = [m['mse'] for m in metrics_list if m['mse'] is not None]
        mae_vals = [m['mae'] for m in metrics_list if m['mae'] is not None]
        dtw_vals = [m['dtw'] for m in metrics_list if m['dtw'] is not None]

        avg_mse = sum(mse_vals) / len(mse_vals) if mse_vals else None
        avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else None
        avg_dtw = sum(dtw_vals) / len(dtw_vals) if dtw_vals else None

        averaged.append({
            "method": method,
            "loss": loss,
            "coef": coef,
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "avg_dtw": avg_dtw
        })

    # Find best coef per method-loss
    best_rows = []
    for loss in losses:
        for method in model_names:
            candidates = [row for row in averaged if row["method"] == method and row["loss"] == loss]
            if not candidates:
                continue

            # Add one row with coef=0.0 (if exists) for reference
            zero_coef_rows = [row for row in candidates if float(row["coef"]) == 0.0]
            if zero_coef_rows:
                row = zero_coef_rows[0].copy()
                row["method"] = "NA"
                best_rows.append(row)

            # Exclude coef=0.0 when finding the best
            nonzero_candidates = [row for row in candidates if float(row["coef"]) != 0.0]
            if not nonzero_candidates:
                continue

            if loss == "MSE":
                best = min(nonzero_candidates, key=lambda x: x["avg_mse"] if x["avg_mse"] is not None else float("inf"))
            elif loss == "MAE":
                best = min(nonzero_candidates, key=lambda x: x["avg_mae"] if x["avg_mae"] is not None else float("inf"))
            elif loss == "DTW":
                best = min(nonzero_candidates, key=lambda x: x["avg_dtw"] if x["avg_dtw"] is not None else float("inf"))
            else:
                continue

            best_rows.append(best)

    dataset, model = extract_dataset_and_model(base_folder)
    print("Dataset:", dataset)
    print("Model:", model)

    # Group best_rows by method and loss
    from collections import defaultdict as ddict
    grouped_best = ddict(list)
    for row in best_rows:
        grouped_best[(row["method"], row["loss"])].append(row)

    # ➕ Add BEST_MSE and BEST_MAE rows (including coef=0.0)
    all_valid_rows = [r for r in averaged if r["avg_mse"] is not None and r["avg_mae"] is not None]

    best_mse_row = min(all_valid_rows, key=lambda r: r["avg_mse"])
    best_mae_row = min(all_valid_rows, key=lambda r: r["avg_mae"])

    best_mse_row_labeled = best_mse_row.copy()
    best_mse_row_labeled["summary"] = "BEST_MSE"

    best_mae_row_labeled = best_mae_row.copy()
    best_mae_row_labeled["summary"] = "BEST_MAE"

    best_rows.extend([best_mse_row_labeled, best_mae_row_labeled])
    grouped_best[(best_mse_row_labeled["method"], best_mse_row_labeled["loss"])].append(best_mse_row_labeled)
    grouped_best[(best_mae_row_labeled["method"], best_mae_row_labeled["loss"])].append(best_mae_row_labeled)

    # Print results
    print("\n🏆 Best coefficient per (method, loss) with single NA per loss:\n")
    na_printed = set()
    for (method, loss), rows in grouped_best.items():
        zero_coef_rows = [r for r in rows if float(r["coef"]) == 0.0]
        nonzero_coef_rows = [r for r in rows if float(r["coef"]) != 0.0]

        if method == "NA" and loss not in na_printed and zero_coef_rows:
            r = zero_coef_rows[0]
            print(f"NA ({loss}): coef=0 → MSE={r['avg_mse']:.4f}, MAE={r['avg_mae']:.4f}, DTW={r['avg_dtw']:.4f}")
            na_printed.add(loss)

        for r in nonzero_coef_rows:
            label = r.get("summary", "")
            print(f"{r['method']} ({loss}): coef={r['coef']} → MSE={r['avg_mse']:.4f}, MAE={r['avg_mae']:.4f}, DTW={r['avg_dtw']:.4f}" + (f" [{label}]" if label else ""))

    # Write to CSV
    csv_path = f"{season_trend}_{dataset}_{model}_best_metrics.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Loss", "Best_Coef", "Avg_MSE", "Avg_MAE", "Avg_DTW", "Summary"])

        na_written = set()
        for (method, loss), rows in grouped_best.items():
            zero_coef_rows = [r for r in rows if float(r["coef"]) == 0.0]
            nonzero_coef_rows = [r for r in rows if float(r["coef"]) != 0.0]

            if method == "NA" and loss not in na_written and zero_coef_rows:
                r = zero_coef_rows[0]
                writer.writerow([r["method"], r["loss"], r["coef"], r["avg_mse"], r["avg_mae"], r["avg_dtw"], ""])
                na_written.add(loss)

            for r in nonzero_coef_rows:
                if "summary" not in r:  # Exclude summary rows here
                    writer.writerow([
                        r["method"], r["loss"], r["coef"],
                        r["avg_mse"], r["avg_mae"], r["avg_dtw"],
                        ""
                    ])

        # ✅ Now write BEST and SECOND_BEST rows at the end
        def write_summary_row(row, label):
            writer.writerow([
                row["method"], row["loss"], row["coef"],
                row["avg_mse"], row["avg_mae"], row["avg_dtw"],
                label
            ])

        # Already computed best rows
        write_summary_row(best_mse_row, "BEST_MSE")
        write_summary_row(best_mae_row, "BEST_MAE")

        # Filter all valid rows for MSE and MAE separately
        mse_rows = [r for r in all_valid_rows if r["loss"] == "MSE"]
        mae_rows = [r for r in all_valid_rows if r["loss"] == "MAE"]

        # Sort by metric ascending (lower is better)
        sorted_mse = sorted(mse_rows, key=lambda r: r["avg_mse"])
        sorted_mae = sorted(mae_rows, key=lambda r: r["avg_mae"])

        # print(sorted_mse)
        # exit()
        # Find 2nd best MSE: first row NOT matching best method+loss+coef
        second_best_mse_row = None
        for r in sorted_mse[1:]:
            if not (r["method"] == best_mse_row["method"]):
                second_best_mse_row = r
                break

        # Find 2nd best MAE similarly
        second_best_mae_row = None
        for r in sorted_mae[1:]:
            if not (r["method"] == best_mae_row["method"]):
                second_best_mae_row = r
                break

        # Write second best rows if found
        if second_best_mse_row:
            write_summary_row(second_best_mse_row, "SECOND_BEST_MSE")
        if second_best_mae_row:
            write_summary_row(second_best_mae_row, "SECOND_BEST_MAE")



