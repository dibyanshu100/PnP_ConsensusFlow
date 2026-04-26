import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def read_metric_file(file_path, y_col):
    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        return None
    df = pd.read_csv(file_path, sep=r"\s+")
    if "rho_admm" not in df.columns or y_col not in df.columns:
        print(f"[WARNING] Missing required columns in {file_path}")
        print("Columns found:", list(df.columns))
        return None

    df["rho_admm"] = pd.to_numeric(df["rho_admm"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["rho_admm", y_col]).sort_values("rho_admm")

    return df


def plot_rho_analysis(
    base_dirs,
    dataset_names,
    methods,
    metric_config,
    save_name="all_datasets_metrics_vs_rho_admm.png"
):
    COMMON_SUBPATH = Path("pnp_flow_admm/test")
    METHOD_LABELS = {
    "denoising": "Denoising",
    "gaussian_deblurring_FFT": "Deblurring",
    "inpainting": "Box Inpainting",
    "paintbrush_inpainting": "Paintbrush Inpainting",
    "random_inpainting": "Random Inpainting",
    "superresolution": "Superresolution",
}

    # Paper-friendly global font sizes
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 22,
    })

    n_rows = len(base_dirs)
    n_cols = len(metric_config)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        sharex=True
    )

    if n_rows == 1:
        axes = [axes]

    all_handles = {}
    plotted_any = False

    for row_idx, (base_dir, dataset_name) in enumerate(zip(base_dirs, dataset_names)):
        for col_idx, (metric_file, (y_col, ylabel)) in enumerate(metric_config.items()):
            ax = axes[row_idx][col_idx]

            for method in methods:
                file_path = base_dir / method / COMMON_SUBPATH / metric_file
                df = read_metric_file(file_path, y_col)

                if df is None or df.empty:
                    continue

                pretty_label = METHOD_LABELS.get(method, method)

                line, = ax.plot(
                    df["rho_admm"],
                    df[y_col],
                    marker="o",
                    linewidth=2.8,
                    markersize=7,
                    label=pretty_label
                )

                all_handles[pretty_label] = line
                plotted_any = True

            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset_name} — {ylabel}", pad=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both")

    if not plotted_any:
        print("[WARNING] Nothing was plotted")
        plt.close()
        return

    fig.legend(
        all_handles.values(),
        all_handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.09, 1, 0.98])

    out_path = Path(save_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")

## For abalations

def safe_read_table(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        return df
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return pd.DataFrame()


def get_experiment_dir(file_path: Path):
    """
    Return the parent directory containing final_*.txt
    """
    return str(file_path.parent)


def normalize_bool(value):
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def get_task_and_method(methods, file_path: Path, root_dir: Path):
    rel_parts = file_path.relative_to(root_dir).parts
    if len(rel_parts) < 3:
        return None, None

    task = rel_parts[0]
    method = None
    for p in rel_parts:
        if p in methods:
            method = p
            break

    return task, method


def collect_results(methods, metric_files, root_dir: Path) -> pd.DataFrame:
    """
    Build one unified dataframe with columns like:
    task, method, adaptive_noise, deterministic_noise, psnr_rec, ssim_rec, lpips_rec, ...
    """
    records = {}

    optional_cols = [
        "steps_pnp",
        "lr_pnp",
        "gamma_style",
        "num_samples",
        "alpha",
        "rho_admm",
        "max_cgiter",
        "cg_tol",
        "flow_step",
        "save_norms_admm",
    ]

    config_cols = [
        "adaptive_noise",
        "deterministic_noise",
    ]

    for metric_name, metric_filename in metric_files.items():
        for file_path in root_dir.rglob(metric_filename):
            task, method = get_task_and_method(methods, file_path, root_dir)

            if task is None or method is None:
                continue

            df = safe_read_table(file_path)
            if df.empty:
                continue

            rec_col = f"{metric_name}_rec"
            noisy_col = f"{metric_name}_noisy"

            if rec_col not in df.columns:
                print(f"Skipping {file_path} because {rec_col} not found.")
                continue

            exp_dir = get_experiment_dir(file_path)

            for _, row in df.iterrows():
                adaptive_noise = normalize_bool(row["adaptive_noise"]) if "adaptive_noise" in df.columns else None
                deterministic_noise = normalize_bool(row["deterministic_noise"]) if "deterministic_noise" in df.columns else None

                key = (
                    task,
                    method,
                    exp_dir,
                    adaptive_noise,
                    deterministic_noise,
                )

                if key not in records:
                    records[key] = {
                        "task": task,
                        "method": method,
                        "experiment_dir": exp_dir,
                        "adaptive_noise": adaptive_noise,
                        "deterministic_noise": deterministic_noise,
                    }

                records[key][rec_col] = row[rec_col]

                if noisy_col in df.columns:
                    records[key][noisy_col] = row[noisy_col]

                for c in optional_cols + config_cols:
                    if c in df.columns:
                        records[key][c] = normalize_bool(row[c])

    result_df = pd.DataFrame(records.values())

    if not result_df.empty:
        final_cols = ["task","method","adaptive_noise","deterministic_noise","psnr_rec","ssim_rec","lpips_rec","psnr_noisy","ssim_noisy","lpips_noisy"]

        # Keep only existing columns (safe)
        existing_cols = [c for c in final_cols if c in result_df.columns]
        result_df = result_df[existing_cols]

        sort_cols = [
            c for c in ["task", "method", "adaptive_noise", "deterministic_noise"]
            if c in result_df.columns
        ]

        result_df = result_df.sort_values(sort_cols).reset_index(drop=True)
    return result_df

def save_outputs(df: pd.DataFrame, xlsx_path: str):
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        # # also save method-wise sheets
        # if "method" in df.columns:
        #     for method_name, subdf in df.groupby("method"):
        #         sheet = method_name[:31]  # Excel sheet name limit
        #         subdf.to_excel(writer, sheet_name=sheet, index=False)

        # # also save task-wise sheets
        # if "task" in df.columns:
        #     for task_name, subdf in df.groupby("task"):
        #         sheet = f"task_{task_name}"[:31]
        #         subdf.to_excel(writer, sheet_name=sheet, index=False)



def extract_basic_metrics(root_dir: Path, out_xlsx: str = "metrics_summary.xlsx") -> pd.DataFrame:
    rows = []

    for file_path in root_dir.rglob("final_*.txt"):
        rel = file_path.relative_to(root_dir).parts

        # expected: task / method / test / ... / final_xxx.txt
        if len(rel) < 3:
            continue

        task = rel[0]
        method = rel[1]

        try:
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

        if df.empty:
            continue

        row = {
            "task": task,
            "method": method,
        }

        cols_to_keep = [
            "psnr_rec", "psnr_noisy",
            "ssim_rec", "ssim_noisy",
            "lpips_rec", "lpips_noisy",
        ]

        for col in cols_to_keep:
            if col in df.columns:
                row[col] = df.iloc[0][col]

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = (
            result.groupby(["task", "method"], as_index=False)
            .first()
            .sort_values(["task", "method"])
            .reset_index(drop=True)
        )
        result.to_excel(out_xlsx, index=False)

    return result


#####################################################################################################
# convergence analysis

METHOD_LABELS = {
    "denoising": "Denoising",
    "gaussian_deblurring_FFT": "Deblurring",
    "inpainting": "Box Inpainting",
    "paintbrush_inpainting": "Paintbrush Inpainting",
    "random_inpainting": "Random Inpainting",
    "superresolution": "Superresolution",
}


def collect_norms_across_methods(root_dir: Path, batch: int = 0):
    norm_file_name = f"norms_batch_{batch}.npz"
    norm_files = list(root_dir.rglob(norm_file_name))

    if not norm_files:
        print(f"[WARNING] No {norm_file_name} files found under {root_dir}")
        return {"primal_res": {}, "dual_res": {}, "u_inc": {}}

    collected = {
        "primal_res": {},
        "dual_res": {},
        "u_inc": {},
    }

    for file_path in norm_files:
        rel_parts = file_path.relative_to(root_dir).parts

        if len(rel_parts) < 2:
            print(f"[WARNING] Could not parse task/method from {file_path}")
            continue

        task = rel_parts[0]
        method_label = METHOD_LABELS.get(task, task)

        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"[WARNING] Could not load {file_path}: {e}")
            continue

        required_keys = ["primal_res", "dual_res", "u_inc"]
        missing = [k for k in required_keys if k not in data]

        if missing:
            print(f"[WARNING] Missing keys {missing} in {file_path}")
            continue

        collected["primal_res"][method_label] = data["primal_res"]
        collected["dual_res"][method_label] = data["dual_res"]
        collected["u_inc"][method_label] = data["u_inc"]

    return collected


def plot_convergence_two_datasets_shared_legend(
    root_dirs,
    dataset_names,
    batch: int = 0,
    save_path: Path = None,
    logy: bool = True,
):
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    })

    norm_config = {
        "primal_res": (
            "Primal Residual",
            r"$\|x^k - z^k\|_2$",
        ),
        "dual_res": (
            "Dual Residual",
            r"$\rho\|z^k - z^{k-1}\|_2$",
        ),
    }

    n_rows = len(root_dirs)
    n_cols = len(norm_config)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 5.5 * n_rows),
        sharex=False,
    )

    if n_rows == 1:
        axes = [axes]

    all_handles = {}
    plotted_any = False

    for row_idx, (root_dir, dataset_name) in enumerate(zip(root_dirs, dataset_names)):
        norms_dict = collect_norms_across_methods(root_dir=root_dir, batch=batch)

        for col_idx, (norm_key, (norm_title, ylabel)) in enumerate(norm_config.items()):
            ax = axes[row_idx][col_idx]

            curves = norms_dict.get(norm_key, {})

            for method_name, values in sorted(curves.items()):
                line, = ax.plot(
                    values,
                    linewidth=2.8,
                    label=method_name,
                )

                all_handles[method_name] = line
                plotted_any = True

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset_name} — {norm_title}", pad=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both")

            if logy:
                ax.set_yscale("log")

    if not plotted_any:
        print("[WARNING] Nothing was plotted")
        plt.close()
        return

    fig.legend(
        all_handles.values(),
        all_handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.98])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()



#############################################################################################################################
# time and memory

def read_average_value(file_path: Path):
    if not file_path.exists():
        return None

    try:
        text = file_path.read_text().strip()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    if not text:
        return None

    match = re.search(r":\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if match:
        return float(match.group(1))

    return None


def extract_time_memory_summary(root_dir: Path) -> pd.DataFrame:
    """
    Extract:
      - time from time_average.txt
      - memory from max_memory_average.txt

    Expected folder structure:
      root_dir / task / method / model / ... / time_average.txt
    """
    rows = []

    for time_file in root_dir.rglob("time_average.txt"):
        exp_dir = time_file.parent
        mem_file = exp_dir / "max_memory_average.txt"

        rel_parts = time_file.relative_to(root_dir).parts
        if len(rel_parts) < 3:
            continue

        task = rel_parts[0]
        method = rel_parts[1]
        model = rel_parts[2]

        time_avg = read_average_value(time_file)
        max_memory_avg = read_average_value(mem_file)
        if max_memory_avg is not None:
            max_memory_avg = max_memory_avg / (1024 ** 3)

        rows.append({
            "task": task,
            "method": method,
            "model": model,
            "time_average": time_avg,
            "max_memory_average_gb": max_memory_avg,
            "experiment_dir": str(exp_dir),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["task", "method", "model"]).reset_index(drop=True)

    return df


def save_time_memory_summary(df: pd.DataFrame, xlsx_path: str, csv_path: str = None):
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        if "task" in df.columns:
            for task_name, subdf in df.groupby("task"):
                subdf.to_excel(writer, sheet_name=f"task_{task_name}"[:31], index=False)

        if "method" in df.columns:
            for method_name, subdf in df.groupby("method"):
                subdf.to_excel(writer, sheet_name=f"method_{method_name}"[:31], index=False)