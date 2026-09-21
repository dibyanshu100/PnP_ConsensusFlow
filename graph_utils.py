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
    COMMON_SUBPATH = Path("pnp_cf/test")
    METHOD_LABELS = {
    "denoising": "Denoising",
    "gaussian_deblurring_FFT": "Deblurring",
    "inpainting": "Box Inpainting",
    "paintbrush_inpainting": "Paintbrush Inpainting",
    "random_inpainting": "Random Inpainting",
    "superresolution": "Superresolution",
}
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
    cols_to_keep = ["psnr_rec", "psnr_noisy", "ssim_rec", "ssim_noisy", "lpips_rec", "lpips_noisy"]

    def to_bool(x):
        return x if isinstance(x, bool) else str(x).strip().lower() == "true"

    for file_path in root_dir.rglob("final_*.txt"):
        rel = file_path.relative_to(root_dir).parts
        if len(rel) < 3:
            continue

        task, method = rel[0], rel[1]

        try:
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

        if df.empty:
            continue

        if method == "pnp_cf":
            for _, r in df.iterrows():
                adaptive = to_bool(r.get("adaptive_noise", False))
                deterministic = to_bool(r.get("deterministic_noise", False))

                if adaptive:
                    method_name = "ADMM adaptive"
                elif deterministic:
                    method_name = "ADMM deterministic"
                else:
                    method_name = "ADMM stochastic"

                row = {"task": task, "method": method_name}
                row.update({col: r[col] for col in cols_to_keep if col in df.columns})
                rows.append(row)

        else:
            r = df.iloc[0]
            row = {"task": task, "method": method}
            row.update({col: r[col] for col in cols_to_keep if col in df.columns})
            rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.groupby(["task", "method"], as_index=False).first()
        result = result.sort_values(["task", "method"]).reset_index(drop=True)
        result.to_excel(out_xlsx, index=False)

    return result


def extract_basic_metrics_variable_params(
    root_dir: Path,
    out_xlsx: str = "metrics.xlsx"
) -> pd.DataFrame:

    metric_files = {
        "psnr": "final_psnr.txt",
        "ssim": "final_ssim.txt",
        "lpips": "final_lpips.txt",
    }

    metric_columns = {
        "psnr": ["psnr_rec", "psnr_noisy"],
        "ssim": ["ssim_rec", "ssim_noisy"],
        "lpips": ["lpips_rec", "lpips_noisy"],
    }

    all_results = []

    # Find directories containing final metric files
    run_dirs = sorted({p.parent for p in root_dir.rglob("final_*.txt")})

    for run_dir in run_dirs:
        rel = run_dir.relative_to(root_dir).parts

        if len(rel) < 2:
            continue

        merged = None

        for metric_name, filename in metric_files.items():
            file_path = run_dir / filename

            if not file_path.exists():
                continue

            try:
                df = pd.read_csv(file_path, sep=r"\s+", engine="python")
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

            if df.empty:
                continue

            available_metrics = [
                col for col in metric_columns[metric_name]
                if col in df.columns
            ]

            # All remaining columns describe the experiment configuration
            parameter_cols = [
                col for col in df.columns
                if col not in metric_columns[metric_name]
            ]

            df = df[parameter_cols + available_metrics]

            if merged is None:
                merged = df
            else:
                common_params = [
                    col for col in parameter_cols
                    if col in merged.columns
                ]

                if common_params:
                    merged = pd.merge(
                        merged,
                        df,
                        on=common_params,
                        how="outer"
                    )
                else:
                    # Fallback: assume corresponding rows represent same runs
                    merged = pd.concat(
                        [
                            merged.reset_index(drop=True),
                            df[available_metrics].reset_index(drop=True),
                        ],
                        axis=1,
                    )

        if merged is None or merged.empty:
            continue

        merged.insert(0, "method", rel[1])
        merged.insert(0, "task", rel[0])

        all_results.append(merged)

    if not all_results:
        print(f"No metric files found under: {root_dir}")
        return pd.DataFrame()

    result = pd.concat(all_results, ignore_index=True)

    preferred_order = [
        "task",
        "method",
        "steps_pnp",
        "rho_admm",
        "max_cgiter",
        "cg_tol",
        "save_norms_admm",
        "adaptive_noise",
        "deterministic_noise",
        "t_schedule_gamma",
        "psnr_rec",
        "psnr_noisy",
        "ssim_rec",
        "ssim_noisy",
        "lpips_rec",
        "lpips_noisy",
    ]

    ordered = [col for col in preferred_order if col in result.columns]
    remaining = [col for col in result.columns if col not in ordered]
    result = result[ordered + remaining]

    sort_cols = [
        col for col in [
            "task",
            "method",
            "steps_pnp",
            "rho_admm",
            "t_schedule_gamma",
        ]
        if col in result.columns
    ]

    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)

    result.to_excel(out_xlsx, index=False)

    print(f"Found {len(result)} experiment configurations.")
    print(f"Saved results to: {out_xlsx}")

    return result

#####################################################################################################


def extract_rebuttal_metrics(
    root_dir: Path,
    out_xlsx: str = "rebuttal_multiseed_results.xlsx"
) -> pd.DataFrame:

    metric_files = {
        "psnr": ("final_psnr.txt", "psnr_rec"),
        "ssim": ("final_ssim.txt", "ssim_rec"),
        "lpips": ("final_lpips.txt", "lpips_rec"),
    }

    results = []

    run_dirs = sorted({
        p.parent
        for p in root_dir.rglob("final_*.txt")
        if p.parent.name == "test"
    })

    for run_dir in run_dirs:
        rel = run_dir.relative_to(root_dir).parts

        if len(rel) < 3:
            continue

        task = rel[-3]
        method = rel[-2]

        summary = {
            "task": task,
            "method": method,
        }

        n_runs = None

        for metric, (filename, column) in metric_files.items():
            file_path = run_dir / filename

            if not file_path.exists():
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue

            try:
                df = pd.read_csv(file_path, sep=r"\s+", engine="python")
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue

            if column not in df.columns:
                print(f"Missing column '{column}' in {file_path}")
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue

            values = pd.to_numeric(df[column], errors="coerce").dropna()

            summary[f"{metric}_mean"] = values.mean()
            summary[f"{metric}_std"] = values.std(ddof=1)

            if n_runs is None:
                n_runs = len(values)

        summary["n_runs"] = n_runs
        results.append(summary)

    if not results:
        print(f"No valid metric files found under: {root_dir}")
        return pd.DataFrame()

    result = pd.DataFrame(results)

    result = result[
        [
            "task",
            "method",
            "n_runs",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "lpips_mean",
            "lpips_std",
        ]
    ]

    result = result.sort_values(
        ["task", "method"]
    ).reset_index(drop=True)

    result.to_excel(out_xlsx, index=False)

    print(f"Saved rebuttal metrics to: {out_xlsx}")
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
    "phase_retrieval": "Phase Retrieval",
    "hdr_reconstruction": "HDR Reconstruction",
    "nonlinear_deblurring": "Nonlinear Deblurring",
    "segmentation": "Segmentation",
}


def collect_norms_across_methods(root_dir: Path, batch: int = 0):
    norm_file_name = f"norms_batch_{batch}.npz"
    norm_files = list(root_dir.rglob(norm_file_name))

    if not norm_files:
        print(f"[WARNING] No {norm_file_name} files found under {root_dir}")
        return {"primal_res": {}, "dual_res": {}, "u_inc": {}, "z_inc": {}}

    collected = {
        "primal_res": {},
        "dual_res": {},
        "u_inc": {},
        "z_inc": {},
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

        if "z_inc" in data:
            collected["z_inc"][method_label] = data["z_inc"]

    return collected


def plot_convergence_two_datasets_shared_legend(
    root_dirs,
    dataset_names,
    batch: int = 0,
    save_path: Path = None,
    logy: bool = True,
    metrics=("primal_res", "z_inc"),
):
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
    })

    all_norm_config = {
        "primal_res": (
            "Primal Residual",
            r"$\|x^k - z^k\|_2$",
        ),
        "dual_res": (
            "Dual Residual",
            r"$\rho\|z^k - z^{k-1}\|_2$",
        ),
        "z_inc": (
            "Dual Residual (unscaled)",
            r"$\|z^k - z^{k-1}\|_2$",
        ),
    }
    norm_config = {key: all_norm_config[key] for key in metrics}

    if len(root_dirs) != len(dataset_names):
        raise ValueError(
            f"root_dirs and dataset_names must have the same length, "
            f"got {len(root_dirs)} and {len(dataset_names)}"
        )

    dataset_pairs = list(zip(root_dirs, dataset_names))

    # Rows = metrics, columns = datasets, so datasets lay out left-to-right
    # in a single row when only one metric is requested.
    n_rows = len(norm_config)
    n_cols = len(dataset_pairs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.5 * n_cols, 5.5 * n_rows),
        sharex=False,
        squeeze=False,
    )

    all_handles = {}
    plotted_any = False

    norms_by_dataset = {
        dataset_name: collect_norms_across_methods(root_dir=root_dir, batch=batch)
        for root_dir, dataset_name in dataset_pairs
    }

    for row_idx, (norm_key, (norm_title, ylabel)) in enumerate(norm_config.items()):
        for col_idx, (_, dataset_name) in enumerate(dataset_pairs):
            ax = axes[row_idx][col_idx]

            curves = norms_by_dataset[dataset_name].get(norm_key, {})

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
    rows = []

    for time_file in root_dir.rglob("time_average.txt"):
        exp_dir = time_file.parent
        mem_file = exp_dir / "max_memory_average.txt"
        rel = time_file.relative_to(root_dir).parts

        if len(rel) < 3:
            continue

        task, method, model = rel[0], rel[1], rel[2]

        if method == "pnp_cf":
            path = str(exp_dir).lower()
            if "adaptive_noise=true" in path:
                method = "ADMM adaptive"
            elif "adaptive_noise=false" in path and "deterministic_noise=true" in path:
                method = "ADMM deterministic"
            elif "adaptive_noise=false" in path and "deterministic_noise=false" in path:
                method = "ADMM stochastic"

        time_avg = read_average_value(time_file)
        memory_avg = read_average_value(mem_file)
        if memory_avg is not None:
            memory_avg /= 1024 ** 3

        rows.append({
            "task": task, "method": method, "model": model,
            "time_average": time_avg, "max_memory_average_gb": memory_avg,
            "experiment_dir": str(exp_dir)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["task", "method", "model"]).reset_index(drop=True)

    return df


#############################################################################################################################
def _read_psnr_vs_steps(file_path: Path, y_col: str = "psnr_rec") -> pd.DataFrame:
    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        return None

    df = safe_read_table(file_path)
    if df.empty or "steps_pnp" not in df.columns or y_col not in df.columns:
        print(f"[WARNING] Missing required columns in {file_path}")
        return None

    df["steps_pnp"] = pd.to_numeric(df["steps_pnp"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    if "adaptive_noise" in df.columns:
        df["adaptive_noise"] = df["adaptive_noise"].apply(normalize_bool)

    df = df.dropna(subset=["steps_pnp", y_col]).sort_values("steps_pnp")
    return df


METRIC_INFO = {
    "psnr": ("final_psnr.txt", "psnr_rec", "PSNR"),
    "ssim": ("final_ssim.txt", "ssim_rec", "SSIM"),
    "lpips": ("final_lpips.txt", "lpips_rec", "LPIPS"),
}


def plot_fbs_vs_admm_psnr_vs_steps(
    base_dir,
    dataset_name,
    tasks,
    metric: str = "psnr",
    save_name: str = None,
):
    """
    For each task, plot `metric` vs steps_pnp for three series:
      - PnP-FBS with adaptive_noise=True
      - PnP-FBS with adaptive_noise=False
      - PnP-ADMM (ours)

    `metric` is one of "psnr", "ssim", "lpips" and determines which
    final_<metric>.txt file is read from base_dir/<task>/pnp_flow_fbs/test
    and base_dir/<task>/pnp_cf/test.
    """
    if metric not in METRIC_INFO:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRIC_INFO)}.")

    metric_filename, y_col, ylabel = METRIC_INFO[metric]

    if save_name is None:
        save_name = f"fbs_vs_admm_{metric}_vs_steps.png"

    base_dir = Path(base_dir)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 22,
    })

    series_style = {
        "PnP-FBS (adaptive)": dict(color="tab:blue", marker="o"),
        "PnP-FBS (non-adaptive)": dict(color="tab:orange", marker="s"),
        "PnP-ADMM (ours)": dict(color="tab:green", marker="^"),
    }

    n_cols = len(tasks)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5.5), squeeze=False)
    axes = axes[0]

    all_handles = {}
    plotted_any = False

    for col_idx, task in enumerate(tasks):
        ax = axes[col_idx]
        task_dir = base_dir / task

        fbs_df = _read_psnr_vs_steps(task_dir / "pnp_flow_fbs" / "test" / metric_filename, y_col)
        admm_df = _read_psnr_vs_steps(task_dir / "pnp_cf" / "test" / metric_filename, y_col)

        if fbs_df is not None and "adaptive_noise" in fbs_df.columns:
            for adaptive_value, label in [(True, "PnP-FBS (adaptive)"), (False, "PnP-FBS (non-adaptive)")]:
                sub_df = fbs_df[fbs_df["adaptive_noise"] == adaptive_value]
                if sub_df.empty:
                    continue
                style = series_style[label]
                line, = ax.plot(
                    sub_df["steps_pnp"],
                    sub_df[y_col],
                    linewidth=2.8,
                    markersize=7,
                    label=label,
                    **style,
                )
                all_handles[label] = line
                plotted_any = True

        if admm_df is not None and not admm_df.empty:
            label = "PnP-ADMM (ours)"
            style = series_style[label]
            line, = ax.plot(
                admm_df["steps_pnp"],
                admm_df[y_col],
                linewidth=2.8,
                markersize=7,
                label=label,
                **style,
            )
            all_handles[label] = line
            plotted_any = True

        ax.set_xlabel("Number of steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset_name} — {METHOD_LABELS.get(task, task)}", pad=12)
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
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.98])

    out_path = Path(save_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")


#############################################################################################################################
# PnP-FBS vs PnP-ADMM: per-iteration metric convergence

CONVERGENCE_METRIC_INFO = {
    "psnr": ("psnr_rec_average.txt", "PSNR"),
    "ssim": ("ssim_rec_average.txt", "SSIM"),
    "lpips": ("lpips_rec_average.txt", "LPIPS"),
}


def _resolve_average_file(path, filename: str) -> Path:
    path = Path(path)
    if path.suffix == ".txt":
        return path

    direct = path / filename
    if direct.exists():
        return direct

    matches = sorted(path.rglob(filename))
    if not matches:
        return direct
    if len(matches) > 1:
        print(
            f"[WARNING] Multiple {filename} found under {path}, using {matches[0]}. "
            f"Pass a more specific directory to disambiguate. All matches: {matches}"
        )
    return matches[0]


def _read_metric_convergence(path, filename: str) -> pd.DataFrame:
    file_path = _resolve_average_file(path, filename)

    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["iteration", "value"])
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["iteration", "value"]).sort_values("iteration")

    return df


def plot_fbs_vs_admm_psnr_convergence(
    runs,
    metric: str = "psnr",
    metrics=None,
    save_name: str = None,
):

    stacked = metrics is not None
    metrics_to_plot = metrics if stacked else [metric]

    for m in metrics_to_plot:
        if m not in CONVERGENCE_METRIC_INFO:
            raise ValueError(f"Unknown metric '{m}'. Choose from {list(CONVERGENCE_METRIC_INFO)}.")

    if save_name is None:
        save_name = f"fbs_vs_admm_{'_'.join(metrics_to_plot)}_convergence.png"

    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.titlesize": 22,
    })

    series_style = {
        "without dual": dict(color="tab:blue", marker="o"),
        "with dual": dict(color="tab:green", marker="^"),
    }

    n_rows = len(metrics_to_plot) if stacked else 1
    n_cols = len(runs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(9 * n_cols, 5.5 * n_rows),
        sharex="col" if stacked else False,
        squeeze=False,
    )

    plotted_any = False

    def draw(ax, run, m):
        nonlocal plotted_any
        filename, ylabel = CONVERGENCE_METRIC_INFO[m]

        fbs_df = _read_metric_convergence(run["fbs_dir"], filename)
        admm_df = _read_metric_convergence(run["admm_dir"], filename)

        for df, label in [(fbs_df, "without dual"), (admm_df, "with dual")]:
            if df is None or df.empty:
                continue
            style = series_style[label]
            ax.plot(
                df["iteration"],
                df["value"],
                linewidth=2.8,
                markersize=7,
                label=label,
                **style,
            )
            plotted_any = True

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both")

        legend_loc = "upper right" if m == "lpips" else "lower right"
        ax.legend(loc=legend_loc, frameon=True)

    if stacked:
        for row_idx, m in enumerate(metrics_to_plot):
            for col_idx, run in enumerate(runs):
                ax = axes[row_idx][col_idx]
                draw(ax, run, m)
                if row_idx == 0:
                    ax.set_title(run.get("title", ""), pad=12)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Iteration")
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
    else:
        for col_idx, run in enumerate(runs):
            ax = axes[0][col_idx]
            draw(ax, run, metric)
            ax.set_xlabel("Iteration")
            ax.set_title(f"{run.get('title', '')} — {metric.upper()}", pad=12)

    if not plotted_any:
        print("[WARNING] Nothing was plotted")
        plt.close()
        return

    plt.tight_layout()

    out_path = Path(save_name)
    # plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")


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