import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re


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


def plot_all_methods(base_dir, methods, metric_file, y_col, ylabel, save_name):
    COMMON_SUBPATH = Path("pnp_flow_admm/test")
    plt.figure(figsize=(6, 4))
    plotted_any = False
    for method in methods:
        file_path = base_dir / method / COMMON_SUBPATH / metric_file
        df = read_metric_file(file_path, y_col)
        if df is None or df.empty:
            continue
        plt.plot(
            df["rho_admm"],
            df[y_col],
            marker="o",
            linewidth=2,
            label=method
        )
        plotted_any = True
    if not plotted_any:
        print(f"[WARNING] Nothing was plotted for {metric_file}")
        plt.close()
        return
    plt.xlabel(r"$\rho$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs $\\rho$ for CELEBA dataset")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = base_dir / save_name
    plt.savefig(out_path, dpi=300)
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
    

def get_task_and_method(METHODS, file_path: Path, root_dir: Path):
    rel_parts = file_path.relative_to(root_dir).parts
    if len(rel_parts) < 3:
        return None, None
    task = rel_parts[0]
    method = None
    for p in rel_parts:
        if p in METHODS:
            method = p
            break
    return task, method

def get_experiment_dir(file_path: Path):
    """
    Return the parent directory containing final_*.txt
    """
    return str(file_path.parent)


def collect_results(METHODS, METRIC_FILES,root_dir: Path) -> pd.DataFrame:
    """
    Build one unified dataframe with columns like:
    task, method, noise_type, noise_flag, psnr_rec, ssim_rec, lpips_rec, ...
    """
    records = {}

    for metric_name, metric_filename in METRIC_FILES.items():
        for file_path in root_dir.rglob(metric_filename):
            task, method = get_task_and_method(METHODS,file_path, root_dir)

            if task is None or method is None:
                continue

            noise_col = METHODS[method]
            df = safe_read_table(file_path)

            if df.empty:
                continue

            rec_col = f"{metric_name}_rec"
            noisy_col = f"{metric_name}_noisy"

            if rec_col not in df.columns:
                print(f"Skipping {file_path} because {rec_col} not found.")
                continue

            if noise_col not in df.columns:
                print(f"Skipping {file_path} because {noise_col} not found.")
                continue

            exp_dir = get_experiment_dir(file_path)

            # keep some useful hyperparameter columns if present
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

            for _, row in df.iterrows():
                noise_flag = row[noise_col]

                # Normalize bool-like values
                if isinstance(noise_flag, str):
                    if noise_flag.lower() == "true":
                        noise_flag = True
                    elif noise_flag.lower() == "false":
                        noise_flag = False

                key = (task, method, exp_dir, noise_flag)

                if key not in records:
                    records[key] = {
                        "task": task,
                        "method": method,
                        "experiment_dir": exp_dir,
                        "noise_type": noise_col,
                        "noise_flag": noise_flag,
                    }

                records[key][rec_col] = row[rec_col]

                if noisy_col in df.columns:
                    records[key][noisy_col] = row[noisy_col]

                for c in optional_cols:
                    if c in df.columns:
                        records[key][c] = row[c]

    result_df = pd.DataFrame(records.values())

    if not result_df.empty:
        # nicer ordering
        preferred_cols = [
            "task",
            "method",
            "noise_type",
            "noise_flag",
            "psnr_rec",
            "ssim_rec",
            "lpips_rec",
            "psnr_noisy",
            "ssim_noisy",
            "lpips_noisy",
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
            "experiment_dir",
        ]
        existing_cols = [c for c in preferred_cols if c in result_df.columns]
        other_cols = [c for c in result_df.columns if c not in existing_cols]
        result_df = result_df[existing_cols + other_cols]

        # sort rows
        sort_cols = [c for c in ["task", "method", "noise_flag"] if c in result_df.columns]
        result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    return result_df


def save_outputs(df: pd.DataFrame, csv_path: str, xlsx_path: str):
    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        # also save method-wise sheets
        if "method" in df.columns:
            for method_name, subdf in df.groupby("method"):
                sheet = method_name[:31]  # Excel sheet name limit
                subdf.to_excel(writer, sheet_name=sheet, index=False)

        # also save task-wise sheets
        if "task" in df.columns:
            for task_name, subdf in df.groupby("task"):
                sheet = f"task_{task_name}"[:31]
                subdf.to_excel(writer, sheet_name=sheet, index=False)



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