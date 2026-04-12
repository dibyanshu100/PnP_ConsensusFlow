import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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
    plt.figure(figsize=(7, 4))
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
    plt.title(f"{ylabel} vs $\\rho$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = base_dir / save_name
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")
