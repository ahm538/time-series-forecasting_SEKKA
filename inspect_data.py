#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src import config


def main():
    csv_path = config.DEFAULT_CSV
    out_dir = Path("sekka_outputs")
    out_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}. Move your dataset into data/ and retry.")
        return

    # Load
    df = pd.read_csv(csv_path)

    # Basic column checks
    required = ["timestamp", "route_id", "congestion_level"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Parse timestamp and coerce types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Handle missing congestion values
    if df["congestion_level"].isna().any():
        df = df.dropna(subset=["congestion_level"]).copy()

    # Ensure numeric congestion
    df["congestion_level"] = pd.to_numeric(df["congestion_level"], errors="coerce")
    df = df.dropna(subset=["congestion_level"]).copy()

    # Derive hour of day for visualization
    df["hour"] = df["timestamp"].dt.hour

    # Statistical summary
    cong = df["congestion_level"].to_numpy()
    min_v = float(np.min(cong)) if len(cong) else float("nan")
    max_v = float(np.max(cong)) if len(cong) else float("nan")
    mean_v = float(np.mean(cong)) if len(cong) else float("nan")
    median_v = float(np.median(cong)) if len(cong) else float("nan")
    count_gt_5 = int(np.sum(cong > 5.0)) if len(cong) else 0
    count_le_1 = int(np.sum(cong <= 1.0)) if len(cong) else 0

    print("\n=== Sekka Dataset: congestion_level Summary ===")
    print(f"CSV: {csv_path}")
    print("----------------------------------------------")
    print(f"Rows: {len(df):,}")
    print(f"Min:   {min_v:.3f}")
    print(f"Max:   {max_v:.3f}")
    print(f"Mean:  {mean_v:.3f}")
    print(f"Median:{median_v:.3f}")
    print(f"> 5.0: {count_gt_5:,}")
    print(f"<= 1.0: {count_le_1:,}")

    # Visualization
    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution (Histogram)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["congestion_level"], bins=40, kde=True, color="#1f77b4")
    plt.title("Distribution of Congestion Level")
    plt.xlabel("Congestion Level")
    plt.ylabel("Frequency")
    hist_path = out_dir / "congestion_distribution.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # Plot 2: Boxplot by Hour of Day
    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=df,
        x="hour",
        y="congestion_level",
        color="#A1C6EA",
        showfliers=False,
    )
    plt.title("Congestion by Hour of Day (0-23)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Congestion Level")
    box_path = out_dir / "congestion_by_hour_boxplot.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=150)
    plt.close()

    # Plot 3: Sample Route over 1 week (choose random route)
    route_ids = df["route_id"].astype(str).unique().tolist()
    if route_ids:
        sample_route = random.choice(route_ids)
        sub = df[df["route_id"].astype(str) == sample_route].copy()
        sub = sub.sort_values("timestamp")
        if not sub.empty:
            # Use last 7 days available for that route
            last_ts = sub["timestamp"].max()
            start_ts = last_ts - pd.Timedelta(days=7)
            sub_week = sub[(sub["timestamp"] > start_ts) & (sub["timestamp"] <= last_ts)].copy()
            if not sub_week.empty:
                plt.figure(figsize=(12, 4))
                plt.plot(sub_week["timestamp"], sub_week["congestion_level"], color="#ff7f0e")
                plt.title(f"Route {sample_route} · Last 7 Days Congestion")
                plt.xlabel("Timestamp")
                plt.ylabel("Congestion Level")
                plt.xticks(rotation=25)
                route_path = out_dir / f"sample_route_{sample_route}_last7d.png"
                plt.tight_layout()
                plt.savefig(route_path, dpi=150)
                plt.close()
            else:
                print("Note: Sample route has no data in the last 7 days window.")
        else:
            print("Note: Sample route subset is empty.")
    else:
        print("Note: No route_ids found.")

    # Diagnosis
    if np.isfinite(max_v) and max_v <= 1.05:
        print("\n⚠️  DATA IS LIKELY NORMALIZED (0-1). UI SCALING REQUIRED.")

    print("\nSaved figures to:")
    print(f"- {hist_path}")
    print(f"- {box_path}")
    if route_ids:
        print(f"- {route_path if 'route_path' in locals() else '(sample route plot not generated)'}")


if __name__ == "__main__":
    main()
