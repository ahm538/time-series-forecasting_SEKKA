#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

from src import config
from src.training import train_and_evaluate_for_route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sekka Training Pipeline (per-route Prophet models)")
    p.add_argument(
        "--csv",
        type=str,
        default=str(config.DEFAULT_CSV),
        help="Path to dataset CSV (default: data/sekka_mansoura_synthetic_dataset_2years_v2.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = ["timestamp", "route_id", "congestion_level"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # Ensure timestamp is parseable
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

    route_ids: List[str] = df["route_id"].astype(str).unique().tolist()
    route_ids = sorted(route_ids)

    report_rows = []

    for rid in route_ids:
        sub = df[df["route_id"].astype(str) == rid].copy()
        print(f"Training route {rid} with {len(sub)} rows ...")
        try:
            _, metrics = train_and_evaluate_for_route(sub, rid)
            row = {"route_id": rid, **metrics}
            print(f"-> Done. MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}")
        except Exception as e:
            row = {"route_id": rid, "error": str(e)}
            print(f"-> Failed: {e}")
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(config.TRAINING_REPORT, index=False)
    print(f"Saved training report to {config.TRAINING_REPORT}")


if __name__ == "__main__":
    main()
