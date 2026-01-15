#!/usr/bin/env python3
from __future__ import annotations

import os
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from src import config
from src.inference import predict_future, interpret_congestion, load_model_and_meta


def list_trained_route_ids() -> List[str]:
    ids: List[str] = []
    if not config.MODELS_DIR.exists():
        return ids
    for name in os.listdir(config.MODELS_DIR):
        if name.startswith("model_route_") and name.endswith(".joblib"):
            rid = name[len("model_route_") : -len(".joblib")]
            ids.append(rid)
    return sorted(ids)


def build_route_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return two dicts: readable->route_id and route_id->readable.
    Reads the main CSV at startup.
    """
    readable_to_id: Dict[str, str] = {}
    id_to_readable: Dict[str, str] = {}
    if not config.DEFAULT_CSV.exists():
        return readable_to_id, id_to_readable

    df = pd.read_csv(config.DEFAULT_CSV)
    # Expected columns: route_id, target_governorate_en, service_type
    cols_needed = {"route_id", "target_governorate_en", "service_type"}
    if not cols_needed.issubset(df.columns):
        # Fallback: just route_id
        route_ids = df.get("route_id", pd.Series(dtype=str)).astype(str).unique().tolist()
        for rid in route_ids:
            label = f"[ID: {rid}]"
            readable_to_id[label] = rid
            id_to_readable[rid] = label
        return readable_to_id, id_to_readable

    # Build a canonical readable label per route_id from first occurrence
    df = df[["route_id", "target_governorate_en", "service_type"]].copy()
    df["route_id"] = df["route_id"].astype(str)
    df = df.drop_duplicates(subset=["route_id"]).reset_index(drop=True)
    for row in df.itertuples(index=False):
        rid = str(row.route_id)
        dest = str(row.target_governorate_en)
        service = str(row.service_type)
        # Mansoura is the origin per project context
        label = f"Mansoura -> {dest} ({service}) - [ID: {rid}]"
        readable_to_id[label] = rid
        id_to_readable[rid] = label
    return readable_to_id, id_to_readable


essential_note = (
    f"Models folder: {config.MODELS_DIR}. Train models first using 'python train_pipeline.py' before using the demo."
)


def make_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["ds"], df["ds"][::-1]]),
            y=pd.concat([df["yhat_upper"], df["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Uncertainty",
        )
    )
    fig.update_layout(
        title="Sekka Congestion Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Congestion Level (0-10)",
        yaxis=dict(range=[config.Y_MIN, config.Y_MAX]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def color_for_status(status: str) -> str:
    if status.startswith("Green"):
        return "#2ca02c"
    if status.startswith("Yellow"):
        return "#ffb000"
    if status.startswith("Orange"):
        return "#ff7f0e"
    if status.startswith("Red"):
        return "#d62728"
    return "#333333"

def predict_ui(readable_choice: str, selected_date, start_hour: int, end_hour: int, readable_to_id: Dict[str, str]):
    if not readable_choice:
        raise gr.Error("Please select a route. If no routes are listed, run training first.")
    if start_hour is None or end_hour is None or selected_date is None:
        raise gr.Error("Please provide date and time window.")
    if end_hour < start_hour:
        raise gr.Error("End hour must be greater than or equal to start hour.")

    # Map UI label to technical route_id
    route_id = readable_to_id.get(readable_choice)
    if not route_id:
        raise gr.Error("Invalid route selection.")

    # Load model metadata to get last_ds
    model, meta = load_model_and_meta(route_id)
    if model is None or meta is None:
        raise gr.Error(f"Model not found for route {route_id}. Train first.")

    last_ds = pd.Timestamp(meta["last_ds"])  # timezone-naive consistent with training

    # User-selected end datetime
    # selected_date from gr.Date is like 'YYYY-MM-DD' or date object
    if isinstance(selected_date, str):
        sel_date = pd.to_datetime(selected_date).date()
    else:
        sel_date = pd.to_datetime(selected_date).date()
    end_dt = pd.Timestamp(sel_date) + pd.Timedelta(hours=int(end_hour))
    start_dt = pd.Timestamp(sel_date) + pd.Timedelta(hours=int(start_hour))

    # Compute needed hours ahead from last_ds to end_dt
    delta_hours = (end_dt - last_ds).total_seconds() / 3600.0
    hours_needed = int(max(0, ceil(delta_hours)))
    if hours_needed == 0:
        # Error: requested end time not after last training timestamp
        return (
            None,
            "<div style='padding:12px;border:1px solid #ccc;border-radius:6px;background:#fff3cd;'>Selected end time is not after model's last training time. Please pick a later date.</div>",
            None,
            None,
        )

    # Predict enough horizon
    df_future = predict_future(route_id=route_id, future_hours=hours_needed)

    # Filter to the requested window on the selected date
    mask = (df_future["ds"] >= start_dt) & (df_future["ds"] <= end_dt)
    df_window = df_future.loc[mask].copy()
    if df_window.empty:
        # Error: no data in requested window
        return (
            None,
            "<div style='padding:12px;border:1px solid #ccc;border-radius:6px;background:#fff3cd;'>No data available for the selected window. Try a later time.</div>",
            None,
            None,
        )

    # Summary: average congestion and interpreted status
    avg_level = float(df_window["yhat"].mean())
    status = interpret_congestion(avg_level)
    color = color_for_status(status)
    summary_html = f"""
    <div style='padding:16px;border:1px solid #e0e0e0;border-radius:8px;background:#f8f9fa;margin-bottom:8px;'>
      <div style='font-size:16px;font-weight:700;margin-bottom:6px;'>Forecast Summary</div>
      <div><b>Route:</b> {readable_choice}</div>
      <div><b>Date:</b> {sel_date}</div>
      <div><b>Window:</b> {int(start_hour):02d}:00 - {int(end_hour):02d}:00</div>
      <div><b>Average Congestion:</b> {avg_level:.2f} / 10</div>
      <div><b>Status:</b> <span style='color:{color}; font-weight:700;'>{status}</span></div>
    </div>
    """

    fig = make_plot(df_window)

    # Also return the numeric average and status HTML for separate components if desired
    return fig, summary_html, avg_level, f"<span style='color:{color}; font-weight:600;'>{status}</span>"


with gr.Blocks(title="Sekka Admin Dashboard - Congestion Forecast") as demo:
    gr.Markdown("# Sekka Admin Dashboard - Congestion Forecast")
    gr.Markdown(essential_note)

    # Build mapping and restrict to trained routes
    readable_to_id, id_to_readable = build_route_mapping()
    trained_ids = set(list_trained_route_ids())
    # Keep only labels that have trained models
    choices = [label for label, rid in readable_to_id.items() if rid in trained_ids]
    choices = sorted(choices)

    with gr.Row():
        route_dropdown = gr.Dropdown(
            choices=choices,
            label="Route",
            value=choices[0] if choices else None,
            interactive=True,
        )
        date_picker = gr.DateTime(label="Select Date", include_time=False)
        start_slider = gr.Slider(minimum=0, maximum=23, step=1, value=8, label="Start Hour (0-23)")
        end_slider = gr.Slider(minimum=0, maximum=23, step=1, value=16, label="End Hour (0-23)")

    summary_html = gr.HTML(label="Summary")
    plot_out = gr.Plot(label="Forecast (Selected Window)")
    level_out = gr.Number(label="Average Congestion (0-10)")
    status_out = gr.HTML(label="Status")

    run_btn = gr.Button("Run Forecast")
    run_btn.click(
        fn=predict_ui,
        inputs=[route_dropdown, date_picker, start_slider, end_slider, gr.State(readable_to_id)],
        outputs=[plot_out, summary_html, level_out, status_out],
        api_name="predict_window",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
