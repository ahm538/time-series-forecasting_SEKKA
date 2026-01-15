# 🚖 Sekka: AI-Powered Transportation Intelligence Engine

> Class of 2026 Graduation Project · Faculty of Computers & Information, Mansoura University

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-success)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)

---

##  Executive Summary
Sekka is a transportation "Super App" focused on Mansoura, Egypt, and its connections to surrounding governorates (Alexandria, Damietta, Dakahlia (internal), Sharqia, Gharbia). Sekka tackles the day-to-day unpredictability of microbus and bus services by forecasting route congestion levels up to 7 days ahead. This enables better trip planning, dynamic pricing, fleet scheduling, and improved commuter experience.

- Forecast horizon: up to 7 days (hourly resolution)
- Per-route models: one Prophet model per transportation route
- Core output: `congestion_level` on a 0–10 scale

---

##  AI Logic & Egyptian Context
Sekka leverages Facebook Prophet (a robust additive time series model) extended with external regressors to capture Egypt-specific patterns:

- Multiple seasonalities
  - Daily intraday pattern ("M-Shape": morning peak ~7–9 AM; evening peak ~4–6 PM)
  - Weekly effects (Friday dips; Thursday night/Saturday morning spikes)
  - Yearly effects (broad seasonal shifts)
- External regressors (per-interval features)
  - National holidays (EG) via Prophet holidays and/or custom `is_public_holiday`
  - Academic calendar via `school_term_phase` (Term / Exam / Holiday)
  - Summer season via `is_summer_peak` (June–September), especially for coastal routes (Damietta, Alexandria)

Each route is trained separately to respect unique demand dynamics and service types (Bus, Microbus, Student, etc.). Models and metadata are persisted to enable real-time inference through an API and an admin dashboard.

---

## 📂 Data Dictionary
High-fidelity synthetic dataset: `data/sekka_mansoura_synthetic_dataset_2years_v2.csv` — hourly data for 2024–2025.

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime (hourly) | Timestamp of observation (YYYY-MM-DD HH:MM:SS). |
| `route_id` | string | Unique identifier for the transport line. |
| `congestion_level` | float (0–10) | Target variable representing congestion intensity. |
| `target_governorate_en` | string | Destination (e.g., Alexandria, Damietta). |
| `service_type` | string | Type of service (Bus, Microbus, Student, etc.). |
| `is_public_holiday` | 0/1 | Whether the timestamp falls on a national holiday. |
| `school_term_phase` | categorical | One of `Term`, `Exam`, or `Holiday`. |
| `is_summer_peak` | 0/1 | Summer season indicator (June–September). |

---

##  Project Structure
```
project_root/
├── data/                   # CSV files (place dataset here)
├── models/                 # Saved .joblib models and metadata per route (generated)
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths, hyperparameters, horizon, API config
│   ├── preprocessing.py    # Feature engineering (holidays, school phases, seasonality helpers)
│   ├── training.py         # Train per-route Prophet models and save artifacts
│   └── inference.py        # Load models and predict future windows
├── app.py                  # FastAPI application (real-time inference)
├── demo.py                 # Gradio Admin Dashboard (readable routes, calendar/date window)
├── train_pipeline.py       # Train models for ALL routes and create training_report.csv
└── requirements.txt        # Dependencies
```

---

##  Installation & Usage
The following steps assume Windows PowerShell; adjust accordingly for your OS.

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Data setup
- Move `sekka_mansoura_synthetic_dataset_2years_v2.csv` into the `data/` directory.

### 3) Train all route models
```powershell
python train_pipeline.py
```
Artifacts created:
- `models/model_route_{route_id}.joblib` (Prophet model)
- `models/metadata_route_{route_id}.json` (last training timestamp & metadata)
- `training_report.csv` (per-route MAE/RMSE summary)

### 4) Run the Admin Dashboard (Gradio)
```powershell
python demo.py
```
Open the local URL (e.g., http://127.0.0.1:7861). The dashboard provides:
- Readable route names from the dataset (e.g., "Mansoura -> Alexandria (Bus) - [ID: R-1005]")
- Date picker + start/end hour window selection
- Plotly forecast with uncertainty
- Summary card with average congestion and interpreted status

### 5) Start the API (FastAPI)
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```
Example request:
```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "route_id": "R-1005",
  "future_hours": 24
}
```
Example response (truncated):
```json
{
  "route_id": "R-1005",
  "points": [
    { "timestamp": "2026-02-15T08:00:00", "yhat": 7.5, "yhat_lower": 6.8, "yhat_upper": 8.2 }
  ]
}
```

---

##  Interpreting Results (Traffic Light System)
The AI output (0–10) is mapped to human-friendly statuses for decision-making:

| Score | Color | Status | Arabic | Meaning |
|---|---|---|---|---|
| 0–3 | 🟢 Green | Clear | سالكة | Excellent flow; optimal travel time. |
| 3–6 | 🟡 Yellow | Moderate | ماشية | Normal traffic; minor delays possible. |
| 6–8 | 🟠 Orange | Heavy | زحمة | Slow movement; significant delays likely. |
| 8–10 | 🔴 Red | Severe | واقفة | Gridlock; avoid if possible. |

This mapping is implemented in `src/inference.py: interpret_congestion()` and surfaced in both the API consumer logic and the dashboard summary.

---

##  Technical Notes
- Prophet is configured with daily, weekly, and yearly seasonalities and tuned Fourier orders to emphasize intraday patterns (M-Shape) and weekly effects.
- Country holidays (Egypt) are added when available; otherwise, a custom holiday signal is used.
- Academic calendar is encoded via one-hot regressors for `Term`, `Exam`, and `Holiday` phases.
- All predictions are clipped to the 0–10 target range.

---

##  Evaluation
- Training uses the first ~23 months; evaluation is performed on the last 30 days (MAE and RMSE). Results per route are captured in `training_report.csv`.

---

##  License & Acknowledgments
- Built by the Class of 2026, Faculty of Computers & Information, Mansoura University.
- Facebook Prophet is developed by Meta; FastAPI by Sebastián Ramírez.

For questions or collaboration, please open an issue or contact the maintainers.
