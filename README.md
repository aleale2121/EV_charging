# EV Charging Active-Session Forecast

This project analyzes municipal EV charging sessions and forecasts station-hour charging demand. The core notebook and Python package both treat the raw data correctly as event-level charging sessions, then convert it into a regular hourly time series of active charger usage.

The main planning question is:

How many vehicles are actively connected at each station during each future hour?

## Project Overview

The dataset contains individual charging sessions with connection times, disconnection times, station identifiers, energy delivered, and duration fields. A single charging session can span many hours, so the project expands every session across each hour it overlaps before modeling demand.

This produces the main target:

`session_count`: active connected sessions per `station_name`, `location_name`, and hour.

The project also models hourly energy consumption and includes user-behavior clustering to summarize different charging patterns.

## Main Components

```text
.
├── notebooks/
│   ├── EV_Charging_Time_Series_Forecast.ipynb
├── src/
│   ├── ev_charging_forecast/
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── forecast.py
│   │   └── modeling.py
│   ├── ev_charging_analysis.py
│   └── api.py
├── dashboard/
│   └── streamlit_app.py
├── outputs/
│   ├── time_series_forecast_notebook/
│   └── time_series_forecast_project/
├── data/
├── reports/
├── requirements.txt
└── README.md
```

## Notebook

`EV_Charging_Time_Series_Forecast.ipynb` is the main analysis notebook. It includes:

- raw data inspection and cleaning
- session expansion into active hourly station demand
- demand and energy target analysis
- temporal lag, rolling, EWM, station-level, and global-load features
- demand forecasting model comparison
- energy forecasting model comparison
- feature-importance analysis
- user-behavior clustering
- future station-hour forecast generation
- dashboard-ready output files


## Python Package

The `src/ev_charging_forecast` package turns the notebook workflow into reusable project code:

- `data.py`: loads raw sessions, parses timestamps, fixes overnight sessions, and builds active station-hour demand
- `features.py`: creates calendar, lag, rolling, exponential smoothing, station-memory, and global-load features
- `modeling.py`: trains and evaluates forecasting models, saves predictions, feature importance, and artifacts
- `forecast.py`: creates recursive future station-hour forecasts
- `cli.py`: coordinates the full pipeline

`src/ev_charging_analysis.py` is a compatibility entry point for the package.

## Dashboard

The Streamlit dashboard summarizes the saved notebook or project outputs. It includes:

- demand prediction overview
- energy prediction overview
- predicted vs actual active-session charts
- future system demand forecast
- user-behavior clustering summary
- sorted demand and energy feature importance
- curated notebook figures, including model comparison and clustering visuals
- browsable CSV output tables

## API

The FastAPI app exposes saved forecast artifacts and model outputs. It is designed for lightweight local serving of:

- model comparison tables
- test predictions
- hourly forecasts
- daily station forecasts
- engineered-feature predictions when a trained project artifact is available

## Outputs

Notebook outputs are saved under:

`outputs/time_series_forecast_notebook`

Python project outputs are saved under:

`outputs/time_series_forecast_project`

Typical outputs include:

- model comparison tables
- demand and energy test predictions
- feature-importance files
- station-hour and station-day forecasts
- user-behavior cluster summaries
- forecast and diagnostic figures
- cleaned-data and run metadata

## Modeling Approach

The strongest tabular models use leakage-safe temporal features:

- short-term lags
- daily and weekly lags
- rolling means, sums, and maxima
- exponential weighted means
- station-level historical behavior
- system-wide load signals
- calendar seasonality

This feature design is important because EV charging demand is sparse, seasonal, station-specific, and spike-prone.

## Interpretation

The forecasts are intended as decision-support signals for charger planning and operational monitoring. They should be interpreted alongside real-world constraints such as site capacity, charger availability, electrical infrastructure, parking policy, equity priorities, and budget.
