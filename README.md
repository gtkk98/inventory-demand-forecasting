# Inventory & Demand Forecasting System

An end-to-end machine learning project that forecasts retail product demand from historical sales and turns demand statistics into inventory replenishment recommendations.

The project combines data validation and preparation, time-series feature engineering, model comparison, a FastAPI forecasting service, and an interactive Streamlit dashboard.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run the Application](#run-the-application)
- [API Usage](#api-usage)
- [Inventory Planning](#inventory-planning)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)

## Project Overview

Retailers need to estimate future demand to reduce stockouts, avoid unnecessary overstock, and make better replenishment decisions. This project explores historical store-item sales, trains and evaluates forecasting models, and provides a user interface for examining predictions and calculating inventory recommendations.

### Goals

- Validate and prepare historical sales data.
- Explore demand patterns by date, store, and item.
- Build calendar, lag, and rolling-statistic features.
- Compare machine learning forecasts with simple baselines.
- Expose forecasting functionality through a REST API.
- Present analysis and forecasts in a Streamlit dashboard.
- Calculate inventory planning estimates from historical demand and manually entered stock information.

## Features

### Data preparation and analysis

- Load and validate historical sales data.
- Check data quality, dates, sales values, and duplicate store-item-date records.
- Explore daily sales and store/item demand patterns.
- Prepare processed sales data for feature generation and forecasting.

### Feature engineering

The XGBoost pipeline uses calendar features and historical demand features, including:

- Day of week and day of month.
- Month, year, quarter, and week of year.
- Weekend indicator.
- One-day and seven-day sales lags.
- Seven-day and 30-day rolling mean features.
- Seven-day rolling standard deviation.

Time-series features are designed to use past observations rather than future sales. The evaluation workflow uses a chronological holdout period.

### Forecasting models

The evaluated approaches include:

- Previous-day Naive baseline.
- Seasonal Naive baseline.
- Linear Regression.
- XGBoost.

### FastAPI service

- REST endpoints for forecasting.
- Request and response validation.
- JSON responses.
- Interactive OpenAPI/Swagger documentation.
- Automated tests for endpoint behavior and feature generation.

### Streamlit dashboard

The dashboard contains six areas:

1. **Overview & Models** — model performance and overview.
2. **Daily Sales** — historical sales exploration.
3. **Error Analysis** — forecast error analysis.
4. **Store & Item Explorer** — inspect selected store/item results.
5. **Live Forecast** — request forecasts through the API.
6. **Inventory Planning** — calculate stock replenishment recommendations.

### Inventory recommendation engine

The inventory planning feature calculates:

- Average daily demand.
- Daily demand standard deviation.
- Safety stock.
- Reorder point.
- Target stock level.
- Recommended order quantity.
- A `REORDER` or `MONITOR` status.

The user supplies current stock, supplier lead time, target service level, and target days of stock.

**Important:** The historical dataset does not contain live stock balances, supplier records, or purchase orders. Inventory recommendations are estimates based on historical demand and user-provided inventory inputs, not a live inventory integration.

## Dataset

This project uses the **Store Item Demand Forecasting** dataset from Kaggle.

The data includes daily sales for 10 stores and 50 items over five years.

| Property | Value |
|---|---:|
| Historical records | 913,000 |
| Stores | 10 |
| Items | 50 |
| Store-item combinations | 500 |
| Start date | 2013-01-01 |
| End date | 2017-12-31 |
| Total recorded sales | 47,704,512 units |

The input CSV contains these columns:

| Column | Description |
|---|---|
| `date` | Sales observation date |
| `store` | Store identifier |
| `item` | Item identifier |
| `sales` | Number of units sold |

Download the dataset from Kaggle and place the training file at:

```text
data/raw/train.csv
```

Respect the dataset's license and terms when redistributing data.

## Model Evaluation

The current evaluation uses 2013–2016 for training and 2017 as the chronological test period. The following metrics were recorded for the test period:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| XGBoost | 6.0858 | 7.9132 | 10.3474% |
| Linear Regression | 7.0660 | 9.2381 | 12.0139% |
| Seasonal Naive | 8.8415 | 11.5967 | 15.0326% |
| Previous-day Naive | 11.2447 | 15.3892 | 19.1188% |

XGBoost produced the lowest MAE, RMSE, and WAPE among these evaluated models.

- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted sales. Lower is better.
- **RMSE (Root Mean Squared Error):** square root of the average squared error. Lower is better; larger errors have greater influence.
- **WAPE (Weighted Absolute Percentage Error):** sum of absolute errors divided by total actual sales, expressed as a percentage. Lower is better.

These are historical holdout results, not a guarantee of future performance. Results may differ for other datasets, stores, products, or operating conditions.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.14+ |
| Dependency and environment management | uv |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Model serialization | Joblib |
| Testing | Pytest |
| Code quality tooling | Ruff |
| Version control | Git, GitHub |

## Project Structure

The project uses a `src`-based Python package. The following is a representative view of important files and directories; additional analysis and validation modules are present.

```text
inventory-demand-forecasting/
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       ├── sales_clean.csv
│       └── xgboost_predictions.csv
├── models/
│   └── xgboost_model.joblib
├── reports/
├── src/
│   ├── inventory_demand_forecasting/
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   ├── schemas.py
│   │   │   └── forecast_service.py
│   │   ├── dashboard.py
│   │   ├── data.py
│   │   ├── cleaning.py
│   │   ├── features.py
│   │   ├── lag_features.py
│   │   ├── rolling_features.py
│   │   ├── train_xgboost.py
│   │   ├── evaluate.py
│   │   └── xgboost_model.py
│   └── inventory/
│       └── recommendations.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### Prerequisites

- Python 3.14 or newer, as specified in `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/) for project and dependency management.
- Git, if cloning the repository.

### 1. Clone the repository

Replace the placeholder with the URL of your GitHub repository:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd inventory-demand-forecasting
```

### 2. Install dependencies

From the project root:

```powershell
uv sync
```

This installs the dependencies declared in `pyproject.toml` and uses the lock file to resolve the project environment.

### 3. Prepare the dataset and artifacts

Place the Kaggle CSV at `data/raw/train.csv`. Follow the project's data preparation and training workflow to generate the processed data, reports, and model artifacts required by the API and dashboard.

Do not assume that downloading the raw dataset alone generates the processed files or trained model.

## Run the Application

Run the API and dashboard in separate terminals from the project root.

### Start the FastAPI service

```powershell
uv run uvicorn inventory_demand_forecasting.api.main:app --app-dir src --reload
```

Open the interactive API documentation:

<http://127.0.0.1:8000/docs>

### Start the Streamlit dashboard

```powershell
uv run python -m streamlit run src/inventory_demand_forecasting/dashboard.py
```

Open the local URL printed in the Streamlit terminal. Streamlit may choose a different port if its default port is already occupied.

## API Usage

The FastAPI service exposes forecasting functionality. To inspect the actual endpoints, parameters, validation constraints, and response schemas:

1. Start the API using the command above.
2. Open <http://127.0.0.1:8000/docs>.
3. Expand an endpoint and select **Try it out**.
4. Submit a valid request and inspect the JSON response.

The live OpenAPI documentation is the source of truth for the API contract. It is preferable to relying on example endpoint names or schemas that may become outdated.

## Inventory Planning

1. Start the Streamlit dashboard.
2. Open **Inventory Planning**.
3. Choose a store and item.
4. Enter current stock and supplier lead time.
5. Choose a target service level and target days of stock.
6. Select **Calculate Recommendation**.
7. Review demand statistics, safety stock, reorder point, recommended order quantity, and status.
8. Download the recommendation as a CSV if needed.

Validate recommendations against real inventory records, supplier conditions, and business constraints before using them in operational decisions.

## Testing

Run the full automated test suite:

```powershell
uv run python -m pytest -v
```

The current verified suite contains **75 passing tests**.

Compile the dashboard module to check its syntax:

```powershell
uv run python -m py_compile src/inventory_demand_forecasting/dashboard.py
```

The tests cover API behavior, schema validation, feature engineering, model evaluation, data quality, dashboard helpers, and inventory recommendation logic.

## Limitations

- Historical sales are not the same as real-time inventory.
- The inventory recommendation engine relies on manually supplied stock and lead-time values.
- Model evaluation results are based on a historical holdout period.
- Forecast quality can change as demand patterns, prices, promotions, and operating conditions change.
- Deployment, monitoring, and external integrations require separate verification before operational use.

## Future Improvements

Potential future enhancements include:

- Cloud deployment and application monitoring.
- Docker packaging.
- Continuous integration and delivery.
- Automated retraining and model drift monitoring.
- Integration with inventory, purchasing, and supplier systems.
- Evaluation of additional forecasting algorithms.
- Authentication and role-based access controls where appropriate.

These are potential improvements, not claims that they are already implemented.

## Skills Demonstrated

**Data science and machine learning**
- Data validation and preparation.
- Exploratory data analysis.
- Time-series feature engineering.
- Demand forecasting and baseline comparisons.
- Model evaluation using MAE, RMSE, and WAPE.

**Backend and application development**
- REST API development with FastAPI.
- Request and response validation.
- Model inference.
- Interactive dashboards with Streamlit.

**Software engineering**
- Modular Python package structure.
- Dependency management with uv.
- Automated testing with Pytest.
- Code quality tooling.
- Git-based version control and technical documentation.

## Project Status

The forecasting pipeline, model evaluation workflow, FastAPI service, Streamlit dashboard, and inventory recommendation functionality have been implemented and tested locally. The current automated test suite has 75 passing tests.

Cloud deployment and final portfolio validation are tracked as separate project milestones.

## License

Add an appropriate license before publicly distributing the project. Also comply with the terms of the dataset provider.
