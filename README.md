# Inventory & Demand Forecasting System

A machine learning-powered demand forecasting system that predicts future product sales using historical sales data. The project helps businesses optimize inventory levels, reduce stock shortages, and minimize overstocking by providing accurate demand forecasts.

---

## Project Overview

Inventory management is a critical challenge for retail, supply chain, and food service companies. Inaccurate demand planning can lead to:

- Stock shortages
- Excess inventory
- Increased storage costs
- Revenue loss

This project uses historical sales data and machine learning models to forecast future demand and provide inventory recommendations.

---

## Business Problem

Companies need to answer questions such as:

- How many units of a product will be sold next week?
- Which products are expected to have increased demand?
- When should inventory be replenished?
- How much stock is required to avoid shortages?

This system provides demand predictions that help businesses make data-driven inventory decisions.

---

## Features

### Data Analysis
- Historical sales analysis
- Product-level demand trends
- Store-level performance insights
- Seasonal pattern detection

### Feature Engineering
- Time-based features
- Lag features
- Rolling averages
- Seasonal indicators

### Machine Learning
- Linear Regression baseline model
- XGBoost forecasting model
- Model performance comparison
- MAE and RMSE evaluation metrics

### Forecasting API
- REST API built with FastAPI
- Product demand prediction endpoint
- JSON response support
- Interactive Swagger documentation

### Dashboard
- Streamlit web dashboard
- Product selection interface
- Demand forecast visualization
- Historical vs predicted demand charts

### Inventory Recommendations
- Predicted demand estimation
- Stock shortage alerts
- Reorder quantity recommendations

---

## System Architecture

```text
                    Historical Sales Data
                              │
                              ▼
                     Data Preprocessing
                              │
                              ▼
                    Feature Engineering
             (Lag Features, Rolling Means,
                  Seasonal Features)
                              │
                              ▼
                      Machine Learning
               (Linear Regression/XGBoost)
                              │
                              ▼
                    Demand Predictions
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
   FastAPI REST API                    Streamlit Dashboard
```

---

## Technology Stack

### Programming Language

- Python 3.13

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn
- XGBoost

### Backend

- FastAPI
- Uvicorn

### Frontend Dashboard

- Streamlit
- Plotly

### Database (Future Enhancement)

- PostgreSQL
- SQLAlchemy
- Alembic

### DevOps

- Git
- GitHub
- Docker (Optional)

---

## Project Structure

```text
inventory-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── demand_model.pkl
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## Dataset

### Store Item Demand Forecasting Dataset

Dataset contains:

| Column | Description |
|----------|------------|
| date | Transaction date |
| store | Store identifier |
| item | Product identifier |
| sales | Number of units sold |

Example:

```csv
date,store,item,sales
2013-01-01,1,1,13
2013-01-02,1,1,11
2013-01-03,1,1,14
```

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Missing value handling
2. Date conversion
3. Duplicate removal
4. Outlier analysis
5. Data type optimization

Example:

```python
df["date"] = pd.to_datetime(df["date"])
```

---

## Feature Engineering

### Time Features

Generated features:

```text
year
month
quarter
day
day_of_week
week_of_year
```

### Lag Features

To capture historical demand patterns:

```text
lag_7
lag_14
lag_30
```

Example:

```text
Sales from 7, 14 and 30 days ago
```

### Rolling Statistics

Generated features:

```text
rolling_mean_7
rolling_mean_14
rolling_mean_30
```

These features help capture short-term demand trends.

---

## Machine Learning Models

### Baseline Model

Linear Regression

Purpose:

- Establish benchmark performance
- Simple and interpretable

### Advanced Model

XGBoost Regressor

Advantages:

- Handles nonlinear patterns
- Captures feature interactions
- High forecasting accuracy

---

## Model Evaluation

Evaluation metrics:

### MAE

Mean Absolute Error

```text
MAE = Average Absolute Prediction Error
```

### RMSE

Root Mean Squared Error

```text
RMSE = Square Root of Average Squared Errors
```

Sample Results:

| Model | MAE | RMSE |
|---------|---------|---------|
| Linear Regression | 6.20 | 8.05 |
| XGBoost | 3.75 | 5.02 |

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### Forecast Demand

```http
GET /forecast
```

Parameters:

```text
store_id
item_id
forecast_date
```

Example Request:

```http
GET /forecast?store_id=1&item_id=10
```

Response:

```json
{
  "store_id": 1,
  "item_id": 10,
  "predicted_demand": 42
}
```

---

## Dashboard Features

### User Inputs

- Product ID selection
- Store ID selection
- Forecast date selection

### Visualizations

- Historical sales trend
- Predicted demand trend
- Inventory recommendations

### Business Insights

Example:

```text
Current Stock: 100

Predicted Demand: 160

Recommended Reorder Quantity: 60
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/inventory-forecasting.git

cd inventory-forecasting
```

### Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train Model

```bash
python src/train.py
```

### Start FastAPI Server

```bash
uvicorn api.main:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

Planned enhancements:

- LightGBM model implementation
- Prophet forecasting model
- PostgreSQL integration
- Docker containerization
- CI/CD pipeline
- User authentication
- Multi-store forecasting
- Automated model retraining
- Cloud deployment (AWS/Azure)

---

## Skills Demonstrated

### Data Science

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Time Series Forecasting
- Model Evaluation

### Machine Learning

- Regression Models
- XGBoost
- Hyperparameter Tuning
- Performance Optimization

### Backend Development

- FastAPI
- REST API Design
- Serialization
- Model Serving

### Software Engineering

- Git Version Control
- Project Structure Design
- Documentation
- Testing

---

## Learning Outcomes

By completing this project, I gained hands-on experience in:

- Building end-to-end machine learning pipelines
- Forecasting business demand using historical data
- Productionizing machine learning models
- Developing APIs for model inference
- Creating interactive data dashboards

---

## Author

Your Name

GitHub: https://github.com/your-username

LinkedIn: https://linkedin.com/in/your-profile

---
