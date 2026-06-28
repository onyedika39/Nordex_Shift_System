# Nordex Shift System

Nordex Shift System is a manufacturing analytics and machine learning project for understanding and predicting production shift efficiency. It combines SQLite data extraction, exploratory analysis, feature engineering, model training, MLflow/DagsHub experiment tracking, a FastAPI inference service, and a Streamlit dashboard.

## Business Problem

Manufacturing teams need clear answers to operational questions such as:

- Which shifts produce the highest output?
- How does downtime affect production efficiency?
- Are maintenance events linked to lower shift performance?
- Which machine, quality, and operator factors contribute to defects?
- Can shift efficiency be predicted from historical operational data?

The project uses historical shift-level production data to identify performance patterns and support better planning, maintenance, and quality decisions.

## Dataset

The project uses a SQLite database named `ShiftData.db` with a `ShiftPerformance` table.

Dataset summary:

- 296,334 shift records
- 31 original columns
- Production, maintenance, machine, operator, quality, and environmental features
- Target variable: `shift_efficiency_score`

## Project Workflow

1. Load shift performance data from SQLite.
2. Validate empty records, missing values, and duplicates.
3. Clean environmental and maintenance-related fields.
4. Engineer performance features such as `shift_duration`, `defect_rate`, `downtime_ratio`, `day_of_week`, and `hour_of_day`.
5. Analyze production output, downtime, defects, maintenance impact, and OEE trends.
6. Train regression models for shift efficiency prediction.
7. Track experiments and model versions with MLflow and DagsHub.
8. Serve predictions through FastAPI.
9. Provide an interactive Streamlit dashboard for prediction, optimization, and retraining.

## Machine Learning

The project compares regression models including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Model performance is evaluated with:

- R2 score
- Mean Absolute Error
- Mean Squared Error

The production pipeline uses preprocessing for numerical and categorical features through scikit-learn `ColumnTransformer` and `Pipeline` objects.

## Repository Structure

```text
Nordex_Shift_System/
|-- app/
|   `-- main.py
|-- config/
|   |-- constant.py
|   `-- schema.yml
|-- docs/
|   |-- DATA_DICTIONARY.md
|   `-- PROJECT_REPORT.md
|-- notebooks/
|   `-- EDA.ipynb
|-- src/
|   |-- data/
|   |-- exception/
|   |-- features/
|   |-- logger/
|   |-- models/
|   |-- pipeline/
|   `-- utils/
|-- .gitignore
|-- .dockerignore
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- setup.py
|-- ShiftData.db
`-- streamlit_app.py
```

## Technologies Used

- Python
- pandas and NumPy
- SQLite
- Matplotlib and Seaborn
- scikit-learn
- MLflow and DagsHub
- FastAPI and Uvicorn
- Streamlit
- Optuna
- Jupyter Notebook

## Setup

Clone the repository:

```bash
git clone https://github.com/onyedika39/Nordex_Shift_System.git
cd Nordex_Shift_System
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8001
```

Open the API documentation at:

```text
http://127.0.0.1:8001/docs
```

## Run the Dashboard

Start the API first, then run:

```bash
streamlit run streamlit_app.py
```

By default, the dashboard connects to:

```text
http://127.0.0.1:8001
```

For a deployed API, set the `API_URL` environment variable before starting Streamlit.

## Docker Deployment

Build the API image:

```bash
docker build -t nordex-shift-system .
```

Run the API container locally:

```bash
docker run -p 8001:8001 nordex-shift-system
```

Open:

```text
http://127.0.0.1:8001/docs
```

## Documentation

- [Project Report](docs/PROJECT_REPORT.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)

## Cloud Deployment

The FastAPI service was containerized with Docker and deployed to Azure Container Instances. Runtime credentials such as DagsHub or MLflow tokens should be provided through environment variables and should not be committed to GitHub.

The deployed API exposes a health endpoint and interactive API documentation:

```text
/
/docs
```

## Author

Created by Michael Kenechukwu as a manufacturing analytics and machine learning portfolio project.
