# Nordex Shift System

## Project Overview

Nordex Shift System is a data analytics and machine learning project focused on manufacturing shift performance. The project analyzes production output, machine downtime, maintenance activity, operator experience, quality inspection results, and environmental conditions to understand what drives operational efficiency across different production shifts.

The goal is to help a manufacturing business identify performance patterns, reduce downtime, improve product quality, and support better shift planning decisions.

## Business Problem

Manufacturing teams often need to answer practical operational questions:

- Which shifts produce the highest output?
- How does downtime affect production efficiency?
- Are maintenance events linked to lower performance?
- Which quality or machine factors contribute to defects?
- Can shift efficiency be predicted from operational data?

This project uses historical shift data to explore these questions and build a machine learning workflow for predicting shift efficiency.

## Dataset

The project uses a SQLite database named `ShiftData.db` containing the `ShiftPerformance` table.

Dataset summary:

- 296,334 production shift records
- 31 original columns
- Production, maintenance, machine, operator, quality, and environmental features
- Target variable: `shift_efficiency_score`

## Project Workflow

1. Connected to a SQLite production database.
2. Loaded shift performance data into pandas.
3. Reviewed dataset structure, missing values, duplicates, and statistical summaries.
4. Cleaned missing maintenance and environmental records.
5. Performed exploratory data analysis on numerical and categorical features.
6. Analyzed shift performance, downtime, maintenance impact, and production output per hour.
7. Estimated OEE (Overall Equipment Effectiveness) using availability, performance, and quality.
8. Engineered machine learning features such as shift duration, defect rate, downtime ratio, day of week, and operating hours.
9. Built preprocessing pipelines for numerical and categorical data.
10. Trained and compared regression models for shift efficiency prediction.
11. Logged model experiments with MLflow and DagsHub.

## Machine Learning Models

The notebook compares the following regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Model performance is evaluated using:

- R2 score
- Mean Absolute Error
- Mean Squared Error

## Key Business Insights Explored

- Shift-level production output and efficiency trends
- Maintenance vs no-maintenance operational performance
- Downtime impact on production and machine availability
- Monthly OEE trend analysis
- Operator experience relationship with defect count
- Production output per operating hour by shift

## Repository Structure

```text
Nordex_Shift_System/
├── .vscode/
│   └── settings.json
├── docs/
│   ├── DATA_DICTIONARY.md
│   └── PROJECT_REPORT.md
├── notebooks/
│   └── EDA.ipynb
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── ShiftData.db
```

## Technologies Used

- Python
- pandas
- NumPy
- SQLite
- Matplotlib
- Seaborn
- scikit-learn
- MLflow
- DagsHub
- FastAPI
- Uvicorn
- Jupyter Notebook

## How to Run the Project

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
pip install -r requirements.txt ipykernel
```

Open the notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Documentation

- [Project Report](docs/PROJECT_REPORT.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)

## Future Improvements

- Move reusable notebook logic into Python modules under a `src/` folder.
- Add automated tests for data cleaning and feature engineering steps.
- Build a FastAPI endpoint for model inference.
- Add dashboard visuals for operations managers.
- Store large datasets in cloud storage or Git LFS for production use.
- Add model versioning and deployment documentation.

## Author

Created by Michael Kenechukwu as a manufacturing analytics and machine learning portfolio project.
