# Project Report: Nordex Shift System

## Executive Summary

Nordex Shift System is a manufacturing analytics and machine learning project designed to evaluate production shift performance and predict shift efficiency using operational data. The project combines database extraction, exploratory data analysis, operational metrics, feature engineering, supervised learning, experiment tracking, and application deployment components.

The solution demonstrates how production, maintenance, machine, operator, quality, and environmental data can be combined to support better manufacturing decisions.

## Objective

The main objective is to analyze shift-level production performance and build a machine learning workflow that predicts `shift_efficiency_score`.

The business goal is to support:

- better shift planning
- improved machine uptime
- reduced defect rates
- stronger maintenance decisions
- higher production efficiency

## Data Source

The data is stored in `ShiftData.db`, a SQLite database containing a `ShiftPerformance` table.

The dataset includes 296,334 records and 31 original columns covering:

- shift information
- production output
- defect counts
- machine runtime and downtime
- maintenance records
- quality control results
- operator experience
- environmental conditions

## Data Preparation

The project performs several preparation steps:

- connects to the SQLite database
- loads the `ShiftPerformance` table into pandas
- validates empty data, missing values, and duplicates
- fills missing temperature and humidity values
- fills missing maintenance fields with business-readable default values
- removes identifier fields that do not support modeling
- creates new performance-related features

## Exploratory Data Analysis

The EDA section studies both numerical and categorical variables.

Key areas analyzed:

- production output distribution
- defect count distribution
- cycle time distribution
- downtime distribution
- runtime hours
- shift efficiency score
- machine status
- maintenance issues
- quality inspection result
- operator experience level

Visualizations include histograms, boxplots, bar charts, and line plots.

## Operational Analysis

The project analyzes shift performance across several operational dimensions:

- average and total units produced by shift
- average cycle time by shift
- defect rate by shift
- downtime by shift
- production output per operating hour
- maintenance flag impact on performance

This gives management a clearer view of which shifts perform best and where process losses may occur.

## OEE Estimation

The notebook estimates OEE using three components:

- Availability: based on downtime and planned shift time
- Performance: based on actual production compared with theoretical output
- Quality: based on defect count compared with units produced

This OEE estimate helps summarize how effectively production time is converted into quality output.

## Feature Engineering

The machine learning workflow creates new features including:

- `shift_duration`
- `defect_rate`
- `downtime_ratio`
- `day_of_week`
- `hour_of_day`

These features help the model better understand production behavior and efficiency drivers.

## Modeling Approach

The project uses a scikit-learn `ColumnTransformer` to handle numerical and categorical features.

Numerical features are passed through directly, while categorical features are encoded with `OneHotEncoder`.

Models compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Metrics used:

- R2 score
- Mean Absolute Error
- Mean Squared Error

Experiment tracking is configured with MLflow and DagsHub.

## Application Layer

The project includes two application interfaces:

- FastAPI service for model loading, prediction, retraining, and shift optimization.
- Streamlit dashboard for interactive shift input, prediction results, optimization exploration, and model retraining requests.

## Business Value

This project shows how data can improve manufacturing operations by:

- identifying low-performing shifts
- understanding downtime impact
- measuring maintenance-related performance changes
- estimating OEE trends
- predicting shift efficiency
- supporting evidence-based operational decisions

## Recommendations

Recommended next steps:

- maintain the Dockerized FastAPI service for consistent local and cloud deployment
- keep the Azure deployment configuration aligned with the production API port and runtime environment variables
- store sensitive DagsHub and MLflow credentials only as environment variables
- move larger generated artifacts to external cloud storage when the project grows
- add automated tests for data processing, feature engineering, and prediction logic
- add CI checks for code quality before future GitHub updates
- monitor model performance over time as new shift data becomes available

## Conclusion

Nordex Shift System is a strong foundation for a production analytics and machine learning solution. It demonstrates database access, data cleaning, exploratory analysis, performance engineering, model training, experiment tracking, API serving, and dashboard interaction in a manufacturing context.
