# Project Report: Nordex Shift System

## Executive Summary

Nordex Shift System is a manufacturing analytics project designed to evaluate production shift performance and predict shift efficiency using operational data. The project combines exploratory data analysis, operational performance metrics, OEE estimation, feature engineering, and machine learning.

The project demonstrates how production, maintenance, machine, operator, quality, and environmental data can be combined to support better manufacturing decisions.

## Objective

The main objective is to analyze shift-level production performance and build a machine learning workflow that can predict `shift_efficiency_score`.

The business goal is to support:

- better shift planning
- improved machine uptime
- reduced defect rates
- stronger maintenance decision-making
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

The notebook performs several preparation steps:

- connects to the SQLite database
- loads the `ShiftPerformance` table into pandas
- converts date and time fields to datetime format
- checks duplicates and missing values
- fills missing temperature and humidity values
- fills missing maintenance fields with business-friendly default values
- removes identifier fields that do not help modeling
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
- `total_operating_hours`
- `day_of_week`
- `hour_of_day`

These features help the model better understand production behavior and efficiency drivers.

## Modeling Approach

The project uses a scikit-learn `ColumnTransformer` to handle both numerical and categorical features.

Numerical features are passed through directly, while categorical features are encoded using `OneHotEncoder`.

Models compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Metrics used:

- R2 score
- Mean Absolute Error
- Mean Squared Error

Experiment tracking is configured with MLflow and DagsHub.

## Business Value

This project shows how data can be used to improve manufacturing operations by:

- identifying low-performing shifts
- understanding downtime impact
- measuring maintenance-related performance changes
- estimating OEE trends
- predicting shift efficiency
- giving managers evidence for operational decisions

## Recommendations

Recommended next steps:

- convert notebook logic into reusable Python scripts
- create a dashboard for shift supervisors
- deploy the best model through FastAPI
- monitor model performance over time
- add automated tests and CI checks
- document model assumptions and limitations

## Conclusion

Nordex Shift System is a strong foundation for a production analytics and machine learning solution. It demonstrates database access, data cleaning, exploratory analysis, performance engineering, model training, and experiment tracking in a real-world manufacturing context.
