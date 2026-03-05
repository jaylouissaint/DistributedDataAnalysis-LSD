# Distributed Data Analysis: Airline Delay Prediction (Spark)

End-to-end Spark workflow for cleaning, feature engineering, exploratory analytics, and machine learning on 10 years of U.S. flight performance data (2009–2018). The project builds predictive models for departure delays using engineered time- and airport-aware features, and evaluates performance against a baseline.

## Project Summary

This project uses a large-scale U.S. airline flights dataset (tens of millions of rows) stored on an Azure Data Lake–backed Spark cluster. We:

1. Load and clean raw flight records into a typed Spark table.
2. Produce analytics reporting and visualizations to understand trends in delays.
3. Engineer predictive features (including rolling/time-based features).
4. Train and evaluate Spark ML models to predict departure delays, including cancellations.

Our prediction task is binary classification:
- Delayed (1): DEP_DELAY > 0 or CANCELLED == 1

- Not delayed (0): otherwise

Because we create rolling/time-based features, we use a chronological split strategy (not fully random) to avoid leakage and mimic real forecasting.

## Data

The raw dataset is read from Azure Data Lake Storage via Spark:

```python
delays = spark.read.csv(
  "abfss://sampledata@lsdsampledata2026.dfs.core.windows.net/delays",
  header=True,
  nullValue="NULL"
)
```

### Core Columns (examples)

- Flight identifiers: OP_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST
- Times (scheduled/actual): CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME
- Outcomes: DEP_DELAY, ARR_DELAY, CANCELLED, DIVERTED
- Delay types: CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY
- Route info: DISTANCE, CRS_ELAPSED_TIME, AIR_TIME, etc.

## Pipeline Overview
### Part 1: Cleaning + Exploratory Analytics

Notebook: load_prepare_data.ipynb

Casts columns from strings to appropriate Spark types (dates, ints).

Converts hhmm time fields into “minutes since midnight.”

Fills null delay-type fields with 0.

Saves cleaned data to the metastore:

lsd_2026.default.polyhymnia_cleaned_airline

Notebook: aggregate_calculations.ipynb
Produces required analytics outputs:

Flights per month across the full time range

Weekly delay percentage time series

Weekly delay counts by delay type

Carrier performance summary table

Top 50 airports by delay percentage

Part 2: Feature Engineering

Notebook: feature_engineering.ipynb
Adds predictive features and saves the augmented table:

lsd_2026.default.polyhymnia_feature_engineered

Required features:

day_of_week

rate_weather_delay_dep (origin weather delay rate in previous hour)

rate_weather_delay_arr (destination weather delay rate in previous hour)

dep_traffic_z_score (airport traffic z-score vs historical baseline)

Additional engineered features:

carrier_delay_rate_7d (carrier historical delay rate over trailing 7 days)

route_delay_rate_7d (route historical delay rate over trailing 7 days)

crs_dep_time_bucket, arr_time_bucket (time-of-day buckets)

These features introduce time dependence, so model evaluation uses time-aware splits.

Part 3: Delay Prediction (Spark ML)

We train multiple models using the engineered feature table. The notebooks emphasize:

Avoiding leakage by fitting preprocessing on train only

Efficient splitting and caching for large datasets

Comparing against a baseline model (always predict “no delay”)

Notebooks:

delay_prediction_Random_forest.ipynb

delay_prediction_decision_tree.ipynb

Delay_prediction_SVM.ipynb

Split strategy

We use chronological splits because of rolling features. Two approaches appear in the project:

Date-based split: fixed date cutoffs (e.g., train before 2015, test 2015–2016, validate 2017+)

Quantile-based time split: uses an approximate percentile split on a combined timestamp column (FL_DATE + CRS_DEP_TIME) via approxQuantile to create an ~60/20/20 split without global sorting.

Models

Random Forest and Decision Tree for non-linear interactions and robustness.

LinearSVC (SVM) for a strong linear margin-based classifier (with feature scaling).

Results Highlight (SVM)

Using the quantile-based split, the dataset was split into:

Train: 37,232,160 rows

Test: 11,436,737 rows

Validation: 12,888,067 rows

Hyperparameter tuning on a 10% training sample selected:

regParam = 0.01

maxIter = 20

Validation performance (LinearSVC vs baseline):

Accuracy: 0.6763 vs 0.6403

TPR/Recall: 0.3908 vs 0.0000

FPR: 0.1634 vs 0.0000

Specificity: 0.8366 vs 1.0000

Interpretation: the SVM is much more useful than the baseline because it catches a substantial portion of delayed flights, but this comes with a tradeoff of more false alarms.

Installation / Requirements

Runs in Databricks/Spark environment with:

PySpark (Spark SQL + Spark ML)

pandas, matplotlib (for reporting tables/plots)

No local installation is required if using the course Databricks workspace.

Usage

Run Part 1 cleaning to create the cleaned metastore table:

lsd_2026.default.polyhymnia_cleaned_airline

Run Part 2 feature engineering to create the feature table:

lsd_2026.default.polyhymnia_feature_engineered

Run Part 3 modeling notebooks to train and evaluate models:

Random Forest / Decision Tree / SVM notebooks

File Structure
Part 1 — Cleaning and Reporting

load_prepare_data.ipynb
Loads raw CSV data, casts types, fixes nulls, and writes cleaned Spark table.

aggregate_calculations.ipynb
Produces required exploratory aggregates and plots.

Part 2 — Feature Engineering

feature_engineering.ipynb
Creates rolling and categorical features and writes feature-engineered Spark table.

Part 3 — Modeling

delay_prediction_Random_forest.ipynb
Random Forest training, tuning, evaluation vs baseline.

delay_prediction_decision_tree.ipynb
Decision Tree / Random Forest variants with scaling pipeline.

Delay_prediction_SVM.ipynb
Linear SVM with StandardScaler, quantile-based split, evaluation vs baseline.

Notes on Evaluation

We report:

Accuracy

Confusion matrix

True Positive Rate (Recall/Sensitivity)

False Positive Rate

Specificity

We also compare every model to a baseline that always predicts “no delay” to show the added value of machine learning beyond class imbalance.

Data Source

U.S. airline on-time performance dataset provided via the course cluster in Azure Data Lake Storage (lsdsampledata2026), spanning 2009–2018.