# Australian Rainfall Predictor Model (2008-2017)

This repository contains a reproducible end-to-end machine learning pipeline to predict whether it will rain tomorrow using the Australian Weather data. The pipeline is full-stack, containing data ingestion, cleaning, feature engineering, time-aware splitting, model selection via validation, and final evaluation on a held-out test set. 

It is possible to create pipelines in Cloud infrastructure with minimal coding, or to use LLMs to create a pipeline automatically. However, the aim of this project was to challenge myself to create one manually, to develop my understanding of the key components of a pipeline, and better enable my abilities in data engineering. This repository demonstrates my work on this pipeline, which I have developed over several weeks of work. There is work that can done to develop this pipeline further, but in its current state it is functional and portable package, and I am happy to upload it as is.

The data source is the Australian Weather Dataset (WeatherAUS from Kaggle). The target variable of RainTomorrow (Yes/No). 

---

## Table of Contents
- [Overview](#overview)
- [Technical Workflow](#technical-workflow)
  - [Ingestion](#ingestion)
    - [Ingestion report](#ingestion-report)
  - [Cleaning](#cleaning)
  - [Feature engineering](#feature-engineering)
    - [Splitting the dataset](#splitting-the-dataset)
  - [Modeling and selection](#modeling-and-selection)
    - [Experiment log](#experiment-log)
  - [Evaluation](#evaluation)
    - [Confusion matrices](#confusion-matrices)
    - [Plots](#plots)
    - [Validation ROC AUC vs Regularisation Strength](#validation-roc-auc-vs-regularisation-strength)
- [Summary](#summary)
  - [Summary of model](#summary-of-model)
    - [Limitations of model](#limitations-of-model)
  - [Summary of machine learning pipeline](#summary-of-machine-learning-pipeline)
    - [Limitations-of-machine-learning-pipeline](#limitations-of-machine-learning-pipeline)
- [Appendices](#appendices)
  - [Metadata](#metadata)
  - [File directory](#file-directory)
  - [Cmd prompts](#cmd-prompts)

---

## Overview
The pipeline transforms raw weather records through preprocessing, feature construction, and data partitioning (adaptive sampling) . Models are trained and compared using validation-based selection, with final performance assessed on unseen data and supported by diagnostic metrics and visual analyses.

```
Configuration (YAML)
        ↓
Ingestion → Interim Parquet (audits & missingness)
        ↓
Cleaning → Processed Dataset
        ↓      
Feature Engineering → X_train / X_val / X_test
        ↓       
Training (Validation-based Grid Search)
        ↓       
Evaluation → Metrics + Visual Reports
```

---

## Technical Workflow
This section details each of the key pipeline stages.

---

### Configuration
The parameters used in this pipeline are centralised in ```config.yaml```, such as the file and directory paths, the ratios for training/validation/test splits, the random seed, figure and report output locations.

This was done to remove hard-coded paths and constraints from scripts, ensuring the pipeline can be re-run on different machines or environments without code changes. Further, it can be audited and versioned, as changes to the experiments are explicit in the config file. Finally, using a config file allows changes to be made to the dataset, model, or outputs without breaking the pipeline, only requiring edits to the ```config.yaml``` file.

---

### Ingestion
Raw data is ingested from .CSV files and stored into a parquet intended for processes in the interim, ```weatherAUS_interim.parquet```. This was done so all preprocessing can be done on the interim dataset, leaving the raw dataset unmodified. 

Parquets where chosen as the file format for this pipeline as they are columnar (not row-based which is less efficient for steps in ML pipelines such as feature engineering), and is compressed, which is ideal for large datasets.

After ingestion, the data quality artefacts are generated and detailed below.

---

#### Ingestion report
The ingestion report states the ```WeatherAUS``` data has 14560 rows, 23 columns, 0 duplicates, and dates range from 01/11/2007 to 25/06/2017. Refer to the [Metadata](#metadata) section of the Appendices for the definition and units of each column.

From this report, we can determine that since one column consists of dates, we will need to ensure splitting method that pulls data randomly throughout the time series (and not sequentually). This step is described in section [Splitting the dataset](#splitting-the-dataset).

Further, we know there are no duplicates, so we do not need to include this step in the cleaning process. 

---

### Missingness report (before cleaning)

From looking at the data within the .csv file, it is immediately obvious there are null values, with some variables have consisting of considerable fractions of missing values. The pipeline generates a ```missingness_report_before_cleaning``` to alert us of what proportion of data is missing from each variable (included in the [Missingness report (after cleaning](#missingness-report-(after-cleaning) section of the Appendix).

The most notable variables are sunshine, evaporation, and the cloud data (each above 38% missing values), so these needed to be removed otherwise they will reduce quality of engineered features and affect model performance.

For the remainder of the variables, these will be imputed using the average value of each variable from each station.

---

### Cleaning
The pipeline applies structural cleaning and schema validation on the ```weatherAUS_interim.parquet``` to prepare the data for feature engineering and modelling.

At the column level, any variable with missingness above 38%, namely ```Sunshine```, ```Evaporation```, ```Cloud3pm```, ```Cloud9am```, are removed as they will not suitable for meaningful feature engineering.

The remaining fields are validated, and cast to consistent data types (i.e. numerical, categorical, dates) confirm to a single schema.

At the row level, records with missing or invalid target values are dropped, reducing the row count to ensure all observations have observed outcomes.

The cleaned dataset is saved as ```rainfall_processed.parquet```.

<img width="500" height="450" alt="class_distribution_train" src="https://github.com/user-attachments/assets/ba619870-61bd-49b5-ab29-96e9262368eb" />

---

### Feature engineering
Feature engineering transforms the cleaned dataset into inputs that are model-ready, while preventing leakage by removing variables that are probabiliy multi-collinear with the target variable, i.e. ```Rainfall``` and ```RainToday``` are likely collinear with ```RainTomorrow```, and since the model is trying to predict rainfall from weather data, not necessarily prior rainfall data, these were removed. 

Further, non-predictive fields (```Date```, ```Rainfall```) are excluded.

Categorical variables are then encoded into numeric representations, and missing values in predictor fields are imputed using reproducible strategies defined in configuration. 

No information from the target or future observations is used during transformation. This step converts the human-readable dataset into structured feature matrices suitable for machine learning while ensuring that all modelling assumptions—encoding and column selection—are explicitly documented and repeatable. 

The output of this stage is a numeric representation of the data, which is then split.

---

#### Splitting the dataset
The dataset is partitioned using a chronological split based on the observation date, rather than random sampling. The record is first sorted in ascending order by ```Date```, intended to preserve the natural temporal sequence of weather observations. An assumption I had for the splitting was that ```RainTomorrow``` may be predicted by the weather data more than a single day prior, hence the chronological splitting. 

In praxctice, this split approximates a rolling-forecast type scenario. The model is trained on the first 70% of the dataset, ordered by date, the tuned on a more recent time window, and is then folly assessed on the most recent adn fully unseen last 15%. The training dataset consisted of 99535 rows, with validation and testing having each 21328 and 21330 rows respectively.

The splits are saved as separate Parquet files: ```X_train.parquet```, ```X_val.parquet```, ```X_test.parquet``` for features, and ```y_train.parquet```, ```y_val.parquet```, ```y_test.parquet``` for targets.

---

## Modeling and selection
Logistic Regression was selected as the baseline model due to its interpretability, ability to process on high-dimensional tabular data, and suitability for probabilistic classification, such as predicting rainfall forecasts.

Hyperparameter tuning was automated using a grid search across the regularisation parameter C and the class_weight setting to address class imbalance. 

Each candidate model was trained on the training split and evaluated on a temporally separated validation set to prevent information leakage.

```
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    for cw in [None, "balanced"]:
        model = LogisticRegression(C=C, class_weight=cw)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

Model selection was based primarily on validation ROC-AUC, reflecting the model’s ability to discriminate between rainy and non-rainy days independent of decision threshold. 

Where ROC-AUC values were comparable, ties were resolved using F1-score, Precision, Recall, and Accuracy, in that order, ensuring balanced performance across error types.

The model used is a Logistic Regression as it is easily interpretable, and strong for high-dimensional tabular data. 

---

### Experiment log
The table below details the evaluation for each of the model versions. The best selected model was run ```run 10```, and was retained to ```models/rain_model.pkl```.

| run 	| threshold 	| selection_metric 	| C   	| class_weight 	| penalty 	| roc_auc 	| accuracy 	| precision 	| recall 	| f1    	|
|-----	|-----------	|------------------	|-----	|--------------	|---------	|---------	|----------	|-----------	|--------	|-------	|
| 10  	| 0.5       	| roc_auc          	| 10  	| balanced     	| l2      	| 0.855   	| 0.801    	| 0.508     	| 0.727  	| 0.598 	|
| 8   	| 0.5       	| roc_auc          	| 3   	| balanced     	| l2      	| 0.855   	| 0.801    	| 0.508     	| 0.727  	| 0.598 	|
| 6   	| 0.5       	| roc_auc          	| 1   	| balanced     	| l2      	| 0.855   	| 0.801    	| 0.508     	| 0.727  	| 0.598 	|
| 4   	| 0.5       	| roc_auc          	| 0.3 	| balanced     	| l2      	| 0.855   	| 0.801    	| 0.508     	| 0.727  	| 0.598 	|
| 2   	| 0.5       	| roc_auc          	| 0.1 	| balanced     	| l2      	| 0.855   	| 0.801    	| 0.508     	| 0.727  	| 0.598 	|
| 9   	| 0.5       	| roc_auc          	| 10  	|              	| l2      	| 0.854   	| 0.852    	| 0.722     	| 0.449  	| 0.554 	|
| 7   	| 0.5       	| roc_auc          	| 3   	|              	| l2      	| 0.854   	| 0.852    	| 0.722     	| 0.449  	| 0.554 	|
| 5   	| 0.5       	| roc_auc          	| 1   	|              	| l2      	| 0.854   	| 0.852    	| 0.723     	| 0.449  	| 0.554 	|
| 3   	| 0.5       	| roc_auc          	| 0.3 	|              	| l2      	| 0.854   	| 0.852    	| 0.723     	| 0.449  	| 0.554 	|
| 1   	| 0.5       	| roc_auc          	| 0.1 	|              	| l2      	| 0.854   	| 0.852    	| 0.723     	| 0.449  	| 0.554 	|

---

## Evaluation
The pipeline evaluates the trained model on both the validation and held-out test sets, recording quantitative metrics in metrics.csv and generating diagnostic artefacts to support detailed error analysis and model inspection. Evaluation outputs are intentionally multi-modal, combining tabular metrics with visual tools that characterise decision behaviour across thresholds and operating conditions. Evaluated metrics saved into ```metrics.csv```, confusion matrices, and plots, below.

| Evaluation metric 	| Validation 	| Test  	|
|-------------------	|------------	|-------	|
| Accuracy          	| 0.801      	| 0.775 	|
| Precision         	| 0.508      	| 0.519 	|
| Recall            	| 0.727      	| 0.755 	|
| F1-Score          	| 0.598      	| 0.615 	|
| ROC AUC           	| 0.855      	| 0.849 	|

---

### Confusion matrices
The pipeline evaluates the trained model on both the validation and held-out test sets, recording quantitative metrics in metrics.csv and generating diagnostic artefacts to support detailed error analysis and model inspection.

|Validation|Testing| 
|----------|----------|
<img width="500" height="450" alt="confusion_matrix_val" src="https://github.com/user-attachments/assets/a37e39ad-3ace-49a0-bdf6-7743544c3978" /> | <img width="500" height="450" alt="confusion_matrix" src="https://github.com/user-attachments/assets/3b45c1b5-0de9-4ce6-a6ac-4983e7fedeb5" />  |

---

### Plots
The Precision–Recall curve visualises the trade-off between positive predictive value and sensitivity across all decision thresholds, offering insight into classifier behaviour under varying operating points. 

This representation is included to support threshold selection and to examine performance in class-imbalanced conditions where accuracy alone may be insufficient.

The ROC curve illustrates the relationship between true positive and false positive rates over all probability thresholds, providing a threshold-independent view of class separability. 

It is included to assess the model’s ranking ability and discrimination characteristics.

|  Precision-Recall | ROC Curve   |
|----------|----------|
<img width="500" height="450" alt="pr_curve_test" src="https://github.com/user-attachments/assets/a44d332a-686c-4b07-aec8-8d8aa051c116" /> | <img width="500" height="450" alt="roc_curve_test" src="https://github.com/user-attachments/assets/2095c2ce-23aa-46ba-ada4-ca9ea6929105" />

---

### Validation ROC AUC vs Regularisation Strength
Finally, the validation ROC-AUC vs regularisation strength plot documents the hyperparameter search process. 

It shows how changes in the regularisation parameter affect validation performance, serving as both an audit trail for model selection and a diagnostic of bias–variance behaviour across model complexity.

<img width="500" height="450" alt="hyperparameter_curve_logreg" src="https://github.com/user-attachments/assets/b8cc435c-0714-447f-981a-89fd88fa3794" />

---

## Summary
The evaluation artefacts show that the model achieves strong class separation, with ROC AUC values near 0.85 indicating reliable ranking between rainy and non-rainy days. 

The decision behaviour prioritises sensitivity to rainfall events, capturing most positive cases while accepting a moderate level of false positives. 

Performance remains consistent across validation and held-out test splits, supporting the stability of the modelling approach under the applied time-aware partitioning. 

Together, the metrics and visual diagnostics demonstrate that the system provides a dependable probabilistic signal for rainfall prediction within the scope of the available features and modelling assumptions.

---

### Summary of model
The final model is a regularised logistic regression trained on a high-dimensional feature set derived from cleaned and encoded meteorological variables. 

It produces calibrated probability estimates and supports transparent analysis of feature effects. 

Model selection was conducted using validation-based hyperparameter tuning, with experiment results logged to enable reproducibility and auditability. 

Evaluation outputs include confusion matrices and threshold-agnostic curves, providing insight into classification behaviour across operating points. 

The chosen configuration represents a stable baseline suitable for both interpretability and future comparative experimentation.

---

### Limitations of model
The final model is a regularised logistic regression trained on a high-dimensional feature set derived from cleaned and encoded meteorological variables. 

It produces calibrated probability estimates and supports transparent analysis of feature effects. Model selection was conducted using validation-based hyperparameter tuning, with experiment results logged to enable reproducibility and auditability. 

Evaluation outputs include confusion matrices and threshold-agnostic curves, providing insight into classification behaviour across operating points. 

The chosen configuration represents a stable baseline suitable for both interpretability and future comparative experimentation.

---

## Summary of machine learning pipeline
he pipeline provides a complete workflow from raw data ingestion through cleaning, feature construction, model training, and evaluation. Each stage produces persistent artefacts, enabling traceability across data transformations, experimental runs, and final outputs. 

Configuration is externalised, allowing consistent execution across environments while maintaining reproducibility.

Automated reporting ensures that quantitative metrics and visual diagnostics are generated systematically, supporting transparent model assessment and iterative development.

---

### Limitations of machine learning pipeline
The pipeline provides a complete workflow from raw data ingestion through cleaning, feature construction, model training, and evaluation. Each stage produces persistent artefacts, enabling traceability across data transformations, experimental runs, and final outputs. 

Configuration is externalised, allowing consistent execution across environments while maintaining reproducibility. 

Automated reporting ensures that quantitative metrics and visual diagnostics are generated systematically, supporting transparent model assessment and iterative development.

---

## Appendices
### Metadata
| Metric        	| Unit               	| Description                                                         	|
|---------------	|--------------------	|---------------------------------------------------------------------	|
| Date          	| YYYY-MM-DD         	| The date of observation.                                            	|
| Location      	| [string]           	| The common name of the location of the weather station.             	|
| MinTemp       	| Celcius            	| The minimum temperature.                                            	|
| MaxTemp       	| Celcius            	| The maximum temperature.                                            	|
| Rainfall      	| mm                 	| The amount of rainfall recorded for the day.                        	|
| Evaporation   	| mm                 	| Class A pan evaporation in 24 hours prior to 9am.                   	|
| Sunshine      	| hours              	| Length of time of bright sunshine in the day.                       	|
| WindGustDir   	| Cardinal direction 	| Direction of strongest wind gust in 24 hours prior to midnight.     	|
| WindGustSpeed 	| km/h               	| Speed of the strongest wind gust in the 24 hours prior to midnight. 	|
| WindDir9am    	| Cardinal direction 	| Direction of the wind at 9am.                                       	|
| RainToday     	| Yes/No             	| Whether or not it had rained.                                       	|
| RainTomorrow  	| Yes/No             	| The target variable. Is it expected to rain the next day?           	|

---
### Missingness report (before cleaning)

| column        	| missing_fraction 	|
|---------------	|------------------	|
| Sunshine      	| 0.48             	|
| Evaporation   	| 0.43             	|
| Cloud3pm      	| 0.41             	|
| Cloud9am      	| 0.38             	|
| Pressure9am   	| 0.10             	|
| Pressure3pm   	| 0.10             	|
| WindDir9am    	| 0.07             	|
| WindGustDir   	| 0.07             	|
| WindGustSpeed 	| 0.07             	|
| Humidity3pm   	| 0.03             	|
| WindDir3pm    	| 0.03             	|
| Temp3pm       	| 0.02             	|
| RainTomorrow  	| 0.02             	|
| Rainfall      	| 0.02             	|
| RainToday     	| 0.02             	|
| WindSpeed3pm  	| 0.02             	|
| Humidity9am   	| 0.02             	|
| WindSpeed9am  	| 0.01             	|
| Temp9am       	| 0.01             	|
| MinTemp       	| 0.01             	|
| MaxTemp       	| 0.01             	|
| Date          	| 0                	|
| Location      	| 0                	|

---

### File directory
```
Rain Predictor/
│
├── config.yaml
│
├── data/
│   ├── raw/            # Original dataset (CSV)
│   ├── interim/        # Post-ingestion (Parquet)
│   └── processed/      # Cleaned data + model-ready features (Parquet)
│
├── src/
│   ├── utils/          # Config loading
│   ├── data/           # ingest.py, clean.py
│   ├── features/       # build_features.py
│   ├── models/         # train.py, evaluate.py
│   └── reports/        # figure generation & dashboard
│
├── models/          # Persisted model artefacts
│
└── reports/
    ├── tables/         # Metrics, logs, audits
    └── figures/        # Visual diagnostics
```

---

### Cmd prompts
```
python -m src.data.ingest
python -m src.data.clean
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
python -m src.reports.make_figures
python -m src.reports.make_dashboard
```
