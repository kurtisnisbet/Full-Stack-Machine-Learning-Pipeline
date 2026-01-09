# Australian Rainfall Predictor Model (2008-2017)

This repository contains a reproducible end-to-end machine learning pipeline to predict whether it will rain tomorrow using the Australian Weather data. The pipeline is full-stack, containing data ingestion, cleaning, feature engineering, time-aware splitting, model selection via validation,a dn final evaluation on a held-out test set. 

This repository details the key steps of the pipeline, accompanied with key visualisations and plots assessing the dataset and model performance.

The data source is the Australian Weather Dataset (WeatherAUS from Kaggle), with a target variable of RainTomorrow (Yes/No). 

The data are tabular, time-series weather observations taken from multiple weather stations across the country. 


---

## Table of Contents
- [Overview](#overview)
- [Technical Workflow](#technical-workflow)
  - Ingestion
  - Cleaning
  - Feature engineering
  - Modeling and selection
  - Evaluation
- Results
- Limitations and next steps
- References
- Appendix

## Overview

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

## Technical Workflow

### Configuration
The parameters used in this pipeline are centralised in ```config.yaml```, such as the file and directory paths, the ratios for training/validation/test splits, the random seed, figure and report output locations.

This was done to remove hard-coded paths and constraints from scripts, ensuring the pipeline can be re=run on different machines or environments without code changes. Further, it can be audited and versioned, as changes to the experiments are explicit in the config file. Finally, easily allows additions without breaking the pipeline, such as new datasets, models, and outputs, only requiring edits to the config.yaml file.

### Ingestion
Raw data ingested from .CSV files, and writes to an interim Parquet. 

# NOTE TO SELF: EXPLAIN WHY.

Data quality artefacts are generated, detailed below.

#### Ingestion report
Found the dataset has 14560 rows, 23 columns, 0 duplicates, and dates range from 01/11/2007 to 25/06/2017.

From this report, we know we need to ensure splitting the dataset for training, testing, and evaluation need to be randomly selected from dates throughout the time series, and are not sequential.

Further, we know there are no duplicates, so we do not need to include this step in the cleaning process.

### Missingness report (before cleaning)

Some variables have considerable fractions of missing values, namely sunshine, evaporation, and the cloud data (each above 38%). These variabeles will need to be removed, as they will reduce quality of engineered features and affect model performance.

For the remainder of the variables, these will be imputed, using the average value of each variable from each station.

The fraction of missing values for each variable:

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


### Cleaning
The pipeline generates a ```cleaning_report.csv``` to catalogue the changes to the interim dataset. We can see that after cleaning, the number of rows were reduced from 145,460 to 142,193, and the number of columns were dropped from 23 to 19 (accounting for the removal of ```Sunshine```, ```Evaporation```, ```Cloud3pm```, ```Cloud9am```. 

By doing this, we ensure valid date parking, and completeness in the data to train the classifier to predict ```RainTomorrow```.

The cleaned data were saved in ```rainfall_processed.parquet```. Further, ```missingness_after_cleaning.csv``` is also generated, shown below.

# NOTE TO SELF:WHY ARE THERE STILL MISSING VALUES

| column        	| missing_fraction 	|
|---------------	|------------------	|
| Pressure9am   	| 0.099            	|
| Pressure3pm   	| 0.098            	|
| WindDir9am    	| 0.070            	|
| WindGustDir   	| 0.066            	|
| WindGustSpeed 	| 0.065            	|
| WindDir3pm    	| 0.027            	|
| Humidity3pm   	| 0.025            	|
| Temp3pm       	| 0.019            	|
| WindSpeed3pm  	| 0.018            	|
| Humidity9am   	| 0.012            	|
| Rainfall      	| 0.010            	|
| RainToday     	| 0.010            	|
| WindSpeed9am  	| 0.009            	|
| Temp9am       	| 0.006            	|
| MinTemp       	| 0.004            	|
| MaxTemp       	| 0.002            	|
| Date          	| 0                	|
| Location      	| 0                	|
| RainTomorrow  	| 0                	|

### Feature engineering
As mentioned above, RainTomorrow is the target variable.

Features are *only* created from the train dataset. Further, RainToday is also removed to prevent same-day leaking for next-day prediction. Date is also removed. 

The pipeline encodes categoricals (one-hot) and imputes the remaining missing variables. 

#### Splitting the dataset
The pipeline performs a time-based split (i.e. to prevent sequential splitting). 70% of the dataset is  dedicated to training the model, with 15% each dedicated to validation and testing.

The dataset is split into the following parquets:
- X_train.parquet, X_val.parquet, X_test.parquet
- y_train.parquet, y_val.parquet, y_test.parquet
- feature_build_report.csv

From ```feature_build_report.csv```, the pipeline has engineered 114 features.

Further, these features will be trained on 99535 rows of training data, 21328 rows of validation rate, and 21330 rows of testing data.

## Modelling and selection
The model used is a Logistic Regression as it is easily interpretable, and strong for high-dimensional tabular data. 

Hyperparameter tuning was automated, using a grid search over ```C``` (regularisation strength), and ```class_weight``` for imbalance handling.

```
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    for cw in [None, "balanced"]:
        model = LogisticRegression(C=C, class_weight=cw)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

The selection criterion was validation ROC-AUC, with tie breakers being settled by F1, Precision, Recall, and Accuracy in descending order of priority.


### Experiment log
Hyperparameter tuning experimentation is saved in ```model_select_logreg.csv```, displayed below.
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

The best selected model was run 10, and was retained to ```models/rain_model.pkl```.

## Training performance of best model
True predictions, shown in the confusion matrix below.

<img width="1077" height="908" alt="confusion_matrix_val" src="https://github.com/user-attachments/assets/a37e39ad-3ace-49a0-bdf6-7743544c3978" />

| split 	| rows  	| accuracy 	| precision 	| recall   	| f1       	| roc_auc  	| threshold 	| target_col   	|
|-------	|-------	|----------	|-----------	|----------	|----------	|----------	|-----------	|--------------	|
| val   	| 21328 	| 0.800685 	| 0.508355  	| 0.726521 	| 0.598166 	| 0.855375 	| 0.5       	| RainTomorrow 	|
| test  	| 21330 	| 0.775199 	| 0.518488  	| 0.755328 	| 0.61489  	| 0.848566 	| 0.5       	| RainTomorrow 	|

## Evaluation
The pipeline evaluations on the validation and held-out test sets, outputting the metrics into ```metrics.csv```, confusion matrices, and plots. All included below.

| Evaluation metric 	| Validation 	| Test  	|
|-------------------	|------------	|-------	|
| Accuracy          	| 0.801      	| 0.775 	|
| Precision         	| 0.508      	| 0.519 	|
| Recall            	| 0.727      	| 0.755 	|
| F1-Score          	| 0.598      	| 0.615 	|
| ROC AUC           	| 0.855      	| 0.849 	|

### Training
<img width="1179" height="908" alt="class_distribution_train" src="https://github.com/user-attachments/assets/ba619870-61bd-49b5-ab29-96e9262368eb" />

### Validation
<img width="1077" height="908" alt="confusion_matrix_val" src="https://github.com/user-attachments/assets/a8443892-6447-4b97-ade8-339554f1fa90" />

### Test
<img width="1077" height="908" alt="confusion_matrix" src="https://github.com/user-attachments/assets/3b45c1b5-0de9-4ce6-a6ac-4983e7fedeb5" />

<img width="1134" height="908" alt="pr_curve_test" src="https://github.com/user-attachments/assets/a44d332a-686c-4b07-aec8-8d8aa051c116" />

<img width="1134" height="908" alt="roc_curve_test" src="https://github.com/user-attachments/assets/2095c2ce-23aa-46ba-ada4-ca9ea6929105" />

# Summary of model
From the above metrics, visualisation, and plots we can see there is a strong discirmination due to the ROC AUC ~0.85, indicating reliable ranking of rainy vs non-rainy days. Further, most rainy days are detected, at the cost of moderate false-positives, as the models were trained to be more recall orientated. Finally, close validation and test performance confirms stable generalisation under the current time-aware splitting regime.

## Limitations of model
Model class of logistic regression likely under-captures non-linear relationships. Adding in a non-linear component, such as a neural network, would address this. Further optimisation regarding the threshold is needed, as it was fixed at 0.5 for this model.
The feature generation and splitting was deterministic, using other methods of splitting should be trialled, such as 
# Enter examples above

# Summary of machine learning pipeline

## Limitations of machine learning pipeline

## Appendix
### Metadata
Metadata
Date - The date of observation
Location - The common name of the location of the weather station
Unique identifier: date + location
MinTemp - The minimum temperature in degrees celsius
MaxTemp - The maximum temperature in degrees celsius
Rainfall - The amount of rainfall recorded for the day in mm
Evaporation - The so-called Class A pan evaporation (mm) in the 24 hours to 9am
Sunshine - The number of hours of bright sunshine in the day.
WindGustDir - The direction of the strongest wind gust in the 24 hours to midnight
WindGustSpeed - The speed (km/h) of the strongest wind gust in the 24 hours to midnight
WindDir9am - Direction of the wind at 9am
RainToday - If rainfall is not 0mm.
RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.


### File directory
```
Rain Predictor/
│
├── 02_config.yaml
│
├── 04_data/
│   ├── raw/            # Original dataset (CSV)
│   ├── interim/        # Post-ingestion (Parquet)
│   └── processed/      # Cleaned data + model-ready features
│
├── 06_src/
│   ├── utils/          # Config loading
│   ├── data/           # ingest.py, clean.py
│   ├── features/       # build_features.py
│   ├── models/         # train.py, evaluate.py
│   └── reports/        # figure generation & dashboard
│
├── 07_models/          # Persisted model artefacts
│
└── 08_reports/
    ├── tables/         # Metrics, logs, audits
    └── figures/        # Visual diagnostics
```

### cmd prompts to run pipeline
```
python -m src.data.ingest
python -m src.data.clean
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
python -m src.reports.make_figures
python -m src.reports.make_dashboard
```
