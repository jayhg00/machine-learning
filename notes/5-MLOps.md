# MLOps - DevOps for Machine Learning
MLOps Zoomcamp by Alexey Grigorev

https://github.com/DataTalksClub/mlops-zoomcamp

https://www.youtube.com/watch?v=s0uaFZSzwfI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=1

## Introduction
- Broadly, MLOps process is **DESIGN --> TRAIN --> OPERATE**. And this is iterative
- **DESIGN** Stage-
  - Data **collection**/ingestion
  - Data cleaning
  - **Feature Engineering**
  - Prepare Training, Validation & Test Datasets
- **TRAIN** Stage-
  - Model **Development** and Training: Experimenting with different algorithms, tuning hyperparameters, and training models, typically using tools like Jupyter notebooks.
  - Model Evaluation and Validation: Assessing the model on test data to ensure it meets performance requirements, is unbiased, and is ready for production.
  - Model **Registry** and Versioning: Storing approved model artifacts, along with metadata (code, datasets, parameters), in a registry to ensure traceability and reproducibility.
- **OPERATE** Stage-
  - Model **Deployment** (CI/CD): Deploying the model as a service (e.g., REST API) to production environments using automated pipelines (e.g., Docker, Kubernetes).
  - Model **Monitoring and Observation**: Tracking model performance in production to detect data drift, concept drift, model drift and performance degradation.

- Key aspects of MLOPs are-
  1. Experiment tracking - When you train a model, you adjust data, hyperparameters etc and capture some metrics. If you change some hyperparameter and metric degrades, you want to be able to revert to previous state. Manually noting so many parameters is error-prone. It is a systematic process of logging and organizing all metadata—code, hyperparameters, datasets, and metrics—**during model training to ensure reproducibility, facilitate comparisons, and manage iterative improvements**
  2. Model Registry & Versioning - For any model in production, you should be able to trace back to how it was trained, with which data, and by whom. So, you store all versions of a Model in a Model Registry. Also, you tag the Production state of the models - none, Staging, Production, Archived
  3. Model deployment via CI/CD
  4. Model monitoring & observability - When model is in Production, it needs to be monitored for any degradation in performance. If the data distribution has changed during inference from that during training, then it leads to Model drift. And the model needs to be re-trained and new version needs to be deployed

### MLOps Maturity Model
- This whole lifecycle of ML from Data Collection to Monitoring can be implemented in parts either manually or automated. For example, if the monitoring process detects a drop in model performance, it can send an alert to the ML team. The ML team can respond to the alert and fix the issue by re-training the model and re-deploying Version 2. In another scenario, the human loop of alert can be removed where an automated system will re-train the model and deploy version 2. These are different maturity models.
- 5 Levels - 0 (No MLOPS) to 4 (Full MLOPS)
- Level **0 - No MLOPS**
  - All tasks are manual (Training, Build, Test, Deployment)
  - Code resides in Jupyter notebook and cells need to be executed in specific order
  - Good for POC
- Level **1 - DevOps, but no MLOps**
  - CI/CD for automated unit tests, integration tests and Releases
  - No experiment tracking
  - No reproducibility
  - POC going to Production
- Level **2 - AUTOMATED TRAINING**
  - You have a train.py which you can simply run ==> TRAINING PIPELINE
  - Experiment tracking
  - Model Registry
  - Manual deployment which is easy
- Level **3 - AUTOMATED DEPLOYMENT**
  - Automatic training
  - Easy to deploy
  - A/B Tests - V1 & V2 of model reside side-by-side to compare performance. Some requests go to V1 and some go to V2
  - Model monitoring
- Level **4 - Full MLOps**
  - Automated Training
  - Automated Deployment
  - Automated Re-training & re-deployment without any human intervention
- Not all model use-cases need to be at Level 4. It all depends on the importance of the use-case and model

## Experiment Tracking
It is the process of keeping track of all the relevant information from an ML experiment, which includes:

    - Source code
    - Environment
    - Data
    - Model
    - Hyperparameters
    - Metrics, etc
    
Key terms-
- ML experiment: the process of building an ML model
- Experiment run: each trial in an ML experiment
- Run artifact: any file that is associated with an ML run
- Experiment metadata: information about an ML experiment like the source code used, the name of the user, etc.
- Experiment tracking is important because:-
  - Reproducibility
  - Organization
  - Optimization
- Manually tracking experiment-related parameters in Excel is cumbersome and error-prone


## MLFlow (open-source)
- MLFlow is open-source, Python package installed with pip and contains following Modules:
  - Experiment Tracking
  - Models
  - Model Registry
  - Projects
- Paid alternatives are Neptune, Comet, Weights & Biases

### Experiment Tracking with MlFlow
- The MLflow Tracking module allows you to organize your **experiments** into **runs**, and each run tracks the following:
  - Parameters
  - Metrics
  - Metadata
  - Artifacts
  - Models
- Along with above information, MLflow automatically logs extra information about the run:
  - Source code
  - Version of the code (git commit)
  - Start and end time
  - Author

- MLFlow uses a client-server framework. It also provides a **Gunicorn web UI to view the info on the server**
- **Tracking server runs on localhost:5000 or remote:5000**. Our Jupyter notebook/Python script code or MLFlowClient API calls methods to manage Experiment Tracking or Model Registry
  - All runs and associated info will be stored either in-
    - Localsystem (under ./mlruns folder where .ipynb/.py file resides) Default, if no backend store is specified for the server
    - SQL compatible database (SQLite, PostGreSQL etc)
  - For Model Registry, backend store is to be specified. The backend stores the metadata about the model in the Model Registry while the actual Model binaries (.pkl, .h5, .keras etc) can be stored either in-
    - Localsystem (under ./models folder where .ipynb/.py file resides)
    - Remote (AWS S3 bucket, Azure Blob Container)
  - The location of mlruns & inside mlruns file structure for localsystem is like below-
    
<kbd><img width="500" alt="image" src="https://github.com/user-attachments/assets/93a4c362-51e3-4072-9800-b5f471a51350" /></kbd> 
<kbd><img width="374" alt="image" src="https://github.com/user-attachments/assets/abf2bf81-bd45-4813-aa40-74b8caf2f3c1" /></kbd>

#### Installing MLFlow
```Python
$> pip install mlflow
```

#### Running the MLFlow UI (run from command prompt) with SQLite backend store. This starts the Gunicorn web UI on localhost:5000/. Open the URL localhost:5000 on browser
```
$> mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### Adding MLFlow to jupyter notebook
```Python
# Other imports

import mlflow

## Set the TRACKING URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

## COde to prepare X_train, X_val etc

## Wrap the Training/Prediction as a "Run"
with mlflow.start_run():
    # SET Tag
    mlflow.set_tag("developer", "cristian")

    # Log some parameters
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    alpha = 0.1
    mlflow.log_param("alpha", alpha)  ## Log the learning rate alpha
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)  ## Log the RMSE metric

    ## Log the model as an artifact
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
```
MLFlow creates an experiment if it does not exist. Everytime you run the cell of "mlflow.start_run()" with different values of alpha or the train-data-path/valid-data-path, MLFlow will create a run in that experiment and log the specified data.

#### Viewing the experiment thru UI
Run the MLFlow UI Cmd from cmd prompt and navigate to localhost:5000/
<kbd><img width="1200" alt="image" src="https://github.com/user-attachments/assets/f35810cb-93d9-49be-b419-f0d75b8ecb12" /></kbd>

Open the Run to see its data alongwith Model Artifacts
<kbd><img width="1200" alt="image" src="https://github.com/user-attachments/assets/06d4076c-8bfe-423f-9bbf-4327fd183ed6" /></kbd>
<kbd><img width="1200" alt="image" src="https://github.com/user-attachments/assets/9ef3bc31-3b72-42fd-878b-6287e19b37dd" /></kbd>


#### Autologging
call mlflow.autolog() before your training code. This will log for all libraries
```Python
import mlflow

mlflow.autolog()
with mlflow.start_run():
    # your training code goes here
    ...
```

  Enable/Disable Autologging for specific libraries. For example, if you train your model on PyTorch but use scikit-learn for data preprocessing, you may want to disable autologging for scikit-learn while keeping it enabled for PyTorch. You can achieve this by either (1) enable autologging only for PyTorch using PyTorch flavor (2) disable autologging for scikit-learn using its flavor with disable=True.
  ```Python
  import mlflow
  
  # Option 1: Enable autologging only for PyTorch
  mlflow.pytorch.autolog()
  
  # Option 2: Disable autologging for scikit-learn, but enable it for other libraries
  mlflow.sklearn.autolog(disable=True)
  mlflow.autolog()
  ```
