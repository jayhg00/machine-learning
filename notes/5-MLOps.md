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
