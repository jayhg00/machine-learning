# MLOps - DevOps for Machine Learning
MLOps Zoomcamp by Alexey Grigorev

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
