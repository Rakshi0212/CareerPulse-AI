# 🚀 CareerPulse AI — Tech Career Skill Gap & Salary Valuation Platform
https://careerpulse-ai-qmvh.onrender.com

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/UnitTests-5%2F5%20Passing-brightgreen.svg)](tests/test_pipeline.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end Data Science and Machine Learning platform built for **Students, Fresh College Graduates, and Tech Job Seekers** to accurately price their market salary valuation (₹ LPA), evaluate job readiness tiers, pinpoint high-ROI missing skills, and unlock personalized career roadmaps.

---

## 🎯 Who is the User of this Project?

### Primary User: The Tech Student & Job Seeker
- **The Problem**: 
  - *"What salary package can I realistically demand with my current projects and skills?"*
  - *"If I want to hit a ₹15 LPA or ₹25 LPA package, which missing skills give me the highest salary jump?"*
  - *"Am I ready for an AI/ML Engineer, Full-Stack, or DevOps role?"*
  - *"Does solving 150+ LeetCode problems actually compress the college tier gap?"*
- **How They Use CareerPulse**:
  1. Input current college tier, degree, verified tech skills (Python, Cloud, PyTorch, Docker, etc.), and coding stats (LeetCode solved, GitHub projects).
  2. The ML model outputs their **Predicted Market Valuation (e.g., ₹14.8 LPA)**, their **Readiness Tier (`Job-Ready Mid-Level`)**, and their market percentile.
  3. They select a target dream role (e.g. *AI / Machine Learning Engineer*) to see their **Role Match %**, **Missing Critical Skills**, and the **Exact Salary ROI of each missing skill** (e.g., *“Learning Docker & AWS adds an estimated +₹2.6 LPA to your compensation”*).

---

## 🌟 The Complete Data Science Lifecycle

```
[Raw Tech Applicant Profiles (3,500 Records)]
                     │
                     ▼
[Candidate Data Cleaner] ────────► Imputes Missing Coding Stats,
                     │             Categorical Normalization, Sanity Bounds
                     ▼
[Tech Market Feature Store] ─────► Coding Intensity Index (DSA + GitHub + Internships),
                     │             Specialization Depths, College Tier Scores
                     ▼
    ┌────────────────┼────────────────┬────────────────┐
    ▼                ▼                ▼                ▼
[Model 1: Regressor] [Model 2: Classifier] [Model 3: Phenotypes] [Model 4: Anomaly]
Gradient Boosting    Gradient Boosting    K-Means & PCA (2D/3D) Isolation Forest
Market Salary (LPA)  Readiness Tier       4 Talent Archetypes   Non-traditional Profiles
R² = 0.9712          Accuracy: 87.57%     Unsupervised Mapping  3.0% Outliers Flagged
MAE = 0.85 LPA       Weighted F1: 0.875
    │                │                │                │
    └────────────────┼────────────────┴────────────────┘
                     ▼
[High-ROI Skill Gap Engine] ─────► Prioritizes Missing Skills by Salary Increment (LPA)
                     │
                     ▼
[Interactive Web Dashboard] ─────► Real-Time Valuation Calculator, Skill Roadmaps,
(Streamlit: Port 8503)             Market Analytics & Talent PCA Clusters
```

---

## 🤖 Multi-Model Machine Learning Benchmarks

| Model Component | Algorithm | Objective | Key Metric | Benchmark Score |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: Salary Regressor** | Gradient Boosting Regressor | Continuous salary package valuation (₹ LPA) | **$R^2$ Score / MAE** | **$R^2 = 0.9712$**, **MAE: 0.85 LPA** |
| **Model 2: Readiness Classifier** | Gradient Boosting Classifier | Employability tier (`Entry`, `Junior`, `Mid-Level`, `Specialist`) | **Accuracy / Weighted F1** | **87.57% Accuracy**, **0.8750 F1** |
| **Model 3: Talent Archetypes** | K-Means ($k=4$) + PCA | Unsupervised talent segmentation across 22 career variables | **Explained Variance** | **4 Industry Archetypes** |
| **Model 4: Profile Outliers** | Isolation Forest | Zero-day detection of non-traditional candidate profiles | **Contamination Rate** | **3.0% Market Anomaly Rate** |

---

## 📁 Repository Directory Structure

```
CareerPulse/
├── data/
│   ├── raw_candidates_market.csv         # 3,500 raw tech candidate profiles
│   └── processed_candidates_market.csv   # Cleaned profiles with 22 engineered features & clusters
├── src/
│   ├── data_generator.py                 # Calibrated candidate market generation engine
│   ├── preprocessing/
│   │   └── data_cleaner.py               # Missing value imputer & bounds check
│   ├── features/
│   │   └── engineering.py                # Coding intensity & domain depth biomarker store
│   ├── models/
│   │   └── train.py                      # Multi-model training, evaluation & serialization
│   └── skill_analyzer/
│       └── analyzer.py                   # Role readiness matcher & high-ROI skill roadmap
├── models/
│   └── saved_models/
│       ├── salary_regressor.joblib       # Serialized Model 1
│       ├── readiness_classifier.joblib   # Serialized Model 2
│       ├── talent_clusterer.joblib       # Serialized Model 3
│       ├── pca_transformer.joblib        # PCA 2D/3D projection weights
│       ├── feature_scaler.joblib         # Standard normalizer
│       ├── anomaly_detector.joblib       # Serialized Model 4
│       ├── label_encoder.joblib          # Tier label encoder
│       └── metrics.json                  # Performance benchmark summary
├── dashboard/
│   └── app.py                            # Interactive Streamlit candidate web application
├── notebooks/
│   └── career_market_eda.ipynb           # Interactive EDA & benchmarking notebook
├── tests/
│   └── test_pipeline.py                  # Automated test suite (5/5 passing)
├── run_pipeline.py                       # One-click pipeline execution
├── run_dashboard.py                      # Launches Streamlit dashboard on port 8503
├── run_dashboard.bat                     # Windows batch launcher
├── requirements.txt                      # Project dependency specification
└── README.md                             # Comprehensive project documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Run the Complete Data Science Pipeline
```bash
python run_pipeline.py
```

### 2. Run the Automated Unit Test Suite
```bash
python -m unittest tests/test_pipeline.py
```
*(All 5 unit tests pass in 0.10s).*

### 3. Launch the Interactive Web Dashboard
```bash
python run_dashboard.py
```
Open **`http://localhost:8503`** in your browser to explore:
- **Student Valuation & Readiness**: Live sliders for experience, college tier, LeetCode problems, and tech skills with real-time LPA package predictions.
- **Skill Gap & High-ROI Roadmap**: See missing competencies ranked by exact salary boost potential for target roles.
- **Job Market Analytics**: Salary curves by experience, college tier box plots, and skill premiums.
- **Talent Archetypes (PCA)**: 2D interactive PCA visual showing where your profile sits in the tech market.
- **Model Governance**: Confusion matrix and global salary driver importance weights.
