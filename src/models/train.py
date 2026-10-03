"""
CareerPulse Multi-Model Training & Benchmarking Pipeline
========================================================
Trains and evaluates:
1. Continuous Market Salary Valuation Regressor (Gradient Boosting Regressor)
2. Employability Readiness Tier Classifier (Gradient Boosting Classifier)
3. Unsupervised Talent Archetype Clusterer (K-Means & PCA)
4. Outlier Profile Detector (Isolation Forest)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestClassifier, IsolationForest
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from src.data_generator import generate_candidate_data
from src.preprocessing.data_cleaner import CandidateDataCleaner
from src.features.engineering import CandidateFeatureEngineer


FEATURE_COLUMNS = [
    "experience_years",
    "internships_count",
    "github_projects",
    "dsa_problems_solved",
    "certifications_count",
    "has_python",
    "has_sql",
    "has_react_node",
    "has_cloud_aws",
    "has_docker_k8s",
    "has_ml_pytorch",
    "has_system_design",
    "coding_intensity_index",
    "cloud_devops_depth",
    "ai_data_depth",
    "fullstack_systems_depth",
    "college_tier_score",
    "education_score",
    "loc_tier1_hub",
    "loc_remote",
    "skill_breadth",
    "competitiveness_score"
]


def run_pipeline() -> Dict[str, Any]:
    print("=" * 65)
    print("[CareerPulse] Starting Tech Career & Salary Valuation Pipeline")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "data")
    models_dir = os.path.join(base_dir, "..", "models", "saved_models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load or Generate Candidate Dataset
    raw_path = os.path.join(data_dir, "raw_candidates_market.csv")
    if not os.path.exists(raw_path):
        print("Generating synthetic candidate market cohort (3,500 profiles)...")
        raw_df = generate_candidate_data(n_samples=3500, random_state=42)
        raw_df.to_csv(raw_path, index=False)
    else:
        print(f"Loading candidate market cohort from {raw_path}...")
        raw_df = pd.read_csv(raw_path)

    # 2. Preprocess & Clean
    print("Executing Candidate Data Cleaner...")
    cleaner = CandidateDataCleaner()
    clean_df = cleaner.fit_transform(raw_df)

    # 3. Feature Engineering
    print("Engineering Tech Market & Coding Intensity Biomarkers...")
    engineer = CandidateFeatureEngineer()
    processed_df = engineer.transform(clean_df)
    processed_path = os.path.join(data_dir, "processed_candidates_market.csv")
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed candidate market data saved to {processed_path}")

    # 4. Train / Test Split
    X = processed_df[FEATURE_COLUMNS]
    y_salary = processed_df["salary_lpa"]
    y_tier = processed_df["readiness_tier"]

    tier_order = [
        "Entry-Level Learner",
        "Junior Associate",
        "Job-Ready Mid-Level",
        "High-Demand Specialist"
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(tier_order)
    y_tier_enc = label_encoder.transform(y_tier)

    X_train, X_test, y_sal_train, y_sal_test, y_tier_train, y_tier_test = train_test_split(
        X, y_salary, y_tier_enc, test_size=0.20, random_state=42, stratify=y_tier_enc
    )

    print(f"Training Profiles: {len(X_train)} | Test Profiles: {len(X_test)}")

    # 5. Model 1: Continuous Market Salary Valuation Regressor (LPA)
    print("\n--- Model 1: Market Salary Valuation Regressor (LPA) ---")
    gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    gbr.fit(X_train, y_sal_train)
    gbr_pred = gbr.predict(X_test)
    gbr_r2 = r2_score(y_sal_test, gbr_pred)
    gbr_rmse = float(np.sqrt(mean_squared_error(y_sal_test, gbr_pred)))
    gbr_mae = mean_absolute_error(y_sal_test, gbr_pred)

    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_sal_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_sal_test, ridge_pred)

    print(f"Gradient Boosting Regressor -> R2: {gbr_r2:.4f} | RMSE: {gbr_rmse:.2f} LPA | MAE: {gbr_mae:.2f} LPA")
    print(f"Ridge Regressor             -> R2: {ridge_r2:.4f}")
    best_reg = gbr

    # 6. Model 2: Employability Readiness Tier Classifier
    print("\n--- Model 2: Employability Readiness Tier Classifier ---")
    gb_clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.09, max_depth=4, random_state=42)
    gb_clf.fit(X_train, y_tier_train)
    gb_pred = gb_clf.predict(X_test)
    gb_acc = accuracy_score(y_tier_test, gb_pred)
    gb_f1 = f1_score(y_tier_test, gb_pred, average="weighted")

    rf_clf = RandomForestClassifier(n_estimators=160, max_depth=10, random_state=42)
    rf_clf.fit(X_train, y_tier_train)
    rf_pred = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_tier_test, rf_pred)
    rf_f1 = f1_score(y_tier_test, rf_pred, average="weighted")

    print(f"Gradient Boosting Classifier -> Accuracy: {gb_acc:.4f} | Weighted F1: {gb_f1:.4f}")
    print(f"Random Forest Classifier   -> Accuracy: {rf_acc:.4f} | Weighted F1: {rf_f1:.4f}")

    if gb_f1 >= rf_f1:
        best_clf = gb_clf
        clf_name = "GradientBoostingClassifier"
        clf_acc = gb_acc
        clf_f1 = gb_f1
        clf_pred = gb_pred
    else:
        best_clf = rf_clf
        clf_name = "RandomForestClassifier"
        clf_acc = rf_acc
        clf_f1 = rf_f1
        clf_pred = rf_pred

    clf_cm = confusion_matrix(y_tier_test, clf_pred).tolist()
    clf_report = classification_report(
        y_tier_test, clf_pred, target_names=label_encoder.classes_, output_dict=True
    )

    # 7. Model 3: Unsupervised Talent Archetypes (K-Means & PCA)
    print("\n--- Model 3: Unsupervised Talent Archetypes (K-Means & PCA) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    processed_df["cluster"] = cluster_labels
    processed_df["pca_1"] = pca_coords[:, 0]
    processed_df["pca_2"] = pca_coords[:, 1]
    processed_df["pca_3"] = pca_coords[:, 2]

    cluster_archetypes = {
        0: "Core Systems & Algorithmic Backend Specialists",
        1: "Applied AI/ML & Data Science Specialists",
        2: "Cloud & DevOps Infrastructure Architects",
        3: "Emerging Full-Stack & Entry Developers"
    }

    # 8. Model 4: Outlier Profile Detector
    print("\n--- Model 4: Candidate Anomaly & High-Outlier Detector ---")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    iso_forest.fit(X_scaled)
    anomaly_count = int(np.sum(iso_forest.predict(X_scaled) == -1))
    print(f"Isolation Forest flagged {anomaly_count} non-traditional market profiles (3.0%)")

    # 9. Feature Importance Ranking
    importances = best_reg.feature_importances_
    feature_ranking = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(FEATURE_COLUMNS, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # 10. Serialize Artifacts
    print("\nSerializing trained models and metadata to models/saved_models/...")
    joblib.dump(best_reg, os.path.join(models_dir, "salary_regressor.joblib"))
    joblib.dump(best_clf, os.path.join(models_dir, "readiness_classifier.joblib"))
    joblib.dump(kmeans, os.path.join(models_dir, "talent_clusterer.joblib"))
    joblib.dump(pca, os.path.join(models_dir, "pca_transformer.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "feature_scaler.joblib"))
    joblib.dump(iso_forest, os.path.join(models_dir, "anomaly_detector.joblib"))
    joblib.dump(cleaner, os.path.join(models_dir, "data_cleaner.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))

    processed_df["cluster_archetype"] = processed_df["cluster"].map(cluster_archetypes)
    processed_df.to_csv(processed_path, index=False)

    metrics_summary = {
        "regressor": {
            "model_type": "GradientBoostingRegressor",
            "r2_score": round(gbr_r2, 4),
            "rmse_lpa": round(gbr_rmse, 3),
            "mae_lpa": round(gbr_mae, 3)
        },
        "classifier": {
            "model_type": clf_name,
            "accuracy": round(clf_acc, 4),
            "weighted_f1": round(clf_f1, 4),
            "confusion_matrix": clf_cm,
            "classes": list(label_encoder.classes_),
            "classification_report": clf_report
        },
        "clustering": {
            "algorithm": "K-Means",
            "k_clusters": 4,
            "explained_variance_ratio_pca": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "archetypes": cluster_archetypes
        },
        "top_features": feature_ranking[:8],
        "feature_columns": FEATURE_COLUMNS
    }

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Metrics and evaluation reports saved to {metrics_path}")
    print("=" * 65)
    print("[SUCCESS] Training Pipeline Completed Successfully!")
    print("=" * 65)
    return metrics_summary


if __name__ == "__main__":
    run_pipeline()
