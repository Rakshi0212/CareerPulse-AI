"""
CareerPulse Candidate Data Cleaner & Preprocessor
=================================================
Handles missing value imputation, text normalization, and bounds enforcement
for tech applicant profiles and resume metadata.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class CandidateDataCleaner:
    """
    Cleans raw tech applicant profiles and imputes missing fields
    using median statistics learned on training cohorts.
    """

    def __init__(self):
        self.medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CandidateDataCleaner":
        numerical_cols = [
            "experience_years", "internships_count", "github_projects",
            "dsa_problems_solved", "certifications_count",
            "has_python", "has_sql", "has_react_node", "has_cloud_aws",
            "has_docker_k8s", "has_ml_pytorch", "has_system_design",
            "total_skills"
        ]
        for col in numerical_cols:
            if col in df.columns:
                self.medians[col] = float(df[col].median(skipna=True))
        self.is_fitted = True
        return self

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()

        # 1. Fill Missing Numerical Values
        for col, median_val in self.medians.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(median_val)
            elif col in ["dsa_problems_solved", "github_projects", "certifications_count"]:
                cleaned[col] = cleaned.get(col, pd.Series([0])).fillna(0)

        # 2. Enforce Logical Bounds
        if "experience_years" in cleaned.columns:
            cleaned["experience_years"] = np.clip(cleaned["experience_years"], 0.0, 30.0)
        if "dsa_problems_solved" in cleaned.columns:
            cleaned["dsa_problems_solved"] = np.clip(cleaned["dsa_problems_solved"], 0, 1500)
        if "github_projects" in cleaned.columns:
            cleaned["github_projects"] = np.clip(cleaned["github_projects"], 0, 40)
        if "internships_count" in cleaned.columns:
            cleaned["internships_count"] = np.clip(cleaned["internships_count"], 0, 8)

        # 3. Canonicalize Categoricals
        if "education_level" in cleaned.columns:
            cleaned["education_level"] = cleaned["education_level"].fillna("B.Tech/B.E")
        if "college_tier" in cleaned.columns:
            cleaned["college_tier"] = cleaned["college_tier"].fillna("Tier 2 (Top State/Private)")
        if "location_type" in cleaned.columns:
            cleaned["location_type"] = cleaned["location_type"].fillna("Tier 1 Tech Hub (Bengaluru/NCR/Hyd)")

        return cleaned

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).clean(df)
