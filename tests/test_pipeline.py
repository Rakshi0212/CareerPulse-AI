"""
CareerPulse Automated Unit Test Suite
====================================
Standard unittest test suite validating:
1. Candidate dataset generation integrity
2. Data cleaning and missing value imputation
3. Feature engineering and coding intensity index calculations
4. Skill gap analyzer and salary ROI estimation
5. End-to-end model inference for salary regression and readiness classification
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_generator import generate_candidate_data
from src.preprocessing.data_cleaner import CandidateDataCleaner
from src.features.engineering import CandidateFeatureEngineer
from src.skill_analyzer.analyzer import CareerSkillAnalyzer, ROLE_TARGETS
from src.models.train import FEATURE_COLUMNS


class TestCareerPulsePipeline(unittest.TestCase):

    def setUp(self):
        self.raw_data = generate_candidate_data(n_samples=60, random_state=42)
        self.cleaner = CandidateDataCleaner()
        self.engineer = CandidateFeatureEngineer()
        self.models_dir = os.path.join(BASE_DIR, "models", "saved_models")

    def test_01_data_generation_schema(self):
        """Verify candidate dataframe shape and required fields."""
        self.assertEqual(len(self.raw_data), 60)
        expected_cols = [
            "candidate_id", "experience_years", "education_level", "college_tier",
            "role_track", "location_type", "internships_count", "github_projects",
            "dsa_problems_solved", "certifications_count", "has_python", "has_sql",
            "has_react_node", "has_cloud_aws", "has_docker_k8s", "has_ml_pytorch",
            "has_system_design", "total_skills", "salary_lpa", "readiness_tier"
        ]
        for col in expected_cols:
            self.assertIn(col, self.raw_data.columns)

    def test_02_cleaner_imputation(self):
        """Verify missing value imputation for DSA problem counts."""
        clean_df = self.cleaner.fit_transform(self.raw_data)
        self.assertEqual(clean_df["dsa_problems_solved"].isna().sum(), 0)
        self.assertGreaterEqual(clean_df["dsa_problems_solved"].min(), 0)

    def test_03_feature_calculations(self):
        """Verify coding intensity index and competitiveness score bounds."""
        clean_df = self.cleaner.fit_transform(self.raw_data)
        feats = self.engineer.transform(clean_df)

        self.assertIn("coding_intensity_index", feats.columns)
        self.assertIn("competitiveness_score", feats.columns)
        self.assertGreaterEqual(feats["competitiveness_score"].min(), 0.0)
        self.assertLessEqual(feats["competitiveness_score"].max(), 100.0)

    def test_04_skill_analyzer_roi(self):
        """Verify role gap analysis and estimated salary ROI computation."""
        student_profile = {
            "has_python": 1,
            "has_sql": 1,
            "has_react_node": 0,
            "has_cloud_aws": 0,
            "has_docker_k8s": 0,
            "has_ml_pytorch": 0,
            "has_system_design": 0,
            "dsa_problems_solved": 80,
            "github_projects": 2
        }
        res = CareerSkillAnalyzer.analyze_gap(student_profile, "AI / Machine Learning Engineer")
        self.assertIn("match_score_pct", res)
        self.assertIn("missing_skills", res)
        self.assertIn("potential_salary_boost_lpa", res)
        self.assertGreater(res["potential_salary_boost_lpa"], 0.0)
        self.assertEqual(res["match_score_pct"], 40.0) # 2 out of 5 required

    def test_05_saved_model_inference(self):
        """Verify serialized models load and generate valid salary predictions."""
        reg_path = os.path.join(self.models_dir, "salary_regressor.joblib")
        clf_path = os.path.join(self.models_dir, "readiness_classifier.joblib")
        label_enc_path = os.path.join(self.models_dir, "label_encoder.joblib")

        self.assertTrue(os.path.exists(reg_path), "Salary regressor not found")
        self.assertTrue(os.path.exists(clf_path), "Readiness classifier not found")

        reg = joblib.load(reg_path)
        clf = joblib.load(clf_path)
        label_enc = joblib.load(label_enc_path)

        clean_df = self.cleaner.fit_transform(self.raw_data)
        feats = self.engineer.transform(clean_df)
        X_sample = feats[FEATURE_COLUMNS].iloc[:5]

        # Salary predictions
        salaries = reg.predict(X_sample)
        self.assertEqual(len(salaries), 5)
        for s in salaries:
            self.assertGreater(s, 2.5) # Above fresher floor
            self.assertLess(s, 100.0)

        # Classification predictions
        tiers = label_enc.inverse_transform(clf.predict(X_sample))
        self.assertEqual(len(tiers), 5)
        for t in tiers:
            self.assertIn(t, [
                "Entry-Level Learner",
                "Junior Associate",
                "Job-Ready Mid-Level",
                "High-Demand Specialist"
            ])


if __name__ == "__main__":
    unittest.main()
