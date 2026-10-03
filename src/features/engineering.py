"""
CareerPulse Feature Engineering & Market Scoring
================================================
Extracts market-demanded domain features, coding intensity indices,
and candidate competitiveness scores.
"""

import numpy as np
import pandas as pd


class CandidateFeatureEngineer:
    """
    Transforms raw candidate profile data into predictive tech valuation features.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = df.copy()

        # 1. Coding & Practical Intensity Index
        dsa_comp = np.clip(feats["dsa_problems_solved"] / 40.0, 0, 10.0)
        proj_comp = np.clip(feats["github_projects"] * 0.7, 0, 8.0)
        intern_comp = feats["internships_count"] * 1.5
        feats["coding_intensity_index"] = (dsa_comp + proj_comp + intern_comp).round(2)

        # 2. Skill Domain Specializations
        feats["cloud_devops_depth"] = feats["has_cloud_aws"] + feats["has_docker_k8s"]
        feats["ai_data_depth"] = feats["has_python"] + feats["has_sql"] + feats["has_ml_pytorch"]
        feats["fullstack_systems_depth"] = feats["has_react_node"] + feats["has_system_design"]

        # 3. Categorical Encodings & Scores
        college_map = {
            "Tier 1 (IIT/NIT/BITS)": 3,
            "Tier 2 (Top State/Private)": 2,
            "Tier 3 (Affiliated Colleges)": 1
        }
        feats["college_tier_score"] = feats["college_tier"].map(college_map).fillna(1)

        edu_map = {
            "M.Tech/M.S": 4,
            "B.Tech/B.E": 3,
            "MCA": 2.5,
            "BCA/B.Sc CS": 2,
            "Non-CS Degree": 1
        }
        feats["education_score"] = feats["education_level"].map(edu_map).fillna(2)

        feats["loc_tier1_hub"] = (
            feats["location_type"] == "Tier 1 Tech Hub (Bengaluru/NCR/Hyd)"
        ).astype(int)
        feats["loc_remote"] = (feats["location_type"] == "Remote Global").astype(int)

        # 4. Total Verified Skill Breadth
        feats["skill_breadth"] = (
            feats["has_python"] + feats["has_sql"] + feats["has_react_node"] +
            feats["has_cloud_aws"] + feats["has_docker_k8s"] + feats["has_ml_pytorch"] +
            feats["has_system_design"]
        )

        # 5. Composite Employability Competitiveness Score (0 - 100)
        # Components:
        # - Experience: 25 pts max
        # - Skills breadth: 25 pts max
        # - Coding intensity: 25 pts max
        # - Education & College tier: 25 pts max
        exp_pts = np.clip(feats["experience_years"] / 5.0, 0, 1.0) * 25.0
        skill_pts = np.clip(feats["skill_breadth"] / 6.0, 0, 1.0) * 25.0
        coding_pts = np.clip(feats["coding_intensity_index"] / 15.0, 0, 1.0) * 25.0
        edu_pts = ((feats["college_tier_score"] / 3.0) * 12.5) + ((feats["education_score"] / 4.0) * 12.5)

        feats["competitiveness_score"] = (exp_pts + skill_pts + coding_pts + edu_pts).round(1)

        return feats
