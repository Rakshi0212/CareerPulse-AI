"""
CareerPulse Synthetic Tech Hiring Dataset Generator
===================================================
Generates realistic software, data science, and cloud candidate profiles
calibrated against tech hiring market standards (Levels.fyi, Stack Overflow Developer Survey).
"""

import os
import numpy as np
import pandas as pd


def generate_candidate_data(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate tech candidate profiles with realistic skill combinations,
    education tiers, coding metrics, salary packages (LPA), and job readiness tiers.
    """
    np.random.seed(random_state)

    candidate_ids = [f"CP-{1000 + i}" for i in range(n_samples)]
    
    # Experience years (skewed towards 0 - 5 years for students/junior professionals)
    exp = np.clip(np.random.exponential(scale=2.2, size=n_samples), 0.0, 9.5).round(1)

    education_levels = np.random.choice(
        ["B.Tech/B.E", "BCA/B.Sc CS", "M.Tech/M.S", "MCA", "Non-CS Degree"],
        size=n_samples,
        p=[0.55, 0.15, 0.12, 0.10, 0.08]
    )

    college_tiers = np.random.choice(
        ["Tier 1 (IIT/NIT/BITS)", "Tier 2 (Top State/Private)", "Tier 3 (Affiliated Colleges)"],
        size=n_samples,
        p=[0.16, 0.44, 0.40]
    )

    role_tracks = np.random.choice(
        ["Data Science & AI/ML", "Full Stack Web", "Cloud & DevOps", "Data Analytics & BI", "Backend Systems"],
        size=n_samples,
        p=[0.28, 0.27, 0.16, 0.15, 0.14]
    )

    locations = np.random.choice(
        ["Tier 1 Tech Hub (Bengaluru/NCR/Hyd)", "Tier 2 City", "Remote Global"],
        size=n_samples,
        p=[0.60, 0.25, 0.15]
    )

    # Coding & Projects metrics
    internships = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.40, 0.18, 0.07])
    github_projects = np.clip(np.random.poisson(lam=3.2, size=n_samples) + internships, 1, 12)
    
    # DSA LeetCode problems (0 to 450)
    dsa_base = np.random.choice([0, 40, 120, 220, 350], size=n_samples, p=[0.25, 0.35, 0.22, 0.12, 0.06])
    dsa_problems = np.clip(dsa_base + np.random.randint(-15, 35, size=n_samples), 0, 500)
    
    certifications = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.35, 0.15, 0.05])

    # Core Technical Skills Flags
    has_python = np.random.choice([1, 0], size=n_samples, p=[0.72, 0.28])
    has_sql = np.random.choice([1, 0], size=n_samples, p=[0.68, 0.32])
    has_react_node = np.random.choice([1, 0], size=n_samples, p=[0.48, 0.52])
    has_cloud_aws = np.random.choice([1, 0], size=n_samples, p=[0.38, 0.62])
    has_docker_k8s = np.random.choice([1, 0], size=n_samples, p=[0.32, 0.68])
    has_ml_pytorch = np.random.choice([1, 0], size=n_samples, p=[0.30, 0.70])
    has_system_design = np.random.choice([1, 0], size=n_samples, p=[0.24, 0.76])

    # Composite Skill Breadth & Depth
    total_skills = (
        has_python + has_sql + has_react_node + has_cloud_aws + 
        has_docker_k8s + has_ml_pytorch + has_system_design
    )

    # Market Salary Valuation Engine (LPA in Lakhs per Annum, ₹ INR)
    # Calibrated to Indian Tech Market + Global Remote Standards
    tier_salary_bonus = {
        "Tier 1 (IIT/NIT/BITS)": 3.8,
        "Tier 2 (Top State/Private)": 1.2,
        "Tier 3 (Affiliated Colleges)": 0.0
    }
    degree_salary_bonus = {
        "M.Tech/M.S": 1.8,
        "B.Tech/B.E": 0.8,
        "MCA": 0.5,
        "BCA/B.Sc CS": 0.0,
        "Non-CS Degree": -0.6
    }
    location_salary_mult = {
        "Tier 1 Tech Hub (Bengaluru/NCR/Hyd)": 1.10,
        "Tier 2 City": 0.85,
        "Remote Global": 1.25
    }

    base_salary = 2.8 # Baseline fresher floor without skills

    salary_list = []
    readiness_list = []

    for i in range(n_samples):
        e = exp[i]
        c_tier = college_tiers[i]
        deg = education_levels[i]
        loc = locations[i]

        val = (
            base_salary
            + (e * 2.2)                           # Experience progression
            + tier_salary_bonus[c_tier]           # College prestige premium
            + degree_salary_bonus[deg]            # Degree level
            + (has_cloud_aws[i] * 2.2)            # Cloud premium
            + (has_docker_k8s[i] * 1.8)           # Containerization premium
            + (has_ml_pytorch[i] * 2.5)           # AI/ML specialization premium
            + (has_system_design[i] * 2.8)        # System design
            + (has_react_node[i] * 1.4)           # Full stack utility
            + (has_sql[i] * 0.9)                  # Database proficiency
            + (min(dsa_problems[i], 300) / 100.0) * 1.4 # LeetCode problem solving
            + (internships[i] * 1.2)              # Practical internships
            + (min(github_projects[i], 5) * 0.35) # GitHub portfolio
        )
        
        # Apply location multiplier
        val = val * location_salary_mult[loc]

        # Add realistic market negotiation variance
        noise = np.random.normal(0, 0.9)
        final_salary = round(max(3.0, val + noise), 2)
        salary_list.append(final_salary)

        # Readiness Tier Classification (Industry Recruiting Ladders):
        if final_salary >= 24.0:
            readiness_list.append("High-Demand Specialist")
        elif final_salary >= 15.0:
            readiness_list.append("Job-Ready Mid-Level")
        elif final_salary >= 9.0:
            readiness_list.append("Junior Associate")
        else:
            readiness_list.append("Entry-Level Learner")

    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "experience_years": exp,
        "education_level": education_levels,
        "college_tier": college_tiers,
        "role_track": role_tracks,
        "location_type": locations,
        "internships_count": internships,
        "github_projects": github_projects,
        "dsa_problems_solved": dsa_problems,
        "certifications_count": certifications,
        "has_python": has_python,
        "has_sql": has_sql,
        "has_react_node": has_react_node,
        "has_cloud_aws": has_cloud_aws,
        "has_docker_k8s": has_docker_k8s,
        "has_ml_pytorch": has_ml_pytorch,
        "has_system_design": has_system_design,
        "total_skills": total_skills,
        "salary_lpa": salary_list,
        "readiness_tier": readiness_list
    })

    # Realistic dirty EHR/Resume artifacts:
    # Inject ~2% missing in dsa_problems_solved or certifications to test cleaning pipeline
    nan_mask = np.random.choice([True, False], size=n_samples, p=[0.02, 0.98])
    df.loc[nan_mask, "dsa_problems_solved"] = np.nan

    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "raw_candidates_market.csv")
    
    df = generate_candidate_data(n_samples=3500, random_state=42)
    df.to_csv(out_file, index=False)
    print(f"[CareerPulse] Generated {len(df)} candidate records -> {out_file}")
    print("Readiness Tier Distribution:\n", df["readiness_tier"].value_counts())
    print("\nSalary LPA Summary:\n", df["salary_lpa"].describe().round(2))
