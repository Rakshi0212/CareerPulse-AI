"""
CareerPulse Skill Gap & High-ROI Upskilling Engine
==================================================
Analyzes candidate skillsets against industry target roles, computes role match %,
identifies high-leverage missing skills, and quantifies estimated salary ROI.
"""

from typing import Dict, List, Any


ROLE_TARGETS = {
    "AI / Machine Learning Engineer": {
        "required_skills": ["has_python", "has_sql", "has_ml_pytorch", "has_cloud_aws", "has_docker_k8s"],
        "recommended_dsa": 160,
        "recommended_projects": 4,
        "role_baseline_lpa": 16.5,
        "skill_roi_lpa": {
            "has_ml_pytorch": 3.2,
            "has_cloud_aws": 2.6,
            "has_docker_k8s": 2.1,
            "has_system_design": 2.8,
            "has_sql": 1.2
        }
    },
    "Full Stack Software Engineer": {
        "required_skills": ["has_react_node", "has_sql", "has_system_design", "has_cloud_aws", "has_python"],
        "recommended_dsa": 200,
        "recommended_projects": 5,
        "role_baseline_lpa": 14.5,
        "skill_roi_lpa": {
            "has_system_design": 3.4,
            "has_cloud_aws": 2.5,
            "has_react_node": 2.2,
            "has_docker_k8s": 1.8,
            "has_sql": 1.1
        }
    },
    "Cloud & DevOps Platform Engineer": {
        "required_skills": ["has_cloud_aws", "has_docker_k8s", "has_python", "has_system_design"],
        "recommended_dsa": 120,
        "recommended_projects": 3,
        "role_baseline_lpa": 15.0,
        "skill_roi_lpa": {
            "has_cloud_aws": 3.0,
            "has_docker_k8s": 2.6,
            "has_system_design": 2.8,
            "has_python": 1.5
        }
    },
    "Data Analyst & Analytics Engineer": {
        "required_skills": ["has_sql", "has_python"],
        "recommended_dsa": 60,
        "recommended_projects": 3,
        "role_baseline_lpa": 10.5,
        "skill_roi_lpa": {
            "has_sql": 2.0,
            "has_python": 1.8,
            "has_ml_pytorch": 2.2,
            "has_cloud_aws": 1.9
        }
    }
}

SKILL_DISPLAY_NAMES = {
    "has_python": "Python & Data Structures",
    "has_sql": "SQL & Relational Databases",
    "has_react_node": "React, Node.js & Full-Stack",
    "has_cloud_aws": "Cloud Computing (AWS / GCP)",
    "has_docker_k8s": "Docker & Kubernetes Containers",
    "has_ml_pytorch": "Machine Learning & PyTorch",
    "has_system_design": "Scalable System Design & Architecture"
}


class CareerSkillAnalyzer:
    """
    Evaluates candidate readiness for target roles and computes high-ROI skill roadmap.
    """

    @staticmethod
    def analyze_gap(candidate_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        if target_role not in ROLE_TARGETS:
            target_role = "AI / Machine Learning Engineer"

        role_info = ROLE_TARGETS[target_role]
        required = role_info["required_skills"]
        skill_roi = role_info["skill_roi_lpa"]

        acquired_skills = []
        missing_skills = []
        potential_salary_boost = 0.0

        for s in required:
            if candidate_profile.get(s, 0) == 1:
                acquired_skills.append({
                    "skill_key": s,
                    "name": SKILL_DISPLAY_NAMES.get(s, s),
                    "status": "Acquired"
                })
            else:
                boost = skill_roi.get(s, 1.5)
                missing_skills.append({
                    "skill_key": s,
                    "name": SKILL_DISPLAY_NAMES.get(s, s),
                    "estimated_lpa_boost": boost
                })
                potential_salary_boost += boost

        # Match score %
        match_score = round((len(acquired_skills) / max(len(required), 1)) * 100, 1)

        # DSA & Projects readiness
        curr_dsa = candidate_profile.get("dsa_problems_solved", 0)
        rec_dsa = role_info["recommended_dsa"]
        dsa_gap = max(0, rec_dsa - curr_dsa)

        curr_proj = candidate_profile.get("github_projects", 0)
        rec_proj = role_info["recommended_projects"]
        proj_gap = max(0, rec_proj - curr_proj)

        # Learning roadmap ordered by highest salary ROI
        missing_skills.sort(key=lambda x: x["estimated_lpa_boost"], reverse=True)

        recommendations = []
        if missing_skills:
            top_skill = missing_skills[0]["name"]
            recommendations.append(
                f"🚀 **Highest ROI Skill**: Learn **{top_skill}** first. It commands an estimated **+₹{missing_skills[0]['estimated_lpa_boost']:.1f} LPA** compensation premium."
            )
        if dsa_gap > 0:
            recommendations.append(
                f"💻 **Algorithmic Preparation**: Solve **{dsa_gap} more LeetCode/DSA problems** to meet the recommended standard ({rec_dsa} problems) for technical screening rounds."
            )
        if proj_gap > 0:
            recommendations.append(
                f"📂 **Portfolio Expansion**: Build **{proj_gap} more end-to-end production projects** on GitHub with Dockerization and live demos."
            )
        if not missing_skills and dsa_gap == 0:
            recommendations.append("🌟 **Fully Market-Ready**: Your profile matches 100% of the baseline requirements for this role! Focus on system design interviews and mock rounds.")

        return {
            "target_role": target_role,
            "match_score_pct": match_score,
            "acquired_count": len(acquired_skills),
            "missing_count": len(missing_skills),
            "acquired_skills": acquired_skills,
            "missing_skills": missing_skills,
            "potential_salary_boost_lpa": round(potential_salary_boost, 2),
            "current_dsa": curr_dsa,
            "recommended_dsa": rec_dsa,
            "dsa_gap": dsa_gap,
            "current_projects": curr_proj,
            "recommended_projects": rec_proj,
            "actionable_recommendations": recommendations
        }
