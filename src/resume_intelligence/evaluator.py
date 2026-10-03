"""
CareerPulse Resume Quality & ATS Evaluator
==========================================
Scans resume content for ATS score, passive language, missing quantifiable metrics,
unstructured sections, and keyword density gaps.
"""

import re
from typing import Dict, List, Any


WEAK_ACTION_VERBS = [
    "worked on", "responsible for", "helped with", "assisted in",
    "handled", "participated in", "did", "made", "involved in", "looked after"
]

STRONG_ACTION_VERBS = [
    "engineered", "architected", "developed", "deployed", "optimized",
    "accelerated", "designed", "implemented", "reduced", "scaled", "automated"
]


class ResumeEvaluator:
    """
    Evaluates resume health, ATS pass probability, and structural gaps.
    """

    @staticmethod
    def evaluate(resume_text: str, parsed_profile: Dict[str, Any]) -> Dict[str, Any]:
        lower = resume_text.lower()
        score = 100
        penalties = []
        strengths = []
        weak_bullet_points = []

        # 1. Section Completeness Check (25 pts max)
        expected_sections = ["education", "skills", "projects", "experience"]
        missing_sections = []
        for sec in expected_sections:
            if not re.search(rf"\b{sec}\b", lower):
                missing_sections.append(sec.title())
        
        if missing_sections:
            pen = len(missing_sections) * 6
            score -= pen
            penalties.append(f"Missing core section headers: {', '.join(missing_sections)} (-{pen} pts)")
        else:
            strengths.append("Contains all standard ATS sections (Education, Skills, Projects, Experience).")

        # 2. Portfolio Links (15 pts max)
        if not parsed_profile.get("has_github_link"):
            score -= 8
            penalties.append("No clickable GitHub portfolio or code repository link detected (-8 pts).")
        else:
            strengths.append("Verified GitHub portfolio link present.")

        if not parsed_profile.get("has_linkedin_link"):
            score -= 5
            penalties.append("No LinkedIn profile link found (-5 pts).")

        # 3. Quantifiable Impact & Metrics (25 pts max)
        # Check for numbers, %, scale, latency, users
        metric_matches = re.findall(r"(\d+(?:\.\d+)?%|\$\d+|\b\d+\s*(?:users?|requests?|ms|seconds?|lpa|k\b|x\b))", lower)
        if len(metric_matches) < 2:
            score -= 15
            penalties.append("Insufficient quantifiable metrics (no numbers, %, scale, or latency improvements) (-15 pts).")
        elif len(metric_matches) >= 4:
            strengths.append(f"Strong quantifiable impact: {len(metric_matches)} metrics detected (percentages, scale, improvements).")

        # 4. Weak / Passive Verb Detection (20 pts max)
        lines = [line.strip() for line in resume_text.split("\n") if len(line.strip()) > 20]
        detected_weak_count = 0
        for line in lines:
            line_lower = line.lower()
            for weak in WEAK_ACTION_VERBS:
                if weak in line_lower:
                    detected_weak_count += 1
                    weak_bullet_points.append({
                        "original": line,
                        "weak_phrase": weak
                    })
                    break

        if detected_weak_count > 0:
            pen = min(15, detected_weak_count * 4)
            score -= pen
            penalties.append(f"Found {detected_weak_count} bullet points with passive/weak verbs ('worked on', 'helped with') (-{pen} pts).")
        else:
            strengths.append("Action-driven language: No passive verbs detected.")

        # 5. Technical Skills Density (15 pts max)
        total_skills = parsed_profile.get("total_skills", 0)
        if total_skills < 3:
            score -= 12
            penalties.append(f"Low verified technical skill breadth: only {total_skills} core tech skills found (-12 pts).")
        else:
            strengths.append(f"Solid technical breadth: {total_skills} verified core skills detected.")

        final_score = max(25, min(98, score))

        if final_score >= 80:
            status = "Excellent — Highly ATS-Optimized"
            grade_color = "#10b981"
        elif final_score >= 60:
            status = "Moderate — Needs Metric & Action Verb Tuning"
            grade_color = "#f59e0b"
        else:
            status = "Low — Lacks Structure, Metrics & Action Verbs"
            grade_color = "#ef4444"

        return {
            "ats_score": final_score,
            "status": status,
            "grade_color": grade_color,
            "penalties": penalties,
            "strengths": strengths,
            "weak_bullets": weak_bullet_points[:5],
            "metrics_count": len(metric_matches)
        }
