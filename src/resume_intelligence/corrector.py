"""
CareerPulse Automated Resume Polisher & Bullet Corrector
========================================================
Transforms weak, passive resume bullet points into high-impact Google XYZ-formula achievements,
restructures messy sections, and outputs an ATS-optimized polished resume.
"""

import re
from typing import Dict, List, Any


ACTION_REPLACEMENTS = {
    "worked on": "Engineered and delivered",
    "helped with": "Collaborated across agile sprints to implement",
    "assisted in": "Co-developed and benchmarked",
    "responsible for": "Spearheaded the design and deployment of",
    "handled": "Administered and scaled",
    "participated in": "Contributed production features to",
    "did": "Architected and executed",
    "made": "Designed, developed, and deployed",
    "involved in": "Spearheaded core development for"
}


class ResumeCorrector:
    """
    Automated resume polisher that transforms weak resumes into high-impact documents.
    """

    @staticmethod
    def rewrite_weak_bullet(bullet: str) -> str:
        """Rewrite a single weak bullet point using Google's XYZ formula."""
        lower = bullet.lower().strip().lstrip("-*• ")
        
        # Replace leading weak verb
        for weak, strong in ACTION_REPLACEMENTS.items():
            if lower.startswith(weak):
                rest = bullet.strip().lstrip("-*• ")[len(weak):].strip()
                # Inject impact metric if missing
                if not any(char.isdigit() or "%" in rest for char in rest):
                    rest = f"{rest}, enhancing operational throughput by 32% and reducing latency across 5,000+ simulated requests"
                return f"• **{strong}** {rest}"

        # If weak verb is in the middle
        for weak, strong in ACTION_REPLACEMENTS.items():
            if weak in lower:
                fixed = re.sub(re.escape(weak), strong, bullet, flags=re.IGNORECASE).strip().lstrip("-*• ")
                if not any(char.isdigit() or "%" in fixed for char in fixed):
                    fixed = f"{fixed}, accelerating query performance by 28%"
                return f"• {fixed}"

        return f"• **Engineered** {bullet.strip().lstrip('-*• ')}"

    @classmethod
    def generate_polished_resume(
        cls,
        raw_text: str,
        parsed_profile: Dict[str, Any],
        weak_bullets: List[Dict[str, str]],
        target_role: str = "AI / Machine Learning Engineer"
    ) -> str:
        """
        Generate a fully polished, ATS-optimized markdown resume ready to export.
        """
        skills = []
        if parsed_profile.get("has_python"): skills.append("Python (OOP, Data Structures, Multithreading)")
        if parsed_profile.get("has_sql"): skills.append("SQL (PostgreSQL, MySQL, Query Optimization)")
        if parsed_profile.get("has_ml_pytorch"): skills.append("Machine Learning & PyTorch (Scikit-Learn, Deep Learning, Feature Engineering)")
        if parsed_profile.get("has_cloud_aws"): skills.append("Cloud Computing (AWS S3, EC2, Lambda, GCP)")
        if parsed_profile.get("has_docker_k8s"): skills.append("DevOps & Containers (Docker, Kubernetes, CI/CD Pipelines)")
        if parsed_profile.get("has_react_node"): skills.append("Full-Stack (React.js, Node.js, Express, REST APIs)")
        if parsed_profile.get("has_system_design"): skills.append("System Design (Microservices, Caching, Scalability)")

        if not skills:
            skills = ["Python", "SQL", "Git", "REST APIs", "Data Structures & Algorithms"]

        email = parsed_profile.get("email", "candidate@example.com")
        edu = parsed_profile.get("education_level", "B.Tech/B.E Computer Science")
        tier = parsed_profile.get("college_tier", "Tier 2 University")
        dsa = parsed_profile.get("dsa_problems_solved", 120)

        # Polished corrected bullets
        corrected_bullets = []
        if weak_bullets:
            for item in weak_bullets:
                corrected_bullets.append(cls.rewrite_weak_bullet(item["original"]))
        else:
            corrected_bullets = [
                f"• **Architected and trained** production ML models delivering 91.8% benchmark accuracy and reducing inference latency by 35%.",
                f"• **Engineered** end-to-end data processing pipelines handling 100,000+ records with automated cleaning and feature extraction.",
                f"• **Containerized** microservices with Docker and deployed scalable API endpoints with 99.9% uptime."
            ]

        polished_md = f"""# [YOUR NAME]
**{target_role}** | {email} | [LinkedIn Profile](https://linkedin.com) | [GitHub Portfolio](https://github.com)

---

## 🎯 Professional Summary
Driven **{target_role}** with a strong foundation in {', '.join([s.split(' ')[0] for s in skills[:3]])}. Proven track record of developing end-to-end data systems and high-throughput software architectures. Solved **{dsa}+ algorithmic problems** with strong CS fundamentals in Data Structures, System Design, and Cloud Deployment.

---

## 🛠️ Technical Skills
- **Languages & Core:** {', '.join([s for s in skills if 'Python' in s or 'SQL' in s or 'Full' in s]) or 'Python, SQL, C++, Java'}
- **Frameworks & AI/ML:** {', '.join([s for s in skills if 'Machine' in s or 'React' in s]) or 'Scikit-Learn, PyTorch, Pandas, NumPy, FastAPI'}
- **Cloud & DevOps:** {', '.join([s for s in skills if 'Cloud' in s or 'DevOps' in s]) or 'Docker, AWS (S3, EC2), Git, Linux, CI/CD'}
- **Problem Solving:** LeetCode & DSA ({dsa}+ Problems Solved, Arrays, Trees, Dynamic Programming, Graphs)

---

## 💼 Experience & Key Impact
**Software Engineering / ML Intern** | *Tech Organization*
{chr(10).join(corrected_bullets[:3])}

---

## 🚀 Featured Engineering Projects
**Enterprise Machine Learning & Valuation Intelligence Platform** | *Python, Scikit-Learn, Streamlit, Docker*
• **Architected and trained** multi-tier ML models achieving an **R² score of 0.9712** and **87.6% classification accuracy** across 3,500 candidate profiles.
• **Engineered** automated feature extraction pipelines deriving coding intensity indices and market competitiveness metrics.
• **Containerized** the full-stack web application using Docker and deployed interactive analytical dashboards with sub-second response times.

**High-Throughput Distributed Cloud Service** | *Python, SQL, Docker, AWS*
• **Developed** scalable RESTful backend services processing 10,000+ daily requests with automated schema validation.
• **Optimized** database queries reducing complex aggregation latency by **42%**.

---

## 🎓 Education & Certifications
- **{edu}** | *{tier}* (CGPA: 8.4/10.0)
- **Certifications:** Certified Cloud Practitioner / Applied Data Science Specialist
"""
        return polished_md.strip()
