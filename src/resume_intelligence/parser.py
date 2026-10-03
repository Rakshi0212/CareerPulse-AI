"""
CareerPulse Resume Parser
=========================
Extracts text from PDF/TXT resumes, identifies candidate contact details,
education, verified technical skills, projects, and LeetCode problem solving.
"""

import re
import io
from typing import Dict, List, Any

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class ResumeParser:
    """
    Parses resume documents into structured candidate profiles for ML valuation.
    """

    SKILL_PATTERNS = {
        "has_python": [r"\bpython\b", r"\bpy\b", r"\bpandas\b", r"\bnumpy\b"],
        "has_sql": [r"\bsql\b", r"\bmysql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bsqlite\b"],
        "has_react_node": [r"\breact\b", r"\bnode(?:\.js)?\b", r"\bexpress(?:\.js)?\b", r"\bjavascript\b", r"\btypescript\b", r"\bhtml\b", r"\bcss\b"],
        "has_cloud_aws": [r"\baws\b", r"\bgcp\b", r"\bgoogle cloud\b", r"\bazure\b", r"\bcloud\b", r"\bs3\b", r"\bec2\b"],
        "has_docker_k8s": [r"\bdocker\b", r"\bkubernetes\b", r"\bk8s\b", r"\bcontainer(?:s)?\b", r"\bci/cd\b"],
        "has_ml_pytorch": [r"\bmachine learning\b", r"\bpytorch\b", r"\btensorflow\b", r"\bscikit-learn\b", r"\bsklearn\b", r"\bdeep learning\b", r"\bkeras\b"],
        "has_system_design": [r"\bsystem design\b", r"\bmicroservices\b", r"\bdistributed systems\b", r"\bscalab(?:le|ility)\b", r"\bkfaka\b", r"\bredis\b"]
    }

    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
        """Extract plain text from uploaded PDF or TXT bytes."""
        text = ""
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            if PYPDF_AVAILABLE:
                try:
                    pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                except Exception as e:
                    text = f"[PDF Extraction Error: {str(e)}]"
            else:
                text = file_bytes.decode("utf-8", errors="ignore")
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

        return text.strip()

    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume text into structured features for CareerPulse ML models."""
        lower_text = text.lower()

        # 1. Contact / Name / GitHub / LinkedIn
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        email = email_match.group(0) if email_match else "Not detected"

        github_match = re.search(r"github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
        has_github_link = bool(github_match)

        linkedin_match = re.search(r"linkedin\.com/in/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
        has_linkedin_link = bool(linkedin_match)

        # 2. Education Level & College Tier Detection
        education_level = "B.Tech/B.E"
        if re.search(r"\bm\.?tech\b|\bm\.?s\b|\bmaster(?:'s)?\b", lower_text):
            education_level = "M.Tech/M.S"
        elif re.search(r"\bmca\b", lower_text):
            education_level = "MCA"
        elif re.search(r"\bbca\b|\bb\.?sc\b", lower_text):
            education_level = "BCA/B.Sc CS"
        elif not re.search(r"\bcomputer science\b|\bengineering\b|\bb\.?tech\b|\bit\b", lower_text):
            education_level = "Non-CS Degree"

        college_tier = "Tier 2 (Top State/Private)"
        if re.search(r"\biit\b|\bnit\b|\bbits\b|\biiit\b", lower_text):
            college_tier = "Tier 1 (IIT/NIT/BITS)"
        elif re.search(r"\baffiliated\b|\buniversity college\b|\binstitute of technology\b", lower_text):
            college_tier = "Tier 3 (Affiliated Colleges)"

        # 3. Experience Estimation
        exp_years = 0.5 # Default fresher / early career baseline
        exp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)", lower_text)
        if exp_matches:
            try:
                exp_years = float(exp_matches[0])
            except ValueError:
                exp_years = 1.0

        # Internships detection
        internships = len(re.findall(r"\bintern(?:ship)?\b|\btrainee\b", lower_text))
        internships_count = min(3, max(0, internships))

        # 4. Coding Intensity & LeetCode detection
        dsa_solved = 0
        dsa_matches = re.findall(r"(\d+)\s*(?:\+)?\s*(?:problems?|questions?|dsa|leetcode|codeforces|gfg)", lower_text)
        if dsa_matches:
            try:
                dsa_solved = min(500, int(dsa_matches[0]))
            except ValueError:
                dsa_solved = 80
        elif re.search(r"\bleetcode\b|\bgeeksforgeeks\b|\bhackerrank\b", lower_text):
            dsa_solved = 90

        # Projects Count
        project_mentions = len(re.findall(r"\bproject(?:\s*#?\d+|\s*title|\s*name)?\b", lower_text))
        github_projects = min(8, max(1, project_mentions))

        # 5. Technical Skill Matching
        detected_skills = {}
        for skill_key, patterns in self.SKILL_PATTERNS.items():
            matched = False
            for pat in patterns:
                if re.search(pat, lower_text):
                    matched = True
                    break
            detected_skills[skill_key] = 1 if matched else 0

        # 6. Location / Work preference
        location_type = "Tier 1 Tech Hub (Bengaluru/NCR/Hyd)"
        if "remote" in lower_text:
            location_type = "Remote Global"

        return {
            "email": email,
            "has_github_link": has_github_link,
            "has_linkedin_link": has_linkedin_link,
            "education_level": education_level,
            "college_tier": college_tier,
            "experience_years": exp_years,
            "internships_count": internships_count,
            "dsa_problems_solved": dsa_solved,
            "github_projects": github_projects,
            "certifications_count": 1 if "certified" in lower_text or "certification" in lower_text else 0,
            "location_type": location_type,
            **detected_skills,
            "total_skills": sum(detected_skills.values()),
            "raw_text": text
        }
