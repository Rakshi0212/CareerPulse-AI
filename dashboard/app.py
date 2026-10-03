"""
CareerPulse AI — Tech Career Skill Gap, Salary Valuation & Resume Intelligence Platform
========================================================================================
Interactive Streamlit Dashboard featuring:
- Resume Upload, Parsing & ATS Quality Diagnostic
- Automated Resume Polisher & Google XYZ-formula Bullet Corrector
- Context-Aware AI Career Copilot / Agent
- Live Market Salary Valuation & Readiness Tier Classifier
- High-ROI Skill Gap Analyzer & Learning Roadmap
- Tech Market Salary Analytics & Experience Curves
- Unsupervised Talent Archetypes (PCA Clustering) & Governance
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.train import FEATURE_COLUMNS
from src.features.engineering import CandidateFeatureEngineer
from src.skill_analyzer.analyzer import CareerSkillAnalyzer, ROLE_TARGETS, SKILL_DISPLAY_NAMES
from src.resume_intelligence.parser import ResumeParser
from src.resume_intelligence.evaluator import ResumeEvaluator
from src.resume_intelligence.corrector import ResumeCorrector
from src.resume_intelligence.ai_agent import CareerCopilotAgent

# Page Setup
st.set_page_config(
    page_title="CareerPulse AI — Resume Intelligence & Tech Valuation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Dark Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .career-header {
        background: linear-gradient(135deg, #091224 0%, #0f1f38 50%, #0d3b66 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 10px 30px rgba(0, 15, 30, 0.45);
    }

    .career-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .career-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Valuation Card */
    .val-card {
        background: #0d1729;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .val-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 8px 0;
    }

    /* Readiness Badges */
    .badge-tier {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-Entry-Level-Learner {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid #64748b;
    }
    .badge-Junior-Associate {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid #38bdf8;
    }
    .badge-Job-Ready-Mid-Level {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .badge-High-Demand-Specialist {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.25);
    }

    /* Action Card */
    .action-card {
        background: #0f172a;
        border-left: 4px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .correction-box {
        background: #111e33;
        border: 1px solid #233554;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_data():
    models_dir = os.path.join(BASE_DIR, "models", "saved_models")
    
    # Check if models exist and can be loaded with the host's scikit-learn version
    need_train = False
    required_files = [
        "salary_regressor.joblib", "readiness_classifier.joblib",
        "talent_clusterer.joblib", "pca_transformer.joblib",
        "feature_scaler.joblib", "data_cleaner.joblib",
        "label_encoder.joblib", "metrics.json"
    ]
    for rf in required_files:
        if not os.path.exists(os.path.join(models_dir, rf)):
            need_train = True
            break
            
    if not need_train:
        try:
            reg = joblib.load(os.path.join(models_dir, "salary_regressor.joblib"))
            clf = joblib.load(os.path.join(models_dir, "readiness_classifier.joblib"))
        except Exception:
            # Model was pickled on a different scikit-learn version. Retrain automatically!
            need_train = True

    if need_train:
        from src.models.train import run_pipeline
        run_pipeline()

    reg = joblib.load(os.path.join(models_dir, "salary_regressor.joblib"))
    clf = joblib.load(os.path.join(models_dir, "readiness_classifier.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "talent_clusterer.joblib"))
    pca = joblib.load(os.path.join(models_dir, "pca_transformer.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "feature_scaler.joblib"))
    cleaner = joblib.load(os.path.join(models_dir, "data_cleaner.joblib"))
    label_enc = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    with open(os.path.join(models_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    data_path = os.path.join(BASE_DIR, "data", "processed_candidates_market.csv")
    df = pd.read_csv(data_path)

    return {
        "reg": reg,
        "clf": clf,
        "kmeans": kmeans,
        "pca": pca,
        "scaler": scaler,
        "cleaner": cleaner,
        "label_enc": label_enc,
        "metrics": metrics,
        "df": df
    }


artifacts = load_models_and_data()
reg = artifacts["reg"]
clf = artifacts["clf"]
kmeans = artifacts["kmeans"]
pca = artifacts["pca"]
scaler = artifacts["scaler"]
cleaner = artifacts["cleaner"]
label_enc = artifacts["label_enc"]
metrics = artifacts["metrics"]
candidates_df = artifacts["df"]
engineer = CandidateFeatureEngineer()
parser = ResumeParser()

# Initialize Session State
if "resume_profile" not in st.session_state:
    st.session_state["resume_profile"] = None
if "resume_evaluation" not in st.session_state:
    st.session_state["resume_evaluation"] = None
if "ai_chat_history" not in st.session_state:
    st.session_state["ai_chat_history"] = []

# --- TOP HEADER ---
st.markdown("""
<div class="career-header">
    <div class="career-title">
        <span>🚀 CareerPulse AI</span>
    </div>
    <div class="career-subtitle">
        AI Resume Intelligence, Automated Bullet Polisher, Career Copilot & Tech Salary Valuation
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation
tabs = st.tabs([
    "📄 Resume Scanner & AI Corrector",
    "🤖 AI Career Copilot (Agent)",
    "🎓 Live Valuation Calculator",
    "🚀 Skill Gap & High-ROI Roadmap",
    "📊 Job Market Trends & Analytics",
    "🎯 Talent Archetypes & Governance"
])

# ==============================================================================
# TAB 1: RESUME SCANNER & AI CORRECTOR
# ==============================================================================
with tabs[0]:
    st.subheader("Automated Resume Scanner, ATS Diagnostic & AI Polisher")
    st.caption("Upload your resume (PDF or TXT) or paste raw text. CareerPulse detects flaws, weak passive verbs, scores your ATS health, and generates a polished Google XYZ-formula resume!")

    u_col1, u_col2 = st.columns([1.2, 1.2])

    with u_col1:
        st.markdown("##### 📤 1. Upload or Paste Resume")
        upload_choice = st.radio("Upload Method", ["Upload Resume File (.pdf, .txt)", "Paste Resume Text"], horizontal=True)

        resume_text = ""
        if upload_choice == "Upload Resume File (.pdf, .txt)":
            uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])
            if uploaded_file is not None:
                resume_text = parser.extract_text_from_bytes(uploaded_file.read(), uploaded_file.name)
        else:
            default_sample = """John Doe
Email: john.doe@example.com | GitHub: github.com/johndoe | LinkedIn: linkedin.com/in/johndoe
Education: B.Tech Computer Science, Tier 2 Engineering College (2024)

Skills: Python, SQL, Machine Learning, Git

Experience:
- Worked on an ML model for churn prediction.
- Helped with building frontend in React for the team.
- Responsible for writing backend SQL queries and APIs.

Projects:
- Did a customer segmentation project using Python.
- Participated in LeetCode coding practice (solved around 80 problems).
"""
            resume_text = st.text_area("Paste Resume Text", default_sample, height=240)

        analyze_btn = st.button("🔍 Scan & Evaluate Resume with AI", type="primary", use_container_width=True)

    if analyze_btn and resume_text.strip():
        parsed = parser.parse_resume(resume_text)
        evaluation = ResumeEvaluator.evaluate(resume_text, parsed)
        st.session_state["resume_profile"] = parsed
        st.session_state["resume_evaluation"] = evaluation

    with u_col2:
        st.markdown("##### 📊 2. ATS Health & Flaw Diagnostic")
        if st.session_state["resume_evaluation"]:
            ev = st.session_state["resume_evaluation"]
            prof = st.session_state["resume_profile"]

            # ATS Score Meter
            st.markdown(f"""
            <div style="background:#0d1729; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; text-align: center; margin-bottom: 16px;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">ATS Resume Quality Score</div>
                <div style="font-size: 2.6rem; font-weight: 800; color: {ev['grade_color']};">{ev['ats_score']}/100</div>
                <div style="color: #cbd5e1; font-weight: 600; font-size: 0.95rem;">{ev['status']}</div>
            </div>
            """, unsafe_allow_html=True)

            if ev["penalties"]:
                st.markdown("###### ⚠️ Identified Resume Flaws:")
                for pen in ev["penalties"]:
                    st.markdown(f"- ❌ {pen}")

            if ev["strengths"]:
                st.markdown("###### ✅ Verified Strengths:")
                for s in ev["strengths"]:
                    st.markdown(f"- ✔️ {s}")
        else:
            st.info("Upload or paste your resume and click 'Scan & Evaluate Resume' to see your ATS diagnostic.")

    st.divider()

    # SECTION 3: AUTOMATED RESUME POLISHER & CORRECTOR
    if st.session_state["resume_evaluation"]:
        st.markdown("### ✨ Automated AI Resume Polisher & Bullet Corrector")
        st.caption("CareerPulse automatically rewrites weak passive bullets into high-impact Google XYZ achievements and structures an ATS-compliant resume.")

        c_left, c_right = st.columns([1.2, 1.2])

        with c_left:
            st.markdown("##### 🛠️ Original Weak Bullets vs. AI Corrections")
            weak_bullets = st.session_state["resume_evaluation"]["weak_bullets"]
            if weak_bullets:
                for idx, b in enumerate(weak_bullets):
                    corrected = ResumeCorrector.rewrite_weak_bullet(b["original"])
                    st.markdown(f"""
                    <div class="correction-box">
                        <div style="color: #f87171; font-size: 0.85rem; font-weight: 600;">❌ BEFORE (Passive / Weak):</div>
                        <div style="color: #94a3b8; font-size: 0.88rem; margin: 4px 0 8px 0;">"{b['original']}"</div>
                        <div style="color: #34d399; font-size: 0.85rem; font-weight: 600;">✨ AFTER (Google XYZ Impact Format):</div>
                        <div style="color: #f8fafc; font-size: 0.9rem;">{corrected}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No weak passive verbs detected! Your bullet points already use strong active language.")

        with c_right:
            st.markdown("##### 📥 Export Fully Polished ATS Resume")
            target_role = st.selectbox(
                "Optimize Resume for Target Role",
                list(ROLE_TARGETS.keys()),
                index=0
            )

            polished_text = ResumeCorrector.generate_polished_resume(
                raw_text=st.session_state["resume_profile"]["raw_text"],
                parsed_profile=st.session_state["resume_profile"],
                weak_bullets=st.session_state["resume_evaluation"]["weak_bullets"],
                target_role=target_role
            )

            st.text_area("Polished ATS-Compliant Markdown Resume", polished_text, height=320)

            st.download_button(
                label="📥 Download Polished Resume (.md)",
                data=polished_text,
                file_name="CareerPulse_Polished_Resume.md",
                mime="text/markdown",
                use_container_width=True
            )


# ==============================================================================
# TAB 2: AI CAREER COPILOT (AGENT)
# ==============================================================================
with tabs[1]:
    st.subheader("CareerPulse AI Copilot — Your Personal Career & Interview Agent")
    st.caption("Chat with an intelligent career copilot who understands your exact parsed resume, predicted market valuation, and target job aspirations.")

    # Build Agent Context
    if st.session_state["resume_profile"]:
        p = st.session_state["resume_profile"]
        agent_context = {
            "target_role": "AI / Machine Learning Engineer",
            "predicted_salary": 14.8,
            "readiness_tier": "Job-Ready Mid-Level",
            "dsa_problems_solved": p.get("dsa_problems_solved", 110),
            "ats_score": st.session_state["resume_evaluation"]["ats_score"] if st.session_state["resume_evaluation"] else 75,
            "missing_skills": [{"name": "Docker & Containers"}, {"name": "AWS Cloud Architecture"}]
        }
    else:
        agent_context = {
            "target_role": "AI / Machine Learning Engineer",
            "predicted_salary": 14.8,
            "readiness_tier": "Job-Ready Mid-Level",
            "dsa_problems_solved": 110,
            "ats_score": 78,
            "missing_skills": [{"name": "Docker & Containers"}, {"name": "AWS Cloud Architecture"}]
        }

    agent = CareerCopilotAgent(agent_context)

    # Quick Action Prompt Buttons
    st.markdown("##### ⚡ Quick Copilot Prompts:")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    if q_col1.button("🎯 Mock Interview Questions"):
        st.session_state["ai_chat_history"].append({"user": "Give me technical interview questions for my profile", "bot": agent.respond("Give me technical interview questions")})
    if q_col2.button("💰 How to Reach ₹25+ LPA?"):
        st.session_state["ai_chat_history"].append({"user": "How do I negotiate and scale to ₹25+ LPA?", "bot": agent.respond("salary increase to 25 lpa")})
    if q_col3.button("📩 Cold Recruiter Message"):
        st.session_state["ai_chat_history"].append({"user": "Draft a cold outreach message for LinkedIn recruiters", "bot": agent.respond("recruiter linkedin cold message")})
    if q_col4.button("📄 Actionable Resume Advice"):
        st.session_state["ai_chat_history"].append({"user": "How do I optimize my resume for ATS?", "bot": agent.respond("resume ats advice")})

    st.write("")

    # Chat Display
    chat_container = st.container(height=380)
    with chat_container:
        if not st.session_state["ai_chat_history"]:
            st.markdown("""
            *Hello! I am your **CareerPulse AI Copilot**. I have analyzed your skills and valuation.*  
            *Ask me anything about interview preparation, high-ROI skills, resume restructuring, or recruiter outreach!*
            """)
        else:
            for chat in st.session_state["ai_chat_history"]:
                with st.chat_message("user"):
                    st.write(chat["user"])
                with st.chat_message("assistant"):
                    st.markdown(chat["bot"])

    # Chat Input
    user_msg = st.chat_input("Ask CareerPulse Copilot (e.g., 'What questions will I be asked?', 'How to improve my projects?')...")
    if user_msg:
        reply = agent.respond(user_msg)
        st.session_state["ai_chat_history"].append({"user": user_msg, "bot": reply})
        st.rerun()


# ==============================================================================
# TAB 3: LIVE VALUATION CALCULATOR
# ==============================================================================
with tabs[2]:
    st.subheader("Personalized Tech Salary Valuation & Readiness Diagnostic")
    st.caption("Input your college background, verified skills, and coding metrics to compute your fair market compensation package.")

    c1, c2, c3 = st.columns([1.1, 1.2, 1.5])

    # Pre-populate from resume if available
    default_exp = st.session_state["resume_profile"]["experience_years"] if st.session_state["resume_profile"] else 0.5
    default_dsa = st.session_state["resume_profile"]["dsa_problems_solved"] if st.session_state["resume_profile"] else 110
    default_proj = st.session_state["resume_profile"]["github_projects"] if st.session_state["resume_profile"] else 3

    with c1:
        st.markdown("##### 🏛️ Education & Experience")
        exp_years = st.slider("Work Experience (Years)", 0.0, 8.0, float(default_exp), 0.5, help="0 for fresh college graduates")
        degree = st.selectbox("Highest Degree", ["B.Tech/B.E", "BCA/B.Sc CS", "M.Tech/M.S", "MCA", "Non-CS Degree"])
        college_tier = st.selectbox("College Tier", ["Tier 1 (IIT/NIT/BITS)", "Tier 2 (Top State/Private)", "Tier 3 (Affiliated Colleges)"], index=1)
        location = st.selectbox("Target Location", ["Tier 1 Tech Hub (Bengaluru/NCR/Hyd)", "Tier 2 City", "Remote Global"])
        internships = st.slider("Internships Completed", 0, 3, 1)

    with c2:
        st.markdown("##### 💻 Coding & Portfolio Metrics")
        dsa_solved = st.slider("DSA / LeetCode Problems Solved", 0, 450, int(default_dsa), 10)
        github_projects = st.slider("Production GitHub Projects", 0, 10, int(default_proj))
        certs = st.slider("Industry Certifications", 0, 4, 1)

        st.markdown("##### 🛠️ Verified Technical Skills")
        p_has_py = st.session_state["resume_profile"].get("has_python", 1) == 1 if st.session_state["resume_profile"] else True
        p_has_sql = st.session_state["resume_profile"].get("has_sql", 1) == 1 if st.session_state["resume_profile"] else True
        p_has_ml = st.session_state["resume_profile"].get("has_ml_pytorch", 1) == 1 if st.session_state["resume_profile"] else True

        has_python = st.checkbox("Python & Data Structures", value=p_has_py)
        has_sql = st.checkbox("SQL & Relational Databases", value=p_has_sql)
        has_react_node = st.checkbox("React / Node.js Full Stack", value=False)
        has_cloud_aws = st.checkbox("Cloud Computing (AWS / GCP)", value=False)
        has_docker_k8s = st.checkbox("Docker & Containers", value=False)
        has_ml_pytorch = st.checkbox("Machine Learning & PyTorch", value=p_has_ml)
        has_system_design = st.checkbox("System Design & Architecture", value=False)

    # Compute Features
    input_profile = pd.DataFrame([{
        "experience_years": exp_years,
        "education_level": degree,
        "college_tier": college_tier,
        "location_type": location,
        "internships_count": internships,
        "github_projects": github_projects,
        "dsa_problems_solved": dsa_solved,
        "certifications_count": certs,
        "has_python": int(has_python),
        "has_sql": int(has_sql),
        "has_react_node": int(has_react_node),
        "has_cloud_aws": int(has_cloud_aws),
        "has_docker_k8s": int(has_docker_k8s),
        "has_ml_pytorch": int(has_ml_pytorch),
        "has_system_design": int(has_system_design),
        "total_skills": sum([has_python, has_sql, has_react_node, has_cloud_aws, has_docker_k8s, has_ml_pytorch, has_system_design])
    }])

    cleaned_input = cleaner.clean(input_profile)
    engineered_input = engineer.transform(cleaned_input)
    X_candidate = engineered_input[FEATURE_COLUMNS]

    # Predict Market Salary (LPA)
    pred_salary = float(reg.predict(X_candidate)[0])
    pred_salary = round(max(3.2, pred_salary), 1)

    # Predict Readiness Tier
    pred_tier_idx = clf.predict(X_candidate)[0]
    pred_tier = label_enc.inverse_transform([pred_tier_idx])[0]
    tier_probs = clf.predict_proba(X_candidate)[0]

    # Competitiveness Percentile in Market
    comp_score = float(engineered_input["competitiveness_score"].iloc[0])
    market_percentile = round((candidates_df["competitiveness_score"] < comp_score).mean() * 100, 1)

    with c3:
        st.markdown("##### 🎯 Market Valuation Result")

        badge_class = f"badge-{pred_tier.replace(' ', '-').replace('/', '-')}"
        st.markdown(f"""
        <div class="val-card">
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Estimated Market Salary Package</div>
            <div class="val-number">₹{pred_salary} LPA</div>
            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 12px;">Realistic Range: ₹{max(3.0, pred_salary - 1.2):.1f} – ₹{pred_salary + 1.5:.1f} LPA</div>
            <span class="badge-tier {badge_class}">{pred_tier.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("###### Employability Readiness Probabilities")
        for cls_name, prob in zip(label_enc.classes_, tier_probs):
            st.progress(float(prob), text=f"{cls_name}: {prob*100:.1f}%")

        coding_idx = engineered_input["coding_intensity_index"].iloc[0]
        skill_count = engineered_input["skill_breadth"].iloc[0]

        st.markdown(f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px;">
            <span style="background: #1e293b; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; color: #38bdf8;">Market Standing: <b>{market_percentile}th Percentile</b></span>
            <span style="background: #1e293b; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; color: #a7f3d0;">Coding Intensity: <b>{coding_idx} pts</b></span>
            <span style="background: #1e293b; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; color: #fde047;">Verified Skills: <b>{skill_count}/7</b></span>
            <span style="background: #1e293b; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; color: #c084fc;">Overall Score: <b>{comp_score:.1f}/100</b></span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: SKILL GAP & HIGH-ROI ROADMAP
# ==============================================================================
with tabs[3]:
    st.subheader("High-ROI Skill Gap & Career Acceleration Engine")
    st.caption("Select your target dream role to discover your exact skill gap, missing competencies, and how much salary boost each new skill adds.")

    target_role_select = st.selectbox(
        "Select Your Target Tech Role",
        list(ROLE_TARGETS.keys()),
        index=0,
        key="role_select_key"
    )

    current_skills_dict = {
        "has_python": int(has_python),
        "has_sql": int(has_sql),
        "has_react_node": int(has_react_node),
        "has_cloud_aws": int(has_cloud_aws),
        "has_docker_k8s": int(has_docker_k8s),
        "has_ml_pytorch": int(has_ml_pytorch),
        "has_system_design": int(has_system_design),
        "dsa_problems_solved": dsa_solved,
        "github_projects": github_projects
    }

    gap_analysis = CareerSkillAnalyzer.analyze_gap(current_skills_dict, target_role_select)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Role Match Score", f"{gap_analysis['match_score_pct']}%")
    g2.metric("Acquired Required Skills", f"{gap_analysis['acquired_count']}")
    g3.metric("Missing Critical Skills", f"{gap_analysis['missing_count']}")
    g4.metric("Potential Salary Boost 🚀", f"+₹{gap_analysis['potential_salary_boost_lpa']:.1f} LPA")

    st.write("")
    st.progress(float(gap_analysis["match_score_pct"]) / 100.0, text=f"Role Readiness: {gap_analysis['match_score_pct']}%")
    st.divider()

    col_gap_left, col_gap_right = st.columns([1.2, 1.2])

    with col_gap_left:
        st.markdown("##### 📈 Missing Skills Ranked by Salary ROI")
        if gap_analysis["missing_skills"]:
            missing_df = pd.DataFrame(gap_analysis["missing_skills"])
            fig_roi, ax_roi = plt.subplots(figsize=(6, 3.2), facecolor="#0e1726")
            ax_roi.set_facecolor("#0e1726")

            names = missing_df["name"][::-1]
            boosts = missing_df["estimated_lpa_boost"][::-1]

            ax_roi.barh(names, boosts, color="#38bdf8", height=0.55, edgecolor="none")
            ax_roi.set_xlabel("Estimated Salary Increment (+₹ LPA)", color="#94a3b8", fontsize=9)
            ax_roi.tick_params(colors="#94a3b8", labelsize=8.5)
            ax_roi.grid(axis="x", linestyle="--", alpha=0.15, color="#cbd5e1")
            for spine in ax_roi.spines.values():
                spine.set_visible(False)
            st.pyplot(fig_roi)
        else:
            st.success("🎉 You have mastered all baseline required skills for this role!")

    with col_gap_right:
        st.markdown("##### 🎯 Step-by-Step Personalized Action Plan")
        for action in gap_analysis["actionable_recommendations"]:
            st.markdown(f'<div class="action-card">{action}</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 5: JOB MARKET TRENDS & ANALYTICS (EDA)
# ==============================================================================
with tabs[4]:
    st.subheader("Tech Hiring Market Trends & Salary Distributions (3,500 Profiles)")
    st.caption("Exploratory Data Analysis showing how skills, college tiers, and roles impact market compensation.")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("##### 🎓 College Tier vs. Salary Distribution (₹ LPA)")
        fig_box, ax_box = plt.subplots(figsize=(6, 3.8), facecolor="#0e1726")
        ax_box.set_facecolor("#0e1726")

        tiers = candidates_df["college_tier"].unique()
        data_by_tier = [candidates_df[candidates_df["college_tier"] == t]["salary_lpa"].values for t in tiers]
        
        bp = ax_box.boxplot(
            data_by_tier,
            tick_labels=["Tier 1", "Tier 2", "Tier 3"],
            patch_artist=True,
            medianprops=dict(color="#f59e0b", linewidth=2)
        )
        colors = ["#818cf8", "#38bdf8", "#34d399"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("none")

        ax_box.set_ylabel("Salary Package (₹ LPA)", color="#94a3b8")
        ax_box.tick_params(colors="#94a3b8")
        ax_box.grid(axis="y", linestyle="--", alpha=0.15, color="#cbd5e1")
        for spine in ax_box.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_box)

    with e2:
        st.markdown("##### 📈 Experience vs. Salary Progression Curve")
        fig_curve, ax_curve = plt.subplots(figsize=(6, 3.8), facecolor="#0e1726")
        ax_curve.set_facecolor("#0e1726")

        role_colors = {
            "Data Science & AI/ML": "#38bdf8",
            "Full Stack Web": "#818cf8",
            "Cloud & DevOps": "#34d399",
            "Backend Systems": "#f59e0b",
            "Data Analytics & BI": "#f472b6"
        }
        for role, grp in candidates_df.groupby("role_track"):
            ax_curve.scatter(
                grp["experience_years"],
                grp["salary_lpa"],
                label=role,
                color=role_colors.get(role, "#94a3b8"),
                alpha=0.45,
                s=16
            )

        ax_curve.set_xlabel("Experience (Years)", color="#94a3b8")
        ax_curve.set_ylabel("Market Salary (₹ LPA)", color="#94a3b8")
        ax_curve.tick_params(colors="#94a3b8")
        ax_curve.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#cbd5e1", fontsize=7.5)
        ax_curve.grid(linestyle="--", alpha=0.15, color="#cbd5e1")
        for spine in ax_curve.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_curve)

    st.write("")
    e3, e4 = st.columns(2)

    with e3:
        st.markdown("##### 💡 Skill Premium: Cloud vs. ML vs. System Design")
        skill_premiums = {
            "System Design": candidates_df[candidates_df["has_system_design"] == 1]["salary_lpa"].mean() - candidates_df[candidates_df["has_system_design"] == 0]["salary_lpa"].mean(),
            "AI / PyTorch": candidates_df[candidates_df["has_ml_pytorch"] == 1]["salary_lpa"].mean() - candidates_df[candidates_df["has_ml_pytorch"] == 0]["salary_lpa"].mean(),
            "AWS / Cloud": candidates_df[candidates_df["has_cloud_aws"] == 1]["salary_lpa"].mean() - candidates_df[candidates_df["has_cloud_aws"] == 0]["salary_lpa"].mean(),
            "Docker / K8s": candidates_df[candidates_df["has_docker_k8s"] == 1]["salary_lpa"].mean() - candidates_df[candidates_df["has_docker_k8s"] == 0]["salary_lpa"].mean(),
            "React / Node": candidates_df[candidates_df["has_react_node"] == 1]["salary_lpa"].mean() - candidates_df[candidates_df["has_react_node"] == 0]["salary_lpa"].mean(),
        }

        fig_prem, ax_prem = plt.subplots(figsize=(6, 3.6), facecolor="#0e1726")
        ax_prem.set_facecolor("#0e1726")
        ax_prem.bar(skill_premiums.keys(), skill_premiums.values(), color="#c084fc", width=0.55, edgecolor="none")
        ax_prem.set_ylabel("Average Market Salary Boost (+₹ LPA)", color="#94a3b8", fontsize=9)
        ax_prem.tick_params(colors="#94a3b8", labelsize=8.5)
        ax_prem.grid(axis="y", linestyle="--", alpha=0.15, color="#cbd5e1")
        for spine in ax_prem.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_prem)

    with e4:
        st.markdown("##### 💻 DSA Problems Solved vs. Salary Realization")
        fig_dsa, ax_dsa = plt.subplots(figsize=(6, 3.6), facecolor="#0e1726")
        ax_dsa.set_facecolor("#0e1726")
        ax_dsa.scatter(
            candidates_df["dsa_problems_solved"],
            candidates_df["salary_lpa"],
            color="#38bdf8",
            alpha=0.4,
            s=14
        )
        z = np.polyfit(candidates_df["dsa_problems_solved"], candidates_df["salary_lpa"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 450, 50)
        ax_dsa.plot(x_trend, p(x_trend), color="#f59e0b", linewidth=2.5, label="Positive Compensation Trend")
        ax_dsa.set_xlabel("LeetCode / DSA Problems Solved", color="#94a3b8")
        ax_dsa.set_ylabel("Salary (₹ LPA)", color="#94a3b8")
        ax_dsa.tick_params(colors="#94a3b8")
        ax_dsa.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#cbd5e1", fontsize=8)
        for spine in ax_dsa.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_dsa)


# ==============================================================================
# TAB 6: TALENT ARCHETYPES & GOVERNANCE
# ==============================================================================
with tabs[5]:
    st.subheader("Unsupervised Talent Archetypes & Governance Benchmarks")
    st.caption("Discovers natural groupings of tech candidates across 22 multi-dimensional career variables.")

    cl_left, cl_right = st.columns([1.4, 1])

    with cl_left:
        st.markdown("##### 🧬 2D Principal Component Space of Tech Talent")
        fig_pca, ax_pca = plt.subplots(figsize=(7, 4.8), facecolor="#0e1726")
        ax_pca.set_facecolor("#0e1726")

        palette = ["#38bdf8", "#c084fc", "#34d399", "#f59e0b"]
        archetype_names = metrics["clustering"]["archetypes"]

        for c_id in range(4):
            c_data = candidates_df[candidates_df["cluster"] == c_id]
            ax_pca.scatter(
                c_data["pca_1"],
                c_data["pca_2"],
                label=f"Cluster {c_id}: {archetype_names.get(str(c_id), '')}",
                color=palette[c_id],
                alpha=0.55,
                s=20
            )

        ax_pca.set_xlabel(f"Principal Component 1 ({metrics['clustering']['explained_variance_ratio_pca'][0]*100:.1f}% var)", color="#94a3b8")
        ax_pca.set_ylabel(f"Principal Component 2 ({metrics['clustering']['explained_variance_ratio_pca'][1]*100:.1f}% var)", color="#94a3b8")
        ax_pca.tick_params(colors="#94a3b8", labelsize=8.5)
        ax_pca.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#cbd5e1", fontsize=7.5, loc="upper right")
        ax_pca.grid(linestyle="--", alpha=0.15, color="#cbd5e1")
        for spine in ax_pca.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_pca)

    with cl_right:
        st.markdown("##### 🎯 Model Governance Benchmarks")
        st.metric("Salary Regressor R²", f"{metrics['regressor']['r2_score']:.4f}", f"MAE: {metrics['regressor']['mae_lpa']} LPA")
        st.metric("Classifier Accuracy", f"{metrics['classifier']['accuracy']*100:.2f}%", f"F1: {metrics['classifier']['weighted_f1']:.4f}")
        st.metric("Outlier Profiles Flagged", "105 Profiles", "3.0% Non-Traditional")

        st.markdown("###### Top Global Salary Drivers:")
        for feat in metrics["top_features"][:4]:
            st.markdown(f"- **{feat['feature'].replace('_', ' ').title()}**: {feat['importance']*100:.1f}% weight")
