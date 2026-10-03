"""
CareerPulse AI — Tech Career Skill Gap & Salary Valuation Platform
==================================================================
Interactive Streamlit Dashboard designed for Students, Freshers, and Tech Job Seekers:
- Live Market Salary Valuation & Readiness Tier Classifier
- High-ROI Skill Gap Analyzer & Learning Roadmap
- Tech Market Salary Analytics & Experience Curves
- Unsupervised Talent Archetypes (PCA Clustering)
- Model Benchmarks & Governance KPIs
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

# Page Setup
st.set_page_config(
    page_title="CareerPulse AI — Tech Career & Salary Valuation",
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

    /* Roadmap Action Card */
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_data():
    models_dir = os.path.join(BASE_DIR, "models", "saved_models")
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

# --- TOP HEADER ---
st.markdown("""
<div class="career-header">
    <div class="career-title">
        <span>🚀 CareerPulse AI</span>
    </div>
    <div class="career-subtitle">
        Tech Career Skill Gap, Market Salary Valuation & High-ROI Upskilling Engine for Students & Job Seekers
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation
tabs = st.tabs([
    "🎓 Student Valuation & Readiness",
    "🚀 Skill Gap & High-ROI Roadmap",
    "📊 Job Market Trends & Salary Analytics",
    "🧬 Talent Archetypes (PCA Clusters)",
    "🎯 Model Benchmarks & Governance"
])

# ==============================================================================
# TAB 1: STUDENT VALUATION & READINESS
# ==============================================================================
with tabs[0]:
    st.subheader("Personalized Tech Salary Valuation & Readiness Diagnostic")
    st.caption("Input your current college background, verified skills, and coding metrics to compute your fair market compensation package.")

    c1, c2, c3 = st.columns([1.1, 1.2, 1.5])

    with c1:
        st.markdown("##### 🏛️ Education & Experience")
        exp_years = st.slider("Work Experience (Years)", 0.0, 8.0, 0.5, 0.5, help="0 for fresh college graduates")
        degree = st.selectbox("Highest Degree", ["B.Tech/B.E", "BCA/B.Sc CS", "M.Tech/M.S", "MCA", "Non-CS Degree"])
        college_tier = st.selectbox("College Tier", ["Tier 1 (IIT/NIT/BITS)", "Tier 2 (Top State/Private)", "Tier 3 (Affiliated Colleges)"], index=1)
        location = st.selectbox("Target Location", ["Tier 1 Tech Hub (Bengaluru/NCR/Hyd)", "Tier 2 City", "Remote Global"])
        internships = st.slider("Internships Completed", 0, 3, 1)

    with c2:
        st.markdown("##### 💻 Coding & Portfolio Metrics")
        dsa_solved = st.slider("DSA / LeetCode Problems Solved", 0, 450, 110, 10)
        github_projects = st.slider("Production GitHub Projects", 0, 10, 3)
        certs = st.slider("Industry Certifications", 0, 4, 1)

        st.markdown("##### 🛠️ Verified Technical Skills")
        has_python = st.checkbox("Python & Data Structures", value=True)
        has_sql = st.checkbox("SQL & Relational Databases", value=True)
        has_react_node = st.checkbox("React / Node.js Full Stack", value=False)
        has_cloud_aws = st.checkbox("Cloud Computing (AWS / GCP)", value=False)
        has_docker_k8s = st.checkbox("Docker & Containers", value=False)
        has_ml_pytorch = st.checkbox("Machine Learning & PyTorch", value=True)
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

        # Metric Chips
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
# TAB 2: SKILL GAP & HIGH-ROI ROADMAP
# ==============================================================================
with tabs[1]:
    st.subheader("High-ROI Skill Gap & Career Acceleration Engine")
    st.caption("Select your target dream role to discover your exact skill gap, missing competencies, and how much salary boost each new skill adds.")

    target_role = st.selectbox(
        "Select Your Target Tech Role",
        list(ROLE_TARGETS.keys()),
        index=0
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

    gap_analysis = CareerSkillAnalyzer.analyze_gap(current_skills_dict, target_role)

    # Top KPI Metrics Row
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
        st.caption("Prioritized order of skills to learn based on market compensation premium.")

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
        st.caption("Targeted guidance to prepare for technical interview rounds.")
        for action in gap_analysis["actionable_recommendations"]:
            st.markdown(f'<div class="action-card">{action}</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3: JOB MARKET TRENDS & SALARY ANALYTICS (EDA)
# ==============================================================================
with tabs[2]:
    st.subheader("Tech Hiring Market Trends & Salary Distributions (3,500 Profiles)")
    st.caption("Exploratory Data Analysis showing how skills, college tiers, and roles impact market compensation.")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("##### 🎓 College Tier vs. Salary Distribution (₹ LPA)")
        st.caption("Notice how skills and open-source projects bridge the tier gap for high performers.")
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
# TAB 4: TALENT ARCHETYPES (PCA CLUSTERS)
# ==============================================================================
with tabs[3]:
    st.subheader("Unsupervised Talent Archetypes (K-Means & PCA)")
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
        st.markdown("##### 📊 Archetype Compensation & Profile Breakdown")
        cluster_summary = candidates_df.groupby("cluster_archetype").agg(
            Candidates=("candidate_id", "count"),
            Avg_Exp=("experience_years", "mean"),
            Avg_DSA=("dsa_problems_solved", "mean"),
            Avg_Salary_LPA=("salary_lpa", "mean")
        ).round(1)

        st.dataframe(cluster_summary, width="stretch")

        st.markdown("""
        **Talent Archetype Takeaways:**
        - **Cluster 0 (Core Backend)**: Highest average DSA count (~240+ problems), strong C++/Java system foundations.
        - **Cluster 1 (AI/ML Specialists)**: Premium salaries; heavy Python, SQL, and PyTorch deep learning stacks.
        - **Cluster 2 (Cloud Architects)**: Kubernetes, Docker, and AWS certified engineers commanding remote premiums.
        - **Cluster 3 (Full Stack / Entry)**: Freshers and junior web developers building foundational portfolio projects.
        """)


# ==============================================================================
# TAB 5: MODEL BENCHMARKS & GOVERNANCE
# ==============================================================================
with tabs[4]:
    st.subheader("Model Performance, Benchmarks & Explainability Governance")
    st.caption("Rigorous evaluation on held-out test cohort (20% split) for salary regression and readiness classification.")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Salary Regressor", metrics["regressor"]["model_type"])
    b2.metric("Regression R² Score", f"{metrics['regressor']['r2_score']:.4f}", f"MAE: {metrics['regressor']['mae_lpa']} LPA")
    b3.metric("Classifier Accuracy", f"{metrics['classifier']['accuracy']*100:.2f}%", f"Weighted F1: {metrics['classifier']['weighted_f1']:.4f}")
    b4.metric("Outlier Profiles Flagged", "105 Candidates", "3.0% Market Anomaly Rate")

    st.divider()

    m_left, m_right = st.columns(2)

    with m_left:
        st.markdown("##### 🎯 Readiness Tier Confusion Matrix")
        classes = metrics["classifier"]["classes"]
        cm = np.array(metrics["classifier"]["confusion_matrix"])

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.2), facecolor="#0e1726")
        ax_cm.set_facecolor("#0e1726")
        cax = ax_cm.matshow(cm, cmap="Blues")
        fig_cm.colorbar(cax)

        ax_cm.set_xticks(range(len(classes)))
        ax_cm.set_yticks(range(len(classes)))
        ax_cm.set_xticklabels(["Entry", "High-Spec", "Mid-Level", "Junior"], color="#94a3b8", fontsize=8.5)
        ax_cm.set_yticklabels(["Entry", "High-Spec", "Mid-Level", "Junior"], color="#94a3b8", fontsize=8.5)
        ax_cm.set_xlabel("Predicted Tier", color="#94a3b8")
        ax_cm.set_ylabel("True Market Tier", color="#94a3b8")

        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j]
                ax_cm.text(j, i, str(val), ha="center", va="center", color="#ffffff" if val > cm.max()/2 else "#94a3b8", fontweight="bold")
        st.pyplot(fig_cm)

    with m_right:
        st.markdown("##### 🌟 Global Salary Driver Importance Ranking")
        top_feats = metrics["top_features"]
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4.2), facecolor="#0e1726")
        ax_imp.set_facecolor("#0e1726")

        f_names = [f["feature"].replace("_", " ").title() for f in top_feats][::-1]
        f_vals = [f["importance"] for f in top_feats][::-1]

        ax_imp.barh(f_names, f_vals, color="#38bdf8", height=0.6, edgecolor="none")
        ax_imp.set_xlabel("Relative Importance Weight", color="#94a3b8")
        ax_imp.tick_params(colors="#94a3b8", labelsize=8.5)
        ax_imp.grid(axis="x", linestyle="--", alpha=0.15, color="#cbd5e1")
        for spine in ax_imp.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_imp)
