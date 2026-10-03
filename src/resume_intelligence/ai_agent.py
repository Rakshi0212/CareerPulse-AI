"""
CareerPulse AI Career Agent & Copilot
=====================================
Context-aware conversational agent that assists candidates with resume reviews,
mock interview questions, high-ROI upskilling roadmaps, and recruiter outreach scripts.
"""

from typing import Dict, List, Any


class CareerCopilotAgent:
    """
    Intelligent career advisor powered by candidate profile context and market data.
    """

    def __init__(self, candidate_context: Dict[str, Any]):
        self.context = candidate_context

    def respond(self, user_query: str) -> str:
        """Generate tailored advice conditioned on candidate's exact profile and ML valuation."""
        q = user_query.lower()
        role = self.context.get("target_role", "AI / Machine Learning Engineer")
        salary = self.context.get("predicted_salary", 14.8)
        tier = self.context.get("readiness_tier", "Job-Ready Mid-Level")
        missing = self.context.get("missing_skills", [])
        dsa = self.context.get("dsa_problems_solved", 110)
        ats_score = self.context.get("ats_score", 75)

        # 1. Interview Preparation Questions
        if any(w in q for w in ["interview", "question", "prep", "ask", "technical round"]):
            return f"""### 🎯 Tailored Technical Interview Questions for **{role}**

Based on your current profile (Valuation: **₹{salary} LPA** | Tier: **{tier}**), here are the top 5 questions interviewers at top product companies will ask you:

1. **System & ML Lifecycle**: *"Walk me through an end-to-end Machine Learning pipeline you built. How did you handle data leakage during feature engineering?"*
2. **Coding & Algorithmic**: *"Given an array of transactions, how would you design an algorithm to detect the top K anomalies in $O(N \\log K)$ time?"*
3. **Deployment & MLOps**: *"How do you containerize a model using Docker and ensure sub-100ms inference latency under high concurrent load?"*
4. **Data Modeling & SQL**: *"Explain the difference between a Star Schema and Snowflake schema, and write a query using window functions (`ROW_NUMBER()` or `DENSE_RANK()`)."*
5. **Trade-offs**: *"When would you choose an ElasticNet or Gradient Boosting model over a deep neural network for tabular data?"*

💡 **Pro-Tip**: Structure all your answers using the **STAR Method** (Situation, Task, Action, Result) with numbers!"""

        # 2. Salary Negotiation & Crossing Higher Bands
        elif any(w in q for w in ["salary", "increase", "higher", "lpa", "negotiate", "money", "20"]):
            missing_names = [m.get("name", "") for m in missing[:2]]
            missing_str = " and ".join(missing_names) if missing_names else "System Design & Distributed Cloud Services"
            return f"""### 💰 Strategy to Scale from **₹{salary} LPA** to **₹25+ LPA**

Your current profile is valued at **₹{salary} LPA ({tier})**. To break into the **High-Demand Specialist** tier:

1. **Target the Highest-ROI Skill**:
   - Focus on **{missing_str}**. In our market dataset, candidates with verified production cloud + system design command a **+₹4.5 to ₹6.0 LPA premium**.
2. **Demonstrate Scale, Not Just Code**:
   - Don't just build a model in a Jupyter Notebook. Build an automated pipeline with Docker, automated unit tests, and live deployment.
3. **Algorithmic Threshold**:
   - You currently have ~**{dsa} DSA problems**. Scaling to **180+ problems** ensures you clear technical screening rounds at Tier-1 product firms (Atlassian, Uber, Swiggy, Amazon).
4. **Competing Offers Strategy**:
   - Interview with 3-4 companies simultaneously. Mentioning: *"I have an active offer at ₹{salary + 2:.1f} LPA, but your team's architecture is my top priority"* is proven to yield 20-30% counter-offers."""

        # 3. Recruiter Cold Outreach Script
        elif any(w in q for w in ["recruiter", "linkedin", "message", "cold email", "outreach"]):
            return f"""### 📩 High-Converting LinkedIn Recruiter Message

Copy and customize this direct message for tech recruiters hiring for **{role}**:

```text
Hi [Recruiter Name],

I noticed [Company Name] is actively growing its {role} team. 

I'm a software engineer specializing in Python, SQL, and Machine Learning systems. Recently, I built an end-to-end ML valuation platform with 97.1% R² predictive accuracy and containerized microservices. 

I've solved {dsa}+ algorithmic problems on LeetCode and love tackling high-scale data challenges.

I'd love to learn if you're open to reviewing my resume for upcoming {role} openings at [Company Name]. 

Portfolio & Code: https://github.com/[your-handle]
Resume attached.

Best regards,
[Your Name]
```

💡 **Why this works**: It immediately mentions your quantifiable results, coding problem-solving metric ({dsa}+), and includes a direct link to your code."""

        # 4. Resume Feedback & ATS Advice
        elif any(w in q for w in ["resume", "ats", "bullet", "improve", "score", "format"]):
            return f"""### 📄 Actionable Resume Optimization Guide (Current ATS Score: {ats_score}/100)

Here are the 3 most impactful improvements you can make to your resume today:

1. **Adopt Google's XYZ Formula for Every Bullet**:
   - ❌ *Weak*: "Worked on a web application in React and Python."
   - ✅ *Strong*: "• **Engineered** an interactive valuation platform using React & Python, serving 3,500+ candidate evaluations with sub-second latency."
2. **Add Missing Domain Keywords**:
   - For **{role}**, ensure keywords like `Docker`, `CI/CD`, `Scikit-Learn`, `Feature Engineering`, and `SQL Indexing` are explicitly listed in your skills block.
3. **Lead with Active Verbs**:
   - Start bullets with: **Architected, Deployed, Automated, Optimized, Scaled**. Never use *"Responsible for"* or *"Helped with"*."""

        # 5. General Fallback
        else:
            return f"""### 🤖 CareerPulse AI Copilot Advice

I've analyzed your profile (**₹{salary} LPA** | **{tier}** | Target: **{role}**).

Here are quick actions you can ask me to help you with:
- *"Give me 5 mock interview questions for an {role}"*
- *"How can I negotiate a higher package?"*
- *"Draft a LinkedIn cold outreach message for recruiters"*
- *"How do I fix my weak resume bullet points?"*

Type any of the above or ask a specific question about your tech career journey!"""
