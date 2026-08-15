"""
app.py - Interactive Recruiter Dashboard and Conversational Interface
"""

import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from matching_agent import matching_agent, multi_round_screening_node, generate_report_node, load_all_resumes

st.set_page_config(
    page_title="AI Talent Matcher & Screening Agent",
    page_icon="💼",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "raw_jd": "",
        "parsed_requirements": {},
        "candidate_pool": load_all_resumes(),
        "shortlist": [],
        "comparison_report": "",
        "active_filters": {"min_exp": 3.0, "must_haves": ["React"]},
        "iteration_note": ""
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar: JD Setup & Screening Trigger
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Recruitment Studio")
    st.markdown("### 1. Job Description")

    sample_jd = """Job Title: Senior Frontend Engineer
Experience: 3+ years
Must-Have Skills: React, TypeScript, Next.js
Nice-to-Have Skills: AWS, GraphQL, Docker
Education: BS in Computer Science or equivalent
Responsibilities: Build scalable, responsive frontend applications and collaborate with backend microservices."""

    jd_text = st.text_area("Paste or Edit Job Description:", value=sample_jd, height=180)

    if st.button("🚀 Run Multi-Round Screening", type="primary", use_container_width=True):
        with st.spinner("Executing Round 1 (RAG Retrieval) -> Round 2 (Deep Scoring) -> Round 3 (Rankings)..."):
            st.session_state.agent_state["raw_jd"] = jd_text
            st.session_state.agent_state["parsed_requirements"] = {
                "job_title": "Senior Frontend Engineer",
                "min_years_experience": 3.0,
                "must_have_skills": ["React", "TypeScript"],
                "nice_to_have_skills": ["Next.js", "AWS", "GraphQL"]
            }
            # Execute screening & reporting pipeline
            screened_state = multi_round_screening_node(st.session_state.agent_state)
            st.session_state.agent_state.update(screened_state)

            report_state = generate_report_node(st.session_state.agent_state)
            st.session_state.agent_state.update(report_state)

            st.session_state.chat_history.append(
                AIMessage(
                    content="✅ Multi-round screening complete! I have evaluated and ranked candidates based on the JD.")
            )
            st.success("Screening Complete!")

    st.markdown("---")
    st.markdown("### 2. Iterative Refinement")
    min_exp_slider = st.slider("Minimum Experience (Years)", 0.0, 8.0, 3.0, 0.5)
    added_requirement = st.text_input("Add Must-Have Constraint (e.g. AWS, Next.js)")

    if st.button("🔄 Re-Rank Candidates", use_container_width=True):
        with st.spinner("Adjusting weights and re-evaluating candidate pool..."):
            reqs = st.session_state.agent_state.get("parsed_requirements", {})
            reqs["min_years_experience"] = min_exp_slider
            if added_requirement:
                if "must_have_skills" not in reqs:
                    reqs["must_have_skills"] = []
                if added_requirement not in reqs["must_have_skills"]:
                    reqs["must_have_skills"].append(added_requirement)

            st.session_state.agent_state["parsed_requirements"] = reqs

            # Re-run screening
            screened_state = multi_round_screening_node(st.session_state.agent_state)
            st.session_state.agent_state.update(screened_state)
            report_state = generate_report_node(st.session_state.agent_state)
            st.session_state.agent_state.update(report_state)

            refinement_msg = f"🔄 Re-ranked candidates with new criteria: Min Exp = {min_exp_slider} yrs, Added Skill = '{added_requirement}'."
            st.session_state.chat_history.append(AIMessage(content=refinement_msg))
            st.info("Shortlist re-ranked!")

# ---------------------------------------------------------------------------
# Main Layout: Tabs for Chat, Ranked Shortlist, and Comparisons
# ---------------------------------------------------------------------------
tab_chat, tab_shortlist, tab_compare = st.tabs(
    ["💬 Conversational Recruiter", "📊 Ranked Shortlist", "⚖️ Head-to-Head Compare"])

# Tab 1: Conversational Chat
with tab_chat:
    st.subheader("Interactive Recruiting Assistant")
    st.caption("Ask questions, request interview questions, or ask why one candidate ranked higher than another.")

    # Render previous messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    user_input = st.chat_input("E.g., Why did Priya rank higher than David? or Generate interview questions for C101")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            with st.spinner("Reasoning..."):
                result = matching_agent.invoke(st.session_state.agent_state)
                st.session_state.agent_state.update(result)
                last_reply = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
                st.markdown(last_reply.content)
                st.session_state.chat_history.append(last_reply)

# Tab 2: Ranked Shortlist & Explainability Dossier
with tab_shortlist:
    shortlist = st.session_state.agent_state.get("shortlist", [])
    if not shortlist:
        st.info("No candidates screened yet. Click **'Run Multi-Round Screening'** in the sidebar to start.")
    else:
        st.subheader(f"Screened Candidates ({len(shortlist)} Profiles)")
        for idx, cand in enumerate(shortlist, 1):
            with st.expander(
                    f"#{idx}. {cand['name']} — Score: {cand['match_score']}/100 ({cand.get('recommendation')})",
                    expanded=(idx <= 2)):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**ID:** `{cand['candidate_id']}`")
                    st.write(f"**Experience:** {cand['experience_years']} Years")
                    st.write(f"**Education:** {cand.get('education', 'N/A')}")
                    st.write(f"**Recommendation:** `{cand.get('recommendation')}`")
                with col2:
                    st.write(f"**Skills:** {', '.join(cand.get('skills', []))}")
                    st.markdown("**Strengths:**")
                    for s in cand.get("strengths", []):
                        st.markdown(f"- {s}")
                    st.markdown("**Gaps:**")
                    for g in cand.get("gaps", []):
                        st.markdown(f"- {g}")
                    if cand.get("improvement_suggestions"):
                        st.info(f"💡 **Upskilling / Probe:** {cand['improvement_suggestions']}")

# Tab 3: Head-to-Head Comparison
with tab_compare:
    st.subheader("Side-by-Side Candidate Comparison")
    resumes = load_all_resumes()
    cand_options = {f"{c['candidate_id']} - {c['name']}": c['candidate_id'] for c in resumes}

    col_a, col_b = st.columns(2)
    with col_a:
        cand_a_label = st.selectbox("Select Candidate A:", list(cand_options.keys()),
                                    index=4 if len(cand_options) > 4 else 0)
    with col_b:
        cand_b_label = st.selectbox("Select Candidate B:", list(cand_options.keys()),
                                    index=1 if len(cand_options) > 1 else 0)

    if st.button("⚖️ Run Comparative Analysis"):
        id_a = cand_options[cand_a_label]
        id_b = cand_options[cand_b_label]

        with st.spinner("Analyzing skill sets, experience, and trade-offs..."):
            comp_query = f"Compare candidates {id_a} and {id_b} in detail. Explain trade-offs and who is a better fit."
            st.session_state.agent_state["messages"].append(HumanMessage(content=comp_query))
            res = matching_agent.invoke(st.session_state.agent_state)
            st.session_state.agent_state.update(res)
            last_msg = [m for m in res["messages"] if isinstance(m, AIMessage)][-1]
            st.markdown(last_msg.content)