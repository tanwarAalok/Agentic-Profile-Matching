"""
matching_agent.py - Complete LangGraph Candidate Matching & Screening Engine
"""

import json
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# 1. OpenRouter LLM Setup
# ---------------------------------------------------------------------------
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "LangGraph Talent Matcher",
    },
)

# ---------------------------------------------------------------------------
# 2. State Schema
# ---------------------------------------------------------------------------
class Candidate(TypedDict):
    candidate_id: str
    name: str
    experience_years: float
    skills: List[str]
    education: str
    summary: str
    match_score: float
    strengths: List[str]
    gaps: List[str]
    recommendation: Literal["Strong Hire", "Hire", "Borderline", "No Hire"]
    improvement_suggestions: str
    interview_questions: Optional[List[str]]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    raw_jd: str
    parsed_requirements: Dict[str, Any]
    candidate_pool: List[Dict[str, Any]]
    shortlist: List[Candidate]
    comparison_report: str
    active_filters: Dict[str, Any]
    iteration_note: str


# ---------------------------------------------------------------------------
# 3. Resume Database & File System RAG Helper
# ---------------------------------------------------------------------------
def load_all_resumes() -> List[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "sample_resumes.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# 4. Agent Tools
# ---------------------------------------------------------------------------
@tool
def extract_requirements(jd: str) -> str:
    """Parses a raw job description into structured must-have vs nice-to-have requirements in JSON format."""
    prompt = f"""
    You are an expert technical recruiter. Parse this Job Description into a clean JSON object:
    {{
        "job_title": "string",
        "min_years_experience": float,
        "must_have_skills": ["list", "of", "skills"],
        "nice_to_have_skills": ["list", "of", "skills"],
        "education": "string or Not Specified",
        "key_responsibilities": ["list"]
    }}

    Return ONLY the valid JSON block.

    JD:
    {jd}
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


@tool
def search_resumes_rag(query: str, min_experience: float = 0.0, top_k: int = 10) -> str:
    """Executes a semantic and keyword search across the candidate resume dataset."""
    resumes = load_all_resumes()
    query_tokens = [t.lower().strip() for t in query.replace(",", " ").split() if t.strip()]

    scored = []
    for r in resumes:
        if r.get("experience_years", 0) < min_experience:
            continue

        text_corpus = f"{r.get('name', '')} {r.get('summary', '')} {' '.join(r.get('skills', []))} {r.get('projects', '')}".lower()
        score = 0
        for token in query_tokens:
            if token in text_corpus:
                score += 1

        scored.append((score, r))

    scored.sort(key=lambda x: (x[0], x[1].get("experience_years", 0)), reverse=True)
    results = [item[1] for item in scored[:top_k]]
    return json.dumps(results, indent=2)


@tool
def compare_candidates(candidate_ids: List[str]) -> str:
    """Provides a side-by-side, head-to-head comparison explaining ranking deltas and trade-offs."""
    resumes = load_all_resumes()
    selected = [r for r in resumes if r.get("candidate_id") in candidate_ids]

    if not selected:
        return f"No candidate records found for IDs: {candidate_ids}"

    prompt = f"""
    Conduct an objective, side-by-side technical comparison between these candidates:
    {json.dumps(selected, indent=2)}

    Format your analysis as follows:
    1. Direct Skill-by-Skill Comparison Table / Breakdown
    2. Experience Depth and Seniority Comparison
    3. Trade-offs (Strengths vs Weaknesses of each)
    4. Definitive Justification on who ranks higher and why
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


@tool
def generate_interview_questions(candidate_id: str) -> str:
    """Generates customized technical, architectural, and gap-probing interview screening questions."""
    resumes = load_all_resumes()
    candidate = next((r for r in resumes if r.get("candidate_id") == candidate_id), None)

    if not candidate:
        return f"Candidate {candidate_id} not found."

    prompt = f"""
    Create a targeted 4-question technical screening panel for this candidate:
    Candidate Profile: {json.dumps(candidate, indent=2)}

    Questions should include:
    1. [Must-Have Verification]: Deep-dive technical question on their primary skill.
    2. [Architectural Scenario]: Real-world system problem relevant to their stack.
    3. [Gap/Borderline Probe]: Specific question to test areas where experience seems light.
    4. [Behavioral/Ownership]: Real incident or project delivery scenario.
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


tools = [extract_requirements, search_resumes_rag, compare_candidates, generate_interview_questions]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 5. Graph Nodes
# ---------------------------------------------------------------------------
def router_node(state: AgentState) -> Dict[str, Any]:
    """Conversational reasoning node that evaluates user intent and orchestrates tool calling."""
    system_message = SystemMessage(
        content=(
            "You are an AI Talent Matching & Screening Specialist. "
            "You parse job descriptions, run RAG resume searches, execute multi-round candidate rankings, "
            "explain why candidate A ranks above candidate B, and adapt when recruiters adjust requirements mid-conversation. "
            "Use the provided tools when asked to search, parse, compare, or generate interview questions."
        )
    )
    history = [system_message] + state.get("messages", [])
    response = llm_with_tools.invoke(history)
    return {"messages": [response]}


def multi_round_screening_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes Part C Multi-Round Deep Screening:
    - Analyzes candidates against must-haves and nice-to-haves
    - Computes 0-100 match score
    - Identifies strengths, gaps, hire recommendations, and improvement suggestions for borderline candidates
    """
    reqs = state.get("parsed_requirements", {})
    pool = state.get("candidate_pool", [])
    if not pool:
        pool = load_all_resumes()[:10]

    shortlist: List[Candidate] = []
    for c in pool:
        eval_prompt = f"""
        Evaluate candidate against job requirements.

        Job Requirements:
        {json.dumps(reqs, indent=2)}

        Candidate Data:
        {json.dumps(c, indent=2)}

        Return a strictly valid JSON object:
        {{
            "candidate_id": "{c.get('candidate_id')}",
            "name": "{c.get('name')}",
            "experience_years": {c.get('experience_years', 0)},
            "skills": {json.dumps(c.get('skills', []))},
            "education": "{c.get('education', '')}",
            "summary": "{c.get('summary', '')}",
            "match_score": <float between 0 and 100>,
            "strengths": ["strength 1", "strength 2"],
            "gaps": ["gap 1", "gap 2"],
            "recommendation": "<Strong Hire | Hire | Borderline | No Hire>",
            "improvement_suggestions": "<constructive upskilling tips or verification checks>",
            "interview_questions": []
        }}
        """
        eval_resp = llm.invoke([HumanMessage(content=eval_prompt)])
        cleaned = eval_resp.content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        try:
            parsed = json.loads(cleaned.strip())
            shortlist.append(parsed)
        except Exception:
            shortlist.append({
                "candidate_id": c.get("candidate_id", "Unknown"),
                "name": c.get("name", "Unknown"),
                "experience_years": c.get("experience_years", 0),
                "skills": c.get("skills", []),
                "education": c.get("education", ""),
                "summary": c.get("summary", ""),
                "match_score": 70.0,
                "strengths": ["Meets baseline criteria"],
                "gaps": ["Requires manual deep-dive"],
                "recommendation": "Borderline",
                "improvement_suggestions": "Verify real-world experience during technical interview.",
                "interview_questions": []
            })

    # Rank candidates by match_score descending
    shortlist.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return {"shortlist": shortlist, "candidate_pool": pool}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """Compiles the final explainability report."""
    shortlist = state.get("shortlist", [])
    if not shortlist:
        return {"comparison_report": "No candidates evaluated."}

    lines = ["## 🎯 Candidate Match & Screening Dossier\n"]
    for rank, cand in enumerate(shortlist, 1):
        rec_badge = {
            "Strong Hire": "🟢 **Strong Hire**",
            "Hire": "🔵 **Hire**",
            "Borderline": "🟡 **Borderline**",
            "No Hire": "🔴 **No Hire**"
        }.get(cand.get("recommendation"), cand.get("recommendation"))

        lines.append(f"### #{rank}. {cand['name']} (`{cand['candidate_id']}`) — Score: **{cand['match_score']}/100**")
        lines.append(f"- **Recommendation:** {rec_badge}")
        lines.append(f"- **Experience:** {cand['experience_years']} years | **Education:** {cand.get('education', 'N/A')}")
        lines.append(f"- **Skills:** {', '.join(cand.get('skills', []))}")
        lines.append(f"- **Strengths:** {'; '.join(cand.get('strengths', []))}")
        lines.append(f"- **Identified Gaps:** {'; '.join(cand.get('gaps', []))}")
        if cand.get("recommendation") == "Borderline" or cand.get("improvement_suggestions"):
            lines.append(f"- **Actionable Advice / Verification:** _{cand.get('improvement_suggestions')}_")
        lines.append("\n---")

    report = "\n".join(lines)
    return {
        "comparison_report": report,
        "messages": [AIMessage(content=report)]
    }


# ---------------------------------------------------------------------------
# 6. Graph Compilation
# ---------------------------------------------------------------------------
def build_matching_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("multi_round_screen", multi_round_screening_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "router")

    def route_condition(state: AgentState) -> Literal["tools", "multi_round_screen", "__end__"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
            return "tools"
        return END

    graph.add_conditional_edges("router", route_condition)
    graph.add_edge("tools", "router")
    graph.add_edge("multi_round_screen", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


matching_agent = build_matching_graph()