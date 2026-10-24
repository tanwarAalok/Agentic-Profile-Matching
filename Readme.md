# Agentic Candidate Matching & Screening System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white) ![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6566F1)

An autonomous recruitment screening agent built with LangGraph, OpenRouter, and Streamlit. The system parses unstructured Job Descriptions (JDs), performs multi-round candidate screening, generates explainable match reports, conducts head-to-head candidate comparisons, and dynamically re-ranks candidate pools when criteria change mid-conversation.

## Key Capabilities

### Part A: Agent Architecture (LangGraph)

- **State-driven workflow:** Manages persistent state (AgentState) tracking conversation history, parsed JD criteria, candidate pools, and evaluated shortlists.

- **Integrated tool suite:**
 - `extract_requirements` — Parses unstructured JDs into structured must-have vs. nice-to-have parameters.
 - `search_resumes_rag` — Executes semantic and keyword search across candidate profiles.
 - `compare_candidates` — Generates head-to-head trade-off analyses between candidate IDs.
 - `generate_interview_questions` — Creates targeted technical screening and gap-probing questions.

### Part B: Interactive & Iterative Features

- **Conversational recruiter chat:** Query rankings, request candidate comparisons, and generate custom interview panels via natural language.
- **Mid-stream refinement:** Adjust criteria (experience sliders, mandatory skills) on the fly with automatic re-scoring and rank-shift explanations.

### Part C: Multi-Round Screening & Explainability

- **Round 1 — Pool Retrieval:** Filters resumes by baseline skills and experience.
- **Round 2 — Deep Scoring:** Evaluates candidate attributes on a 0–100 match scale.
- **Round 3 — Decision & Upskilling:** Assigns badges (Strong Hire, Hire, Borderline, No Hire), details specific strengths/gaps, and provides actionable recommendations for borderline profiles.

## State Machine Architecture

```
                  ┌─────────────────────────┐
                  │     START (User/JD)     │
                  └────────────┬────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
     ┌────────────►│      router_node      │◄─────────────┐
     │             └───────────┬───────────┘              │
     │                         │                          │
     │ (Tool call detected)    │ (Screening complete)     │
     ▼                         ▼                          │
┌──────────────┐         ┌───────────────────────┐          │
│  ToolNode    │         │  multi_round_screen   │          │
│ (RAG/Compare/│         └───────────┬───────────┘          │
│ Interview Qs)│                     │                      │
└──────┬───────┘                     ▼                      │
       │                 ┌───────────────────────┐          │
       └────────────────►│    generate_report    ├──────────┘
                         └───────────┬───────────┘ (Human Refinement Loop)
                                     │
                                     ▼
                                  [ END ]
```

## Project Structure

```
├── .env                  # OpenRouter API credentials and model config (create it, not committed)
├── requirements.txt      # Project dependencies
├── sample_resumes.json   # Candidate resume database
├── matching_agent.py     # LangGraph workflow, state schema, and tools
├── app.py                # Streamlit user interface and interactive chat
└── README.md             # Project documentation
```

## Getting Started

Requires Python 3.10+ and an [OpenRouter](https://openrouter.ai/) API key.

```bash
git clone https://github.com/tanwarAalok/Agentic-Profile-Matching.git
cd Agentic-Profile-Matching
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=openai/gpt-4o   # optional, any OpenRouter chat model
```

Run the app and open http://localhost:8501:

```bash
streamlit run app.py
```

Paste a job description in the sidebar, run the screening, then use the **Conversational Recruiter**, **Ranked Shortlist** and **Head-to-Head Compare** tabs to question the rankings, change criteria and compare candidates. The candidate pool comes from `sample_resumes.json` (10 sample profiles); replace it with your own data in the same shape.

## Tech Stack

- **Agent orchestration:** LangGraph (`StateGraph`, `ToolNode`), LangChain tools
- **LLM access:** OpenRouter through `langchain-openai`
- **UI:** Streamlit
- **Schemas:** Pydantic and typed agent state
