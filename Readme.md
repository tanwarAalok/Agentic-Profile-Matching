# Agentic Candidate Matching & Screening System

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
├── .env                  # OpenRouter API credentials and model config
├── requirements.txt      # Project dependencies
├── sample_resumes.json   # Candidate resume database
├── matching_agent.py     # LangGraph workflow, state schema, and tools
├── app.py                # Streamlit user interface and interactive chat
└── README.md             # Project documentation
```
