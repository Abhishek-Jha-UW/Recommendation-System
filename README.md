# AI Recommendation Engine (Portfolio)

Streamlit app demonstrating **user-based collaborative filtering**, **item-based collaborative filtering**, and **item co-occurrence** (“frequently rated together”) on user–item–rating data. Built as a **data scientist / analyst portfolio** piece: clear methods, honest limitations, and optional AI-assisted interpretation.

A frozen copy of the original app lives in [`initial_data/`](initial_data/) for reference.

---

## Features (current)

- **Tabs:** Overview, Data (summary + table), Methods (reference table), Playground, Evaluation.
- **Strategies:** user-based CF, item-based CF, item co-occurrence (honest naming).
- **Explainability:** similar users (user-based), contribution breakdown (item-based), co-occurrence breakdown (co-occurrence).
- **Data tab:** observed rating min/max/mean/std; optional **expected range** validation (warnings only).
- **Item co-occurrence:** summed counts for ranking; **support / confidence / lift** for antecedent → top pick in Playground.
- **Hold-out RMSE / MAE** for user-based CF (rough sanity check).
- **Optional OpenAI:** grounded explanations in **Playground** (recommendations) and **Evaluation** (hold-out metrics).

---

## Roadmap

Detailed, phased tasks (explainability, evaluation, naming accuracy, deployment, optional OpenAI) are in **[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)**. Use that file as the implementation checklist when opening PRs or issues.

---

## Requirements

- Python 3.10+ recommended.
- Dependencies: see [`requirements.txt`](requirements.txt) (includes `streamlit` and `openai` for optional AI).

---

## Local setup

```bash
git clone <your-repo-url>
cd "18. Recommendation System"
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Data format

The engine expects a table whose **first three columns** are interpreted as:

| Position | Meaning   | Example column names |
|----------|-----------|----------------------|
| 1        | User id   | `user_id`, `User`    |
| 2        | Item id   | `item_id`, `Item`    |
| 3        | Rating    | `rating`, `Rating`   |

Ratings should be numeric where applicable. Extra columns may exist but are ignored if they appear **after** the first three.

The app’s template uses `user_id`, `item_id`, `rating`.

---

## Deploying (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. In [Streamlit Cloud](https://streamlit.io/cloud), connect the repo, set **Main file path** to `app.py` and Python version if prompted.
3. Add **Secrets** only for optional features (see below). The app must run **without** secrets for core recommendations.

---

## Optional: OpenAI (Streamlit secrets)

Use a **small** chat model for short, **grounded** explanations (facts from the app only).

### Streamlit Community Cloud

In **App settings → Secrets**, paste TOML, for example:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
```

Nested form is also supported:

```toml
[openai]
api_key = "sk-..."
model = "gpt-4o-mini"
```

Alternate top-level names: `openai_api_key`, `OPENAI_KEY`; optional `openai_model`.

### Local run

Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml`, fill in the key, and run `streamlit run app.py`. The file `secrets.toml` is listed in `.gitignore`.

You can instead set environment variables `OPENAI_API_KEY` and optionally `OPENAI_MODEL`.

### App behavior

The sidebar shows **OpenAI: ready** when a key is loaded. After you generate recommendations, use **Explain this run (AI)** in the Playground (one API call per click).

| Secret key        | Purpose                          |
|-------------------|----------------------------------|
| `OPENAI_API_KEY`  | API key (never commit to git)    |
| `OPENAI_MODEL`    | Optional; defaults to `gpt-4o-mini` |

Cost controls: single call per explicit button click, low `max_tokens`, low temperature, system prompt restricted to provided JSON facts.

---

## Project layout

```
.
├── app.py                 # Streamlit UI (tabs, playground, eval, optional AI)
├── model.py               # RecommenderEngine + dataset stats + hold-out eval
├── data.py                # Built-in sample ratings (no Streamlit)
├── ai_explain.py          # Optional OpenAI call (grounded JSON only)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example   # Copy to secrets.toml locally (gitignored)
├── initial_data/          # Snapshot of original files (baseline)
├── docs/
│   └── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## License

Add a `LICENSE` file when you publish (e.g. MIT) if you want others to reuse the code.

---

## Author

Portfolio project — update this section with your name and links.
