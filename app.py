import io
import os
from typing import Any

import pandas as pd
import streamlit as st

from data import build_sample_dataframe
from model import RecommenderEngine, dataset_summary, rating_column_stats

try:
    from ai_explain import explain_run_json

    _HAS_AI_MODULE = True
except ImportError:
    _HAS_AI_MODULE = False

st.set_page_config(
    page_title="Recommendation Studio | Portfolio",
    layout="wide",
    page_icon="📊",
)

# --- Sample data ---
@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return build_sample_dataframe()


def validate_rating_df(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    if df is None or df.empty:
        return None, "The table is empty."
    if df.shape[1] < 3:
        return None, "Need at least three columns: user, item, rating (in that order)."
    out = df.iloc[:, :3].copy()
    out.columns = ["user", "item", "rating"]
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    dropped = int(out["rating"].isna().sum())
    out = out.dropna(subset=["rating"])
    if out.empty:
        return None, "No valid numeric ratings after cleaning the rating column."
    return out, (f"Dropped {dropped} rows with non-numeric ratings." if dropped else None)


@st.cache_data(show_spinner="Loading…")
def cached_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def init_session() -> None:
    if "last_evidence" not in st.session_state:
        st.session_state.last_evidence = None
    if "last_eval_evidence" not in st.session_state:
        st.session_state.last_eval_evidence = None


def get_openai_credentials() -> tuple[str | None, str]:
    """
    Resolve API key and model from Streamlit secrets and/or environment.
    Supports: OPENAI_API_KEY, openai_api_key; nested [openai] api_key / model;
    OPENAI_MODEL; env vars for local/docker.
    """
    default_model = "gpt-4o-mini"
    key: str | None = None
    model = default_model

    try:
        sec = st.secrets
    except Exception:
        sec = None

    if sec is not None:
        for kname in ("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY"):
            if kname in sec:
                raw = sec[kname]
                if raw is not None and str(raw).strip():
                    key = str(raw).strip()
                    break

        if key is None and "openai" in sec:
            try:
                block = sec["openai"]
                if hasattr(block, "get"):
                    for sub in ("api_key", "OPENAI_API_KEY", "key", "apikey"):
                        raw = block.get(sub)
                        if raw is not None and str(raw).strip():
                            key = str(raw).strip()
                            break
                    m = block.get("model") or block.get("OPENAI_MODEL")
                    if m is not None and str(m).strip():
                        model = str(m).strip()
                elif isinstance(block, dict):
                    for sub in ("api_key", "OPENAI_API_KEY", "key", "apikey"):
                        if sub in block and block[sub]:
                            key = str(block[sub]).strip()
                            break
                    m = block.get("model") or block.get("OPENAI_MODEL")
                    if m and str(m).strip():
                        model = str(m).strip()
                else:
                    for attr in ("api_key", "OPENAI_API_KEY", "key"):
                        raw = getattr(block, attr, None)
                        if raw and str(raw).strip():
                            key = str(raw).strip()
                            break
                    raw_m = getattr(block, "model", None) or getattr(block, "OPENAI_MODEL", None)
                    if raw_m and str(raw_m).strip():
                        model = str(raw_m).strip()
            except Exception:
                pass

        for mname in ("OPENAI_MODEL", "openai_model"):
            if mname in sec:
                raw = sec[mname]
                if raw is not None and str(raw).strip():
                    model = str(raw).strip()
                    break

    if not key:
        env_k = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_k:
            key = env_k
    env_m = os.environ.get("OPENAI_MODEL", "").strip()
    if env_m:
        model = env_m

    return key, model or default_model


init_session()

# --- Sidebar ---
with st.sidebar:
    st.header("Control panel")
    strategy = st.radio(
        "Strategy",
        [
            "User-Based CF",
            "Item-Based CF",
            "Item co-occurrence (binary)",
        ],
        help=(
            "User-based: neighbors with similar rating vectors. "
            "Item-based: items similar to those you rated. "
            "Co-occurrence: counts plus support / confidence / lift for item pairs (Playground)."
        ),
    )
    st.divider()
    data_option = st.selectbox("Data source", ["Built-in sample", "Upload CSV"])

    df_raw: pd.DataFrame

    if data_option == "Upload CSV":
        file = st.file_uploader("CSV (first 3 columns: user, item, rating)", type=["csv"])
        if not file:
            st.info("Upload a CSV to continue.")
            st.stop()
        df_raw = cached_uploaded_csv(file.getvalue())
    else:
        df_raw = load_sample_data()

    df_valid, warn = validate_rating_df(df_raw)
    if df_valid is None:
        st.error(warn or "Invalid data.")
        st.stop()
    if warn:
        st.warning(warn)

    template = pd.DataFrame(columns=["user_id", "item_id", "rating"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV template", data=template, file_name="rating_template.csv")

    st.divider()
    _k, _m = get_openai_credentials()
    if _k and _HAS_AI_MODULE:
        st.success("OpenAI: ready (secrets loaded)")
        st.caption(f"Model: `{_m}`")
    elif _k and not _HAS_AI_MODULE:
        st.warning("OpenAI key found but `openai` package missing — check `requirements.txt`.")
    else:
        st.caption("OpenAI: add `OPENAI_API_KEY` in app secrets to enable AI explanations.")

# Engine (small data — caching the pivot is optional; engine is cheap)
engine = RecommenderEngine(df_valid)
summary = dataset_summary(df_valid)

openai_key, openai_model = get_openai_credentials()

# --- Main tabs ---
tab_overview, tab_data, tab_methods, tab_play, tab_eval = st.tabs(
    ["Overview", "Data", "Methods", "Playground", "Evaluation"]
)

with tab_overview:
    st.title("Recommendation Studio")
    st.markdown(
        "Interactive **explicit feedback** recommender: user–item–rating matrix, "
        "cosine similarity, and transparent diagnostics suitable for a **data scientist / analyst portfolio**."
    )
    st.subheader("What this app does")
    st.markdown(
        """
- **User-based CF** — find users with similar rating patterns; blend their ratings for unseen items.
- **Item-based CF** — find items similar to those you rated highly; rank unseen items by weighted similarity.
- **Item co-occurrence** — binary “also interacted” counts to rank candidates; in **Playground**, the top pick also shows **support, confidence, and lift** for pairs with your rated items (association-rule style metrics on the same binary matrix).

**Limitations:** cold start for new users/items, sparsity, popularity bias, and no temporal split (ratings treated as static).

Optional **AI explanations** (if secrets are set) appear in **Playground** and **Evaluation** after you run the corresponding action — see the sidebar for connection status.
        """
    )

with tab_data:
    st.subheader("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings", f"{summary['n_ratings']:,}")
    c2.metric("Users", summary["n_users"])
    c3.metric("Items", summary["n_items"])
    c4.metric("Sparsity", f"{summary['sparsity']:.1%}")

    rstats = rating_column_stats(df_valid)
    st.subheader("Rating column (observed)")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Min", f"{rstats['min']:.4g}")
    r2.metric("Max", f"{rstats['max']:.4g}")
    r3.metric("Mean", f"{rstats['mean']:.4g}")
    r4.metric("Std", f"{rstats['std']:.4g}")
    st.caption(
        "Algorithms do not assume a 1–5 scale; any numeric ratings work. "
        "RMSE/MAE in Evaluation are on the same scale as these values."
    )

    with st.expander("Optional: validate against an expected range", expanded=False):
        st.markdown(
            "Use this if you know the survey or system scale (e.g. 1–5 stars, 0–100). "
            "The app only **warns**; it does not change your data."
        )
        check_rng = st.checkbox("Flag ratings outside expected min/max", value=False, key="chk_rating_range")
        if check_rng:
            ec1, ec2 = st.columns(2)
            exp_lo = ec1.number_input(
                "Expected minimum",
                value=float(rstats["min"]),
                format="%.6g",
                key="exp_rating_min",
            )
            exp_hi = ec2.number_input(
                "Expected maximum",
                value=float(rstats["max"]),
                format="%.6g",
                key="exp_rating_max",
            )
            if exp_lo > exp_hi:
                st.error("Expected minimum cannot be greater than expected maximum.")
            else:
                bad = df_valid["rating"][(df_valid["rating"] < exp_lo) | (df_valid["rating"] > exp_hi)]
                n_bad = int(bad.shape[0])
                if n_bad:
                    st.warning(
                        f"{n_bad} rating(s) fall outside [{exp_lo}, {exp_hi}]. "
                        "Inspect uploads for typos or mixed scales."
                    )
                else:
                    st.success(f"All ratings lie within [{exp_lo}, {exp_hi}].")

    st.dataframe(df_valid, use_container_width=True, height=320)

with tab_methods:
    st.subheader("Method notes")
    st.markdown(
        """
| Strategy | Idea | Score on an item |
|----------|------|-------------------|
| **User-based CF** | Cosine similarity between **user** vectors (missing → 0). Top 10 neighbors; weighted sum of neighbor rating vectors. | Not calibrated to 1–5; use for **ranking**. |
| **Item-based CF** | Cosine similarity between **item** vectors; multiply by your ratings on rated items. | Same: relative strength. |
| **Item co-occurrence** | Binary matrix; rank by summed co-occurrence. **Playground** adds **support, confidence, lift** for antecedent → top recommended item. | Count rank + rule-style diagnostics. |

**Libraries:** `pandas`, `numpy`, `sklearn.metrics.pairwise.cosine_similarity`.
        """
    )

with tab_play:
    st.subheader("Run recommendations")
    st.caption(f"Active strategy: **{strategy}**")

    col1, col2 = st.columns([1, 2])
    with col1:
        users = df_valid.iloc[:, 0].unique()
        selected_user = st.selectbox("Target user", users)
        num_recs = st.slider("How many recommendations", 1, 10, 5)

    with col2:
        run = st.button("Generate recommendations", type="primary")

    if run:
        with st.spinner("Computing…"):
            note = ""
            if strategy == "User-Based CF":
                recs = engine.get_user_based(selected_user, n=num_recs)
                note = "Users with similar rating vectors contribute more to the score."
                neighbors = engine.get_similar_users(selected_user, k=10)
            elif strategy == "Item-Based CF":
                recs = engine.get_item_based(selected_user, n=num_recs)
                note = "Items similar to your rated items push the ranking."
                neighbors = None
            else:
                recs = engine.get_item_cooccurrence(selected_user, n=num_recs)
                note = "Counts how often candidate items co-occur with your rated items across all users."
                neighbors = None

            evidence: dict[str, Any] = {
                "strategy": strategy,
                "target_user": str(selected_user),
                "method_note": note,
                "top_recommendations": [],
            }

            if not recs.empty:
                st.info(note)
                res_df = pd.DataFrame(
                    {
                        "Recommended item": recs.index.astype(str),
                        "Score": recs.values.round(3),
                    }
                )
                st.dataframe(res_df, use_container_width=True)

                top_item = recs.index[0]
                evidence["top_recommendations"] = [
                    {"item": str(i), "score": float(round(float(v), 4))} for i, v in recs.items()
                ]

                if strategy == "User-Based CF":
                    st.subheader("Similar users (cosine)")
                    st.dataframe(neighbors, use_container_width=True)
                    evidence["similar_users"] = neighbors.to_dict(orient="records")

                if strategy == "Item-Based CF":
                    st.subheader("Top contributions for the top recommendation")
                    br = engine.item_based_contributions(selected_user, top_item, top_k=5)
                    st.dataframe(br, use_container_width=True)
                    evidence["item_contributions_top_pick"] = br.to_dict(orient="records")

                if strategy == "Item co-occurrence (binary)":
                    st.subheader("Co-occurrence with your items (top pick)")
                    cb = engine.cooccurrence_breakdown(selected_user, top_item, top_k=5)
                    st.dataframe(cb, use_container_width=True)
                    evidence["cooccurrence_top_pick"] = cb.to_dict(orient="records")

                    st.subheader("Association metrics (antecedent → top pick)")
                    st.caption(
                        "From the same binary matrix: **support** = P(A∩C), "
                        "**confidence** = P(C|A), **lift** = confidence / P(C)."
                    )
                    am = engine.association_metrics_breakdown(selected_user, top_item, top_k=8)
                    if not am.empty:
                        disp = am.copy()
                        disp["support"] = disp["support"].round(4)
                        disp["confidence_if_antecedent"] = disp["confidence_if_antecedent"].round(4)
                        disp["lift"] = disp["lift"].round(4)
                        st.dataframe(disp, use_container_width=True)
                        evidence["association_metrics_top_pick"] = am.to_dict(orient="records")
                    else:
                        st.caption("Not enough rated items to form association rows for this pick.")

                st.session_state.last_evidence = evidence
                st.balloons()
            else:
                st.warning("Not enough signal for this user with the current data.")
                st.session_state.last_evidence = None

    if st.session_state.last_evidence and openai_key and _HAS_AI_MODULE:
        st.divider()
        st.subheader("Optional: AI interpretation")
        st.caption("One API call per click. Uses only the structured evidence from your last successful run.")
        if st.button("Explain this run (AI)"):
            try:
                txt = explain_run_json(
                    st.session_state.last_evidence,
                    api_key=openai_key,
                    model=openai_model,
                    topic="recommendations",
                )
                st.markdown(txt)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
    elif st.session_state.last_evidence and not openai_key:
        st.caption("OpenAI key not detected. Add `OPENAI_API_KEY` in Streamlit **Settings → Secrets** (or `.streamlit/secrets.toml` locally).")
    elif openai_key and _HAS_AI_MODULE and not st.session_state.last_evidence:
        st.info("Run **Generate recommendations** once; then **Explain this run (AI)** will use your secrets.")

with tab_eval:
    st.subheader("Quick hold-out check (user-based CF)")
    st.markdown(
        "Randomly hides a slice of ratings, refits the user–item matrix **without** each held row, "
        "and compares the **user-based score** on the held item to the actual rating. "
        "Use as a **rough** sanity metric — not a full offline evaluation pipeline."
    )
    if st.button("Run hold-out RMSE / MAE", key="eval_btn"):
        with st.spinner("Evaluating (may take a few seconds)…"):
            rmse, mae, n_pred, err = RecommenderEngine.evaluate_user_based_holdout(
                df_valid, random_state=42, max_tests=100, test_fraction=0.2
            )
        if err:
            st.warning(err)
            st.session_state.last_eval_evidence = None
        else:
            e1, e2, e3 = st.columns(3)
            e1.metric("RMSE", f"{rmse:.3f}")
            e2.metric("MAE", f"{mae:.3f}")
            e3.metric("Hold-out predictions", n_pred)
            st.caption("Lower is better. Interpret with care on tiny samples.")

            rs = rating_column_stats(df_valid)
            st.session_state.last_eval_evidence = {
                "topic": "evaluation",
                "method": "User-based collaborative filtering, random hold-out rows",
                "rmse": rmse,
                "mae": mae,
                "n_holdout_predictions": n_pred,
                "observed_rating_min": rs["min"],
                "observed_rating_max": rs["max"],
                "observed_rating_mean": rs["mean"],
                "caveats": [
                    "Not a temporal split; ratings treated as exchangeable.",
                    "User-based scores are not calibrated to the rating scale; errors are in raw rating units.",
                    "Small samples make RMSE/MAE noisy.",
                ],
            }

    if st.session_state.last_eval_evidence and openai_key and _HAS_AI_MODULE:
        st.divider()
        st.subheader("Optional: AI interpretation of metrics")
        st.caption("One API call per click. Uses only the numbers and notes from your last evaluation run.")
        if st.button("Explain this evaluation (AI)", key="eval_ai_btn"):
            try:
                txt = explain_run_json(
                    st.session_state.last_eval_evidence,
                    api_key=openai_key,
                    model=openai_model,
                    topic="evaluation",
                )
                st.markdown(txt)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
    elif st.session_state.last_eval_evidence and not openai_key:
        st.caption("Add `OPENAI_API_KEY` in Streamlit secrets to enable evaluation explanations.")
    elif openai_key and _HAS_AI_MODULE and not st.session_state.last_eval_evidence:
        st.info("Run **hold-out RMSE / MAE** first; then you can use **Explain this evaluation (AI)**.")

st.divider()
st.caption("Portfolio demo — collaborative filtering and co-occurrence | Abhishek Jha")
