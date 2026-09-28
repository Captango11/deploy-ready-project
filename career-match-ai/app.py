"""CareerMatch AI — NLP-Powered Resume Intelligence & Career Matching (Streamlit)."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.career_analysis import (FIT_WEIGHTS, adjacency, career_fit, evidence_score,
                                   skill_coverage, what_if)
from utils.explainability import find_evidence, top_signals
from utils.preprocessing import (PDFExtractionError, extract_text_from_pdf, normalize,
                                 preprocess, tokenize)
from utils.role_profiles import ROLE_SKILLS
from utils.skills import ALL_SKILLS, extract_skills

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MIN_WORDS = 20

st.set_page_config(page_title="CareerMatch AI", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
.stApp { background: radial-gradient(circle at 10% 0%, #13233a 0%, #0b1220 45%, #080d17 100%); color: #e6edf7; }
section[data-testid="stSidebar"] { background: #0a111d; border-right: 1px solid #1c2a40; }
.hero { padding: 28px 32px; border-radius: 22px; background: linear-gradient(135deg, #0f2a3d, #10233f 60%, #1a1f3d);
        border: 1px solid #22405e; margin-bottom: 22px; }
.hero h1 { margin: 0; font-size: 2.6rem; color: #f2f7ff; }
.hero h1 span { color: #34d8b0; }
.hero p { margin: 6px 0 0; color: #9fb4cf; font-size: 1.05rem; }
.card { background: #0f1a2b; border: 1px solid #1f3150; border-radius: 18px; padding: 20px 22px; margin-bottom: 16px; }
.card h4 { margin-top: 0; color: #cfe0f5; }
.metric { background: #0f1a2b; border: 1px solid #1f3150; border-radius: 16px; padding: 16px 18px; text-align: left; }
.metric .label { color: #8aa2c1; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric .value { font-family: 'Space Grotesk'; font-size: 2rem; font-weight: 700; color: #f2f7ff; }
.metric.accent { border-color: #34d8b0; background: linear-gradient(135deg, #0f2a2a, #0f1a2b); }
.metric.accent .value { color: #34d8b0; }
.top-role { font-family: 'Space Grotesk'; font-size: 2.4rem; font-weight: 700; color: #34d8b0; line-height: 1.1; }
.chip { display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 999px; font-size: .85rem;
        background: #16304d; color: #bfe0ff; border: 1px solid #25507c; }
.chip.ok { background: #0f3a30; color: #8ff0cf; border-color: #1f7a5f; }
.chip.miss { background: #2e2413; color: #f7c978; border-color: #6e5423; }
.chip.add { background: #2b1a3e; color: #d8b8ff; border-color: #5c3a8a; }
.bar-row { margin: 8px 0 12px; }
.bar-label { display: flex; justify-content: space-between; font-size: .92rem; color: #d6e3f3; margin-bottom: 4px; }
.bar { height: 10px; background: #1a2840; border-radius: 999px; overflow: hidden; }
.bar > div { height: 100%; background: linear-gradient(90deg, #2b8cff, #34d8b0); border-radius: 999px; }
.quote { border-left: 3px solid #34d8b0; padding: 8px 14px; margin: 8px 0; background: #0c1626; border-radius: 0 10px 10px 0; color: #d6e3f3; }
.note { color: #8aa2c1; font-size: .85rem; font-style: italic; }
.flow { font-family: 'Space Grotesk'; color: #bfe0ff; }
.stButton > button { border-radius: 12px; background: linear-gradient(90deg, #2b8cff, #34d8b0); color: #06121f;
        font-weight: 700; border: 0; padding: .6rem 1.4rem; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk'; }
</style>
""", unsafe_allow_html=True)


def metric(label, value, accent=False):
    return f'<div class="metric {"accent" if accent else ""}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def bar(label, value, right=None):
    pct = max(0, min(100, value * 100))
    return (f'<div class="bar-row"><div class="bar-label"><span>{label}</span><span>{right or f"{pct:.1f}%"}</span></div>'
            f'<div class="bar"><div style="width:{pct:.1f}%"></div></div></div>')


def chips(items, kind=""):
    if not items:
        return '<span class="note">None</span>'
    prefix = {"ok": "✓ ", "miss": "○ ", "add": "+ "}.get(kind, "")
    return "".join(f'<span class="chip {kind}">{prefix}{i}</span>' for i in items)


def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading NLP + ML model…")
def load_artifacts():
    import joblib
    paths = {k: os.path.join(MODEL_DIR, f"{k}.pkl") for k in ("classifier", "vectorizer", "metadata")}
    try:
        return {k: joblib.load(p) for k, p in paths.items()}, None
    except Exception:
        # Missing / incompatible model files -> retrain from the bundled dataset.
        try:
            from train_model import train
            from evaluate_model import evaluate
            clf, vec, meta = train()
            evaluate()
            return {"classifier": clf, "vectorizer": vec, "metadata": meta}, "retrained"
        except Exception as e:
            return None, f"Model unavailable and could not be trained: {e}"


def load_metrics():
    p = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- analysis
def analyze(raw_text, art):
    clf, vec, meta = art["classifier"], art["vectorizer"], art["metadata"]
    normalized = normalize(raw_text)
    tokens = tokenize(normalized)
    cleaned = " ".join(tokens)
    X = vec.transform([cleaned])
    probs = clf.predict_proba(X)[0]  # real calibrated probabilities
    classes = list(clf.classes_)
    order = np.argsort(-probs)
    top5 = [(classes[i], float(probs[i])) for i in order[:5]]
    top_role, top_prob = top5[0]

    skills = extract_skills(raw_text)
    cov = skill_coverage(skills, top_role)
    signals = top_signals(X[0], top_role, meta)
    evidence, n_sent = find_evidence(raw_text, [s["term"] for s in signals], cov["matched"])
    ev = evidence_score(evidence, n_sent, cov["matched"], top_role)
    fit = career_fit(top_prob, cov["coverage"], ev)

    nz = X[0].tocsr()
    fn = meta["feature_names"]
    tfidf_top = sorted(((fn[i], float(v)) for i, v in zip(nz.indices, nz.data)), key=lambda t: -t[1])[:15]
    return {
        "raw": raw_text, "normalized": normalized, "tokens": tokens, "top5": top5,
        "all_probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "top_role": top_role, "top_prob": top_prob, "skills": skills, "coverage": cov,
        "signals": signals, "evidence": evidence, "evidence_score": ev, "fit": fit,
        "n_features_active": int(nz.nnz), "tfidf_top": tfidf_top,
        "bigrams": [t for t, _ in tfidf_top if " " in t][:8],
    }


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🎯 CareerMatch AI")
    st.caption("NLP-Powered Resume Intelligence")
    page = st.radio("Navigate", [
        "🏠 Resume Analysis", "🧠 Skill Intelligence", "🔮 What-If Simulator",
        "🛣 Career Adjacency", "📊 Model Performance", "🧪 NLP + ML Pipeline", "ℹ️ About",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Runs 100% locally — no external AI APIs.")

st.markdown('<div class="hero"><h1>CareerMatch <span>AI</span></h1>'
            '<p>NLP-Powered Resume Intelligence &amp; Career Matching</p></div>', unsafe_allow_html=True)

art, load_status = load_artifacts()
if art is None:
    st.error(load_status)
    st.stop()
if load_status == "retrained":
    st.info("Model files were missing or incompatible, so the model was retrained from data/resumes.csv.")

res = st.session_state.get("result")


def need_result():
    if not res:
        st.info("Analyze a resume on **🏠 Resume Analysis** first.")
        st.stop()


# ================================================================ pages
if page == "🏠 Resume Analysis":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📄 Upload Resume PDF")
        up = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    with c2:
        st.markdown("#### OR  Paste Resume Text")
        pasted = st.text_area("Paste", height=150, label_visibility="collapsed",
                              placeholder="Paste your resume text here…")
    if st.button("✨ Analyze Resume"):
        text = None
        try:
            if up is not None:
                text = extract_text_from_pdf(up.getvalue())
            elif pasted and pasted.strip():
                text = pasted
            else:
                st.warning("Please upload a PDF or paste resume text.")
        except PDFExtractionError as e:
            st.error(str(e))
        except Exception:
            st.error("Something went wrong while reading the file. Please try another file.")
        if text is not None:
            if len(tokenize(normalize(text))) < MIN_WORDS:
                st.warning(f"The resume is too short to analyze reliably (need at least {MIN_WORDS} meaningful words).")
            else:
                try:
                    st.session_state["result"] = analyze(text, art)
                    st.session_state.pop("whatif", None)
                    res = st.session_state["result"]
                except Exception:
                    st.error("Analysis failed for this input. Please check the resume content and try again.")

    if res:
        st.markdown("---")
        a, b = st.columns([1, 2])
        with a:
            card(f'<div class="metric label" style="border:0;padding:0">TOP MODEL MATCH</div>'
                 f'<div class="top-role">{res["top_role"]}</div>'
                 f'<div style="font-size:1.6rem;font-weight:700;color:#f2f7ff">{res["top_prob"]*100:.1f}%</div>'
                 f'<div class="note">Calibrated probability from model.predict_proba()</div>')
        with b:
            card("<h4>Model-Based Role Matches (Top 5)</h4>" +
                 "".join(bar(f"{i+1}. {r}", p) for i, (r, p) in enumerate(res["top5"])))

        st.markdown("### Career Fit Analysis")
        m = st.columns(4)
        m[0].markdown(metric("ML Probability", f'{res["top_prob"]*100:.0f}%'), unsafe_allow_html=True)
        m[1].markdown(metric("Skill Coverage", f'{res["coverage"]["coverage"]*100:.0f}%'), unsafe_allow_html=True)
        m[2].markdown(metric("Resume Evidence", f'{res["evidence_score"]*100:.0f}%'), unsafe_allow_html=True)
        m[3].markdown(metric("Overall Fit Score", f'{res["fit"]*100:.0f}%', True), unsafe_allow_html=True)
        st.markdown(f'<p class="note">Formula: {int(FIT_WEIGHTS["ml_probability"]*100)}% ML probability + '
                    f'{int(FIT_WEIGHTS["skill_coverage"]*100)}% skill coverage + {int(FIT_WEIGHTS["evidence"]*100)}% evidence coverage. '
                    'Career Fit Score is a project-defined analytical score and is not an industry-standard employability score. '
                    'It does not guarantee employment.</p>', unsafe_allow_html=True)

        card("<h4>Detected Skills</h4>" + chips(res["skills"]))

        t1, t2 = st.tabs(["💡 Why this match?", "🧩 Skill Gap"])
        with t1:
            card(f"<h4>Why {res['top_role']}?</h4><p class='note'>Resume terms with the highest TF-IDF × "
                 "Logistic Regression coefficient contribution toward this role.</p>" +
                 chips([s["term"] for s in res["signals"]]))
            if res["signals"]:
                df = pd.DataFrame(res["signals"]).iloc[::-1]
                fig = go.Figure(go.Bar(x=df["weight"], y=df["term"], orientation="h", marker_color="#34d8b0"))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#d6e3f3", xaxis_title="contribution")
                st.plotly_chart(fig, use_container_width=True)
            ev_html = "".join(f'<div class="quote">"{e["sentence"]}"<br>{chips(e["matches"])}</div>'
                              for e in res["evidence"]) or '<span class="note">No supporting sentences found.</span>'
            card("<h4>Resume Evidence</h4><p class='note'>Quoted verbatim from your resume — nothing is generated.</p>" + ev_html)
        with t2:
            cov = res["coverage"]
            card(f"<h4>Top Role: {res['top_role']}</h4>" + bar("Skill Coverage", cov["coverage"]) +
                 "<b>MATCHED SKILLS</b><br>" + chips(cov["matched"], "ok") +
                 "<br><br><b>SKILLS TO DEVELOP</b><br>" + chips(cov["missing"], "miss") +
                 "<p class='note'>Skills commonly associated with this project's role profile — not universally required.</p>")

elif page == "🧠 Skill Intelligence":
    need_result()
    card("<h4>Detected Skills</h4>" + chips(res["skills"]))
    rows = [{"Role": r, "Skill Coverage": skill_coverage(res["skills"], r)["coverage"],
             "ML Probability": res["all_probs"].get(r, 0.0)} for r in ROLE_SKILLS]
    df = pd.DataFrame(rows).sort_values("Skill Coverage")
    fig = go.Figure()
    fig.add_bar(y=df["Role"], x=df["Skill Coverage"] * 100, name="Skill Coverage %", orientation="h", marker_color="#34d8b0")
    fig.add_bar(y=df["Role"], x=df["ML Probability"] * 100, name="ML Probability %", orientation="h", marker_color="#2b8cff")
    fig.update_layout(barmode="group", height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#d6e3f3", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="note">ML Probability comes from the classifier; Skill Coverage comes from the role-skill dictionary. They are separate signals.</p>', unsafe_allow_html=True)
    for r in ROLE_SKILLS:
        c = skill_coverage(res["skills"], r)
        with st.expander(f"{r} — {c['coverage']*100:.0f}% coverage"):
            st.markdown("Matched: " + chips(c["matched"], "ok") + "<br>To develop: " + chips(c["missing"], "miss"), unsafe_allow_html=True)

elif page == "🔮 What-If Simulator":
    need_result()
    st.markdown("### What If I Learn This Skill?")
    card("<h4>Current detected skills</h4>" + chips(res["skills"]))
    options = [s for s in ALL_SKILLS if s not in res["skills"]]
    added = st.multiselect("Select skills you plan to learn", options, key="whatif")
    before, after = what_if(res["skills"], added)
    rows = sorted(ROLE_SKILLS, key=lambda r: -after[r])
    c1, c2 = st.columns(2)
    with c1:
        card("<h4>BEFORE</h4>" + "".join(bar(r, before[r]) for r in rows))
    with c2:
        card("<h4>AFTER ADDING SKILLS</h4>" + "".join(
            bar(r, after[r], f"{after[r]*100:.1f}%" + (f"  (+{(after[r]-before[r])*100:.0f})" if after[r] > before[r] else ""))
            for r in rows))
    card(f"<h4>Original ML Probability (unchanged)</h4>" + bar(res["top_role"], res["top_prob"]) +
         "<p class='note'>The classifier is not re-run; only Simulated Skill Coverage changes.</p>")
    st.markdown('<p class="note">The What-If Simulator models how adding skills changes the project\'s role-skill coverage. '
                'It does not guarantee that learning a skill will change real-world hiring outcomes.</p>', unsafe_allow_html=True)

elif page == "🛣 Career Adjacency":
    need_result()
    st.markdown("### Potential Career Transitions")
    card(f"<h4>Current strongest role</h4><div class='top-role'>{res['top_role']}</div>")
    for r in adjacency(res["skills"], res["top_role"]):
        card(f"<h4>{res['top_role']} → {r['role']}</h4><span class='note'>Potential skill-based transition</span>" +
             bar("Current skill coverage for this role", r["coverage"]) +
             bar("Role profile similarity", r["profile_similarity"]) +
             "<b>Current overlap</b><br>" + chips(r["overlap"], "ok") +
             "<br><br><b>Additional skills needed</b><br>" + chips(r["missing"], "add"))
    st.markdown('<p class="note">Based on skill overlap only. This does not imply guaranteed career progression.</p>', unsafe_allow_html=True)

elif page == "📊 Model Performance":
    metrics = load_metrics()
    if not metrics:
        st.warning("metrics.json not found. Run `python evaluate_model.py`.")
        st.stop()
    st.markdown(f"**Evaluated on held-out test data** — {metrics['evaluated_on']}, n = {metrics['n_test']}.")
    m = st.columns(4)
    m[0].markdown(metric("Accuracy", f"{metrics['accuracy']*100:.2f}%", True), unsafe_allow_html=True)
    m[1].markdown(metric("Precision (macro)", f"{metrics['precision_macro']*100:.2f}%"), unsafe_allow_html=True)
    m[2].markdown(metric("Recall (macro)", f"{metrics['recall_macro']*100:.2f}%"), unsafe_allow_html=True)
    m[3].markdown(metric("F1 (macro)", f"{metrics['f1_macro']*100:.2f}%"), unsafe_allow_html=True)
    st.caption(f"Weighted F1: {metrics['f1_weighted']*100:.2f}%")
    labels, cm = metrics["labels"], np.array(metrics["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Teal", text=cm, texttemplate="%{text}"))
    fig.update_layout(title="Confusion Matrix", height=560, xaxis_title="Predicted", yaxis_title="Actual",
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#d6e3f3", yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3), use_container_width=True)
    if metrics["accuracy"] > 0.99:
        st.markdown('<p class="note">Near-perfect scores indicate the dataset\'s resumes are highly consistent per role '
                    '(e.g. many mention the role title). Real-world resumes are noisier, so expect lower accuracy on them.</p>',
                    unsafe_allow_html=True)

elif page == "🧪 NLP + ML Pipeline":
    st.markdown("### How NLP + ML Works")
    c1, c2 = st.columns(2)
    with c1:
        card("<h4>NLP Layer</h4><div class='flow'>Resume<br>↓ Text Extraction (PyPDF2)<br>↓ Cleaning (URLs, emails, symbols)"
             "<br>↓ Normalization (lowercase, C++/C#/Node.js/.NET preserved)<br>↓ Tokenization<br>↓ N-grams (1–2)<br>↓ TF-IDF</div>"
             "<p class='note'>NLP converts unstructured human language into useful numerical/text features.</p>")
    with c2:
        card("<h4>ML Layer</h4><div class='flow'>TF-IDF Features<br>↓ Logistic Regression<br>↓ Multiclass Prediction"
             "<br>↓ Probability Calibration (CalibratedClassifierCV)<br>↓ Top 5 Role Matches</div>"
             "<p class='note'>Machine Learning learns patterns from those features to classify the resume into job-role categories.</p>")
    st.info("TF-IDF converts important words and phrases from the resume into numerical features that can be understood by the ML classifier.")
    meta = art["metadata"]
    card(f"<h4>Trained model</h4>Model: {meta.get('model')}<br>Vocabulary size: {meta.get('n_features')}<br>"
         f"Train rows: {meta.get('n_train')} · Test rows: {meta.get('n_test')}<br>TF-IDF: {meta.get('vectorizer_params')}")
    if res:
        st.markdown("#### Your resume through the pipeline")
        with st.expander("1. Extracted raw text"):
            st.text(res["raw"][:3000])
        with st.expander("2. Normalized text"):
            st.text(res["normalized"][:3000])
        with st.expander(f"3. Tokens ({len(res['tokens'])})"):
            st.write(res["tokens"][:200])
        with st.expander(f"4. TF-IDF vector ({res['n_features_active']} non-zero features)"):
            st.dataframe(pd.DataFrame(res["tfidf_top"], columns=["term / n-gram", "tf-idf"]), use_container_width=True)
            st.write("Bigrams detected:", res["bigrams"])
        with st.expander("5. Calibrated probabilities (all classes)"):
            st.dataframe(pd.DataFrame(sorted(res["all_probs"].items(), key=lambda t: -t[1]),
                                      columns=["role", "probability"]), use_container_width=True)

else:
    st.markdown("### About CareerMatch AI")
    card("<p>CareerMatch AI is an NLP-powered supervised machine-learning system that transforms unstructured resume text "
         "into TF-IDF features, classifies the resume into relevant job-role categories using Logistic Regression, and "
         "provides explainable skill analysis, career-fit analysis, skill-gap identification, what-if skill simulation, "
         "and potential skill-based career transitions.</p>"
         "<p class='note'>All probabilities, metrics, skills and evidence are computed from your input and the trained model. "
         "No external AI APIs are used.</p>")
    card("<h4>Roles</h4>" + chips(list(ROLE_SKILLS.keys())))
