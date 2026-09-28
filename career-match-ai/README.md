# CareerMatch AI
### NLP-Powered Resume Intelligence & Career Matching System

> An NLP + supervised machine-learning system for resume text classification and skill-based career analysis.

```text
NLP:  Unstructured resume text → numerical TF-IDF features
ML:   TF-IDF features → learned role classifier → role probabilities
```

## Pipeline
Resume PDF/Text → Text Extraction (PyPDF2) → NLP Preprocessing → Tokenization/Normalization →
TF-IDF (1–2 grams) → Logistic Regression → Probability Calibration → Top 5 Roles →
Career Fit Score · Explainable Prediction · Skill Gap · What-If Simulator · Career Adjacency

### NLP Concepts
- Text preprocessing (URL/email/symbol removal, whitespace normalization)
- Technical-term preservation (C++, C#, Node.js, .NET, REST API, GitHub Actions…)
- Tokenization and stop-word removal
- N-grams (unigrams + bigrams)
- TF-IDF (`ngram_range=(1,2), sublinear_tf=True, max_features=10000, min_df=1`)
- Dictionary-based skill extraction
- NLP-based evidence analysis (verbatim resume sentences)

### ML Concepts
- Supervised multiclass classification (10 roles)
- Logistic Regression (`max_iter=2000, class_weight="balanced"`)
- Train/test split (80/20, stratified, `random_state=42`)
- Probability calibration (`CalibratedClassifierCV`, sigmoid, cv=3)
- Prediction probabilities via `predict_proba()`
- Accuracy, precision, recall, F1 (macro + weighted), confusion matrix — see `model/metrics.json`

### Career Fit Score
`0.40 × ML probability + 0.40 × skill coverage + 0.20 × evidence coverage`.
It is a project-defined analytical score, not an industry-standard employability score.

## Dataset
`data/resumes.csv` — 10,000 resumes, 1,000 per role (columns: `resume_id, resume_text, job_role`).
Note: the dataset's resumes are very consistent per role, so held-out accuracy is near-perfect; real resumes are noisier.

## Run locally
```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py
python evaluate_model.py
streamlit run app.py
```

## Deploy on Render
1. Push the repository to GitHub.
2. In Render: **New → Blueprint** and select the repo (uses `render.yaml`), **or** **New → Web Service** with:
   - Root Directory: `career-match-ai` (leave empty if this folder is the repo root)
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Env var: `PYTHON_VERSION=3.11.9`
3. No API keys needed. Trained models in `model/` are committed; if they are missing or incompatible, the app retrains automatically on startup.

## Adding a role
Add rows to `data/resumes.csv`, add the role to `utils/role_profiles.py`, then re-run `train_model.py` and `evaluate_model.py`.

## Structure
```text
app.py  train_model.py  evaluate_model.py  requirements.txt  render.yaml
data/resumes.csv
model/classifier.pkl  vectorizer.pkl  metadata.pkl  metrics.json
utils/preprocessing.py  skills.py  role_profiles.py  explainability.py  career_analysis.py
```
