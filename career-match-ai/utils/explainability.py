"""NLP explainability: TF-IDF x Logistic Regression coefficient contributions
and evidence sentences that literally exist in the resume."""
import numpy as np
from .preprocessing import preprocess, split_sentences, normalize


def top_signals(x_vec, role, metadata, k=12):
    """Return resume terms with the largest positive contribution
    (tfidf_value * coefficient) toward `role`."""
    classes = list(metadata["classes"])
    coef = metadata["explain_coef"][classes.index(role)]
    names = metadata["feature_names"]
    row = x_vec.tocsr()
    idx, vals = row.indices, row.data
    contrib = vals * coef[idx]
    order = np.argsort(-contrib)
    out = []
    for o in order:
        if contrib[o] <= 0 or len(out) >= k:
            break
        term = names[idx[o]].replace("_", " ")
        term = term.replace("cplusplus", "c++").replace("csharp", "c#").replace("dotnet", ".net")
        out.append({"term": term, "weight": float(contrib[o])})
    return out


def find_evidence(raw_text, terms, skills, max_items=6):
    """Pick sentences from the ORIGINAL resume containing the model's top terms
    or detected role skills. Nothing is generated — only quoted."""
    from .skills import skill_in_text
    sentences = split_sentences(raw_text)
    scored = []
    for s in sentences:
        norm = " " + preprocess(s) + " "
        hit_terms = [t for t in terms if " " + preprocess(t) + " " in norm]
        hit_skills = [sk for sk in skills if skill_in_text(sk, s)]
        score = len(set(hit_terms)) + len(set(hit_skills))
        if score > 0:
            scored.append((score, s, sorted(set(hit_skills + hit_terms))))
    scored.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for score, s, hits in scored:
        if s not in seen:
            seen.add(s)
            out.append({"sentence": s[:400], "matches": hits[:8]})
        if len(out) >= max_items:
            break
    return out, len(sentences)
