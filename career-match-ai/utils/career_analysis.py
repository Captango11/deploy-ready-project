"""Career intelligence: skill coverage, Career Fit Score, What-If simulation
and career adjacency. All values are computed, never hardcoded."""
from .role_profiles import ROLE_SKILLS

# Documented Career Fit formula weights
FIT_WEIGHTS = {"ml_probability": 0.40, "skill_coverage": 0.40, "evidence": 0.20}


def skill_coverage(skills, role):
    profile = ROLE_SKILLS[role]
    s = set(skills)
    matched = [k for k in profile if k in s]
    missing = [k for k in profile if k not in s]
    return {"coverage": len(matched) / len(profile), "matched": matched, "missing": missing}


def all_coverages(skills):
    return {r: skill_coverage(skills, r)["coverage"] for r in ROLE_SKILLS}


def evidence_score(evidence, n_sentences, matched_skills, role):
    """Share of the role's matched skills that are backed by at least one
    evidence sentence, blended with how many resume sentences support the role."""
    if not matched_skills:
        return 0.0
    backed = set()
    for e in evidence:
        backed.update(m for m in e["matches"] if m in matched_skills)
    skill_part = len(backed) / len(matched_skills)
    sent_part = min(1.0, len(evidence) / max(3, min(6, n_sentences)))
    return 0.6 * skill_part + 0.4 * sent_part


def career_fit(ml_prob, coverage, evidence):
    w = FIT_WEIGHTS
    return w["ml_probability"] * ml_prob + w["skill_coverage"] * coverage + w["evidence"] * evidence


def what_if(skills, added):
    before = all_coverages(skills)
    after = all_coverages(list(set(skills) | set(added)))
    return before, after


def adjacency(skills, current_role, k=3):
    """Rank other roles by Jaccard overlap between role profiles plus user coverage."""
    cur = set(ROLE_SKILLS[current_role])
    user = set(skills)
    rows = []
    for role, prof in ROLE_SKILLS.items():
        if role == current_role:
            continue
        p = set(prof)
        profile_sim = len(cur & p) / len(cur | p)
        cov = len(user & p) / len(p)
        rows.append({
            "role": role,
            "score": 0.5 * profile_sim + 0.5 * cov,
            "profile_similarity": profile_sim,
            "coverage": cov,
            "overlap": [s for s in prof if s in user],
            "missing": [s for s in prof if s not in user],
        })
    rows.sort(key=lambda r: -r["score"])
    return rows[:k]
