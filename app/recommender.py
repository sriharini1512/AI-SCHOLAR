import os
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from eligibility import is_eligible

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
schemes = pd.read_csv(os.path.join(BASE, "data", "scholarships.csv"))
model = joblib.load(os.path.join(BASE, "app", "model.joblib"))

W_RULE, W_ML, W_NLP = 0.2, 0.5, 0.3
vec = TfidfVectorizer(stop_words="english").fit(schemes["description"])
DESC = vec.transform(schemes["description"])


def recommend(st, interests="", n=5):
    q = vec.transform([f"{interests} {st['state']}"])
    nlp_all = cosine_similarity(q, DESC)[0]
    top = nlp_all.max()
    if top > 0:
        nlp_all = nlp_all / top

    today = datetime.date.today()
    rows, meta = [], []
    for i, sc in schemes.iterrows():
        if not is_eligible(st, sc):
            continue
        days = (pd.to_datetime(sc["deadline"]).date() - today).days
        if days < 0:
            continue
        margin = st["marks"] - sc["min_marks"]
        ratio = st["income"] / sc["max_income"]
        cat_sp = int("All" not in sc["category"].split(";"))
        gen_sp = int(sc["gender"] != "All")
        rows.append({"marks": st["marks"], "income": st["income"],
                     "marks_margin": margin, "income_ratio": ratio,
                     "cat_specific": cat_sp, "gender_specific": gen_sp,
                     "amount": sc["amount"], "category": st["category"],
                     "course": st["course"]})
        meta.append((i, sc, margin, ratio, cat_sp, gen_sp, days))

    if not rows:
        return []

    ml = model.predict_proba(pd.DataFrame(rows))[:, 1]
    out = []
    for k, (i, sc, margin, ratio, cat_sp, gen_sp, days) in enumerate(meta):
        rule = 0.5 * float(np.clip(margin / 40, 0, 1)) + 0.5 * float(np.clip(1 - ratio, 0, 1))
        score = W_RULE * rule + W_ML * float(ml[k]) + W_NLP * float(nlp_all[i])
        reasons = [f"Your marks are {margin:.0f}% above the minimum cutoff",
                   f"Family income is within the limit (Rs {int(sc['max_income']):,})"]
        if cat_sp:
            reasons.append("Your category matches this scheme")
        if gen_sp:
            reasons.append(f"Open for {sc['gender'].lower()} students")
        if sc["state"] != "All":
            reasons.append(f"Valid for students in {sc['state']}")
        if nlp_all[i] > 0.15:
            reasons.append("Matches your interests")
        out.append({"name": sc["name"], "provider": sc["provider"],
                    "amount": int(sc["amount"]), "deadline": str(sc["deadline"]),
                    "days_left": int(days), "description": sc["description"],
                    "score": round(score * 100), "ml": round(float(ml[k]) * 100),
                    "reasons": reasons})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:n]