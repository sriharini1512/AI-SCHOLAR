import numpy as np
import pandas as pd
from eligibility import is_eligible

rng = np.random.default_rng(7)
schemes = pd.read_csv("data/scholarships.csv")
students = pd.read_csv("data/students.csv")

rows = []
for _, st in students.iterrows():
    for _, sc in schemes.iterrows():
        if not is_eligible(st, sc):
            continue
        margin = st["marks"] - sc["min_marks"]
        income_ratio = st["income"] / sc["max_income"]
        cat_specific = int("All" not in sc["category"].split(";"))
        gender_specific = int(sc["gender"] != "All")

        # synthetic selection probability (sample rule used to create labels)
        z = (-1.0 + 0.08 * margin + 1.5 * (1 - income_ratio)
             + 0.6 * cat_specific - sc["amount"] / 50000 * 0.5
             + rng.normal(0, 0.5))
        p = 1 / (1 + np.exp(-z))

        rows.append({
            "student_id": st["student_id"],
            "scheme_id": sc["scheme_id"],
            "marks": st["marks"],
            "income": st["income"],
            "category": st["category"],
            "course": st["course"],
            "marks_margin": round(margin, 1),
            "income_ratio": round(income_ratio, 3),
            "cat_specific": cat_specific,
            "gender_specific": gender_specific,
            "amount": sc["amount"],
            "selected": int(rng.random() < p),
        })

pairs = pd.DataFrame(rows)
pairs.to_csv("data/pairs.csv", index=False)

print("pairs:", pairs.shape)
print("selected rate:", round(pairs["selected"].mean(), 3))
print("avg eligible schemes per student:", round(len(pairs) / len(students), 2))