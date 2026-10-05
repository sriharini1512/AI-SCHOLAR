import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

students = pd.DataFrame({
    "student_id": range(1, n + 1),
    "marks": rng.normal(70, 12, n).clip(35, 99).round(1),
    "income": rng.lognormal(12.6, 0.6, n).clip(30000, 2000000).round(-3).astype(int),
    "category": rng.choice(["OC", "BC", "MBC", "SC", "ST"], n, p=[0.2, 0.35, 0.2, 0.18, 0.07]),
    "gender": rng.choice(["Male", "Female"], n),
    "course": rng.choice(["UG", "PG", "Diploma"], n, p=[0.7, 0.2, 0.1]),
    "state": rng.choice(["Tamil Nadu", "Kerala", "Karnataka"], n, p=[0.7, 0.15, 0.15]),
    "minority": rng.choice([0, 1], n, p=[0.85, 0.15]),
})

students.to_csv("data/students.csv", index=False)
print(students.shape)
print(students.head())