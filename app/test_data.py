import pandas as pd

df = pd.read_csv("data/scholarships.csv")
print(df.shape)
print(df[["name", "min_marks", "max_income"]])
