import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/pairs.csv")
num = ["marks", "income", "marks_margin", "income_ratio",
       "cat_specific", "gender_specific", "amount"]
cat = ["category", "course"]

X = df[num + cat]
y = df["selected"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
}

best_name, best_f1, best_pipe = None, -1, None
print(f"{'Model':20s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    pr = precision_score(y_test, pred)
    rc = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name:20s} {acc:6.3f} {pr:6.3f} {rc:6.3f} {f1:6.3f}")
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe

joblib.dump(best_pipe, "app/model.joblib")
print("\nBest model:", best_name, "-> saved to app/model.joblib")