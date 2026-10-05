import pandas as pd


def is_eligible(student, scheme):
    if student["marks"] < scheme["min_marks"]:
        return False
    if student["income"] > scheme["max_income"]:
        return False

    cats = scheme["category"].split(";")
    if "All" not in cats:
        category_ok = student["category"] in cats or (
            "Minority" in cats and student["minority"] == 1
        )
        if not category_ok:
            return False

    if scheme["gender"] != "All" and student["gender"] != scheme["gender"]:
        return False

    courses = scheme["course"].split(";")
    if "All" not in courses and student["course"] not in courses:
        return False

    if scheme["state"] != "All" and student["state"] != scheme["state"]:
        return False

    return True


def get_eligible_schemes(student, schemes_df):
    mask = schemes_df.apply(lambda s: is_eligible(student, s), axis=1)
    return schemes_df[mask]


if __name__ == "__main__":
    schemes = pd.read_csv("data/scholarships.csv")
    students = pd.read_csv("data/students.csv")

    student = students.iloc[0]
    print(student)
    print()
    print(get_eligible_schemes(student, schemes)[["name", "amount", "deadline"]])