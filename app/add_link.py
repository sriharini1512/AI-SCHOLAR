    1: "https://real-official-link-for-scheme-1",
    5: "https://college-site-link",
    import pandas as pd

df = pd.read_csv("data/scholarships.csv")

# Put the REAL official application link for each scheme_id here.
# Schemes not listed stay blank and the website shows a "Find Apply Page" search button instead.
NSP = "https://scholarships.gov.in"
links = {
    2: NSP,
    3: NSP,
    4: NSP,
    13: NSP,
    # 1: "https://...",   # add real links like this
}

if "apply_url" not in df.columns:
    df["apply_url"] = ""
for sid, url in links.items():
    df.loc[df["scheme_id"] == sid, "apply_url"] = url

df.to_csv("data/scholarships.csv", index=False)
print(df[["scheme_id", "name", "apply_url"]])