from flask import Flask, render_template, request, jsonify
from recommender import recommend

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/api/recommend")
def api_recommend():
    d = request.get_json(force=True)
    try:
        st = {"marks": float(d["marks"]), "income": float(d["income"]),
              "category": d["category"], "gender": d["gender"],
              "course": d["course"], "state": d["state"],
              "minority": int(d.get("minority", 0))}
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Please fill all fields correctly"), 400
    return jsonify(results=recommend(st, d.get("interests", "")))


if __name__ == "__main__":
    app.run(debug=True)