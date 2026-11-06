from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load trained models (created by train.py)
workout_plan_model = joblib.load(os.path.join(MODELS_DIR, "workout_plan_model.joblib"))

calorie_model_path = os.path.join(MODELS_DIR, "calorie_model.joblib")
calorie_model = joblib.load(calorie_model_path) if os.path.exists(calorie_model_path) else None

cluster_bundle = joblib.load(os.path.join(MODELS_DIR, "user_cluster_model.joblib"))
cluster_preprocessor = cluster_bundle["preprocessor"]
kmeans = cluster_bundle["kmeans"]

# Workout plans dictionary
workout_plans = {

    "gym_push_pull_legs": {
        "Day 1: Chest": [
            "Bench Press - 4x6-8",
            "Incline DB Press - 3x10",
            "Cable Fly - 3x12"
        ],
        "Day 2: Back": [
            "Deadlift - 4x5",
            "Pull-ups - 3x10",
            "Barbell Row - 3x8"
        ],
        "Day 3: Legs": [
            "Squats - 4x6-8",
            "Leg Press - 3x10",
            "Lunges - 3x12"
        ],
        "Day 4": ["Rest or light cardio"],
        "Day 5": ["Repeat Day 1"],
        "Day 6": ["Repeat Day 2"],
        "Day 7": ["Rest"]
    },

    "home_fatloss_beginner": {
        "Day 1": ["Jumping Jacks - 3x30s", "Push-ups - 3x10", "Squats - 3x15"],
        "Day 2": ["Burpees - 3x10", "Mountain Climbers - 3x20", "Plank - 3x30s"],
        "Day 3": ["High Knees - 3x30s", "Lunges - 3x12", "Sit-ups - 3x15"]
    },

    "home_strength_beginner": {
        "Day 1": ["Incline Push-ups - 3x10", "Bodyweight Squats - 3x15", "Plank - 3x30s"],
        "Day 2": ["Glute Bridges - 3x15", "Pike Push-ups - 3x8", "Dead Bug - 3x10/side"],
        "Day 3": ["Split Squats - 3x10/leg", "Chair Dips - 3x10", "Side Plank - 3x25s/side"]
    },

    "dumbbell_fatloss": {
        "Day 1": ["DB Thrusters - 3x12", "DB Row - 3x12", "Mountain Climbers - 3x30s"],
        "Day 2": ["Goblet Squat - 3x12", "DB Floor Press - 3x12", "Plank - 3x40s"],
        "Day 3": ["DB RDL - 3x12", "DB Shoulder Press - 3x12", "Jump Rope / High Knees - 3x45s"]
    },

    "gym_full_body": {
        "Day 1": ["Squat - 3x8", "Bench Press - 3x8", "Lat Pulldown - 3x10", "Plank - 3x40s"],
        "Day 2": ["Deadlift - 3x5", "Overhead Press - 3x8", "Seated Row - 3x10", "Crunches - 3x15"],
        "Day 3": ["Leg Press - 3x10", "Incline DB Press - 3x10", "Cable Row - 3x10", "Back Extension - 3x12"]
    },

    "gym_fatloss_split": {
        "Day 1: Upper (Fat loss)": ["Bench Press - 3x10", "Lat Pulldown - 3x10", "DB Shoulder Press - 3x12", "Finisher: Rowing - 10 min"],
        "Day 2: Lower (Fat loss)": ["Squat - 3x10", "RDL - 3x12", "Walking Lunges - 3x12/leg", "Finisher: Incline Walk - 12 min"],
        "Day 3: Conditioning": ["Bike Intervals - 8x30s hard/60s easy", "Core Circuit - 3 rounds"],
    },

    "dumbbell_strength_split": {
        "Day 1": ["DB Press - 3x10", "DB Row - 3x10", "Shoulder Press - 3x10"],
        "Day 2": ["Goblet Squat - 3x12", "Lunges - 3x12", "Calf Raises - 3x15"]
    },

    "recomp_hybrid_training": {
        "Day 1": ["Squats - 4x8", "Push-ups - 3x12", "Plank - 3x30s"],
        "Day 2": ["Deadlift - 4x6", "Pull-ups - 3x10", "Crunches - 3x15"]
    },

    # Legacy label (kept for backward compatibility with older models)
    "maintenance_full_body": {
        "Day 1": ["Bench - 3x10", "Squats - 3x10", "Row - 3x10"],
        "Day 2": ["Deadlift - 3x8", "Shoulder Press - 3x10", "Plank - 3x30s"]
    },

    "maintenance_gym_full_body": {
        "Day 1": ["Bench Press - 3x8", "Back Squat - 3x8", "Seated Row - 3x10", "Incline Walk - 10 min"],
        "Day 2": ["Deadlift - 3x5", "Overhead Press - 3x8", "Lat Pulldown - 3x10", "Core: Plank - 3x40s"],
        "Day 3": ["Leg Press - 3x10", "Incline DB Press - 3x10", "Cable Row - 3x10", "Bike - 12 min"],
    },

    "maintenance_dumbbell_full_body": {
        "Day 1": ["Goblet Squat - 3x12", "DB Floor Press - 3x12", "One-arm DB Row - 3x12", "Farmer Carry - 3x30s"],
        "Day 2": ["DB RDL - 3x12", "DB Shoulder Press - 3x12", "DB Split Squat - 3x10/leg", "Plank - 3x40s"],
        "Day 3": ["DB Lunges - 3x12/leg", "DB Chest Press - 3x12", "DB Row - 3x12", "Mountain Climbers - 3x30s"],
    },

    "maintenance_home_full_body": {
        "Day 1": ["Push-ups - 3x10", "Bodyweight Squats - 3x15", "Table Rows (or towel rows) - 3x10", "Plank - 3x35s"],
        "Day 2": ["Split Squats - 3x10/leg", "Chair Dips - 3x10", "Pike Push-ups - 3x8", "Dead Bug - 3x10/side"],
        "Day 3": ["Glute Bridges - 3x15", "Lunges - 3x12/leg", "Superman Hold - 3x30s", "Side Plank - 3x25s/side"],
    }
}

@app.route("/")
def home():
    # Serve the frontend from Flask so http://127.0.0.1:5000 shows the webpage.
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/health")
def health():
    return "Backend running"

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def bmr_mifflin(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
    s = 5 if gender == "male" else -161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s

def activity_factor(activity: str) -> float:
    return {"low": 1.375, "medium": 1.55, "high": 1.725}.get(activity, 1.55)

def goal_adjustment(goal: str) -> float:
    if goal == "fat_loss":
        return -400
    if goal == "muscle_gain":
        return 350
    if goal == "recomposition":
        return -150
    return 0

def cluster_label(cluster_id: int) -> str:
    labels = {
        0: "Cluster 0: Low activity / beginner-leaning",
        1: "Cluster 1: Higher activity / advanced-leaning",
        2: "Cluster 2: Average profile",
        3: "Cluster 3: Strength-focused mix",
    }
    return labels.get(int(cluster_id), f"Cluster {cluster_id}")

def normalize_to_n_days(plan: dict, n_days: int) -> dict:
    """
    Ensure the response contains exactly n_days entries.
    If a template has fewer days, append rest/mobility days.
    If it has more, truncate.
    """
    n_days = max(1, min(int(n_days), 7))
    items = list(plan.items())
    items = items[:n_days]

    rest_variants = [
        "Rest + light walk (20-30 min)",
        "Mobility + stretching (10-15 min)",
        "Easy cardio (zone 2) - 20 min",
        "Active recovery: yoga / foam roll",
    ]

    d = dict(items)
    if len(d) < n_days:
        existing_keys = set(d.keys())
        idx = 0
        next_day = len(d) + 1
        while len(d) < n_days and next_day <= 7:
            key = f"Day {next_day}"
            if key not in existing_keys:
                d[key] = [rest_variants[idx % len(rest_variants)]]
                idx += 1
            next_day += 1
    return d

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        age = _safe_int(data.get("age"))
        height = _safe_float(data.get("height", data.get("height_cm")))
        weight = _safe_float(data.get("weight", data.get("weight_kg")))
        goal = data.get("goal", "maintenance")
        days = int(data.get("days", 3))

        gender = data.get("gender", "male")
        activity = data.get("activity", "medium")
        equipment = data.get("equipment", "gym")
        level = data.get("level", "beginner")

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if age is None or height is None or weight is None:
        return jsonify({"error": "age, height, weight are required"}), 400

    features = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "height_cm": height,
                "weight_kg": weight,
                "activity": activity,
                "goal": goal,
                "equipment": equipment,
                "level": level,
            }
        ]
    )

    # ML prediction: workout plan label
    prediction = str(workout_plan_model.predict(features)[0]).strip()

    # Get plan (fallback: if legacy label or missing template, choose by equipment)
    pred_key = str(prediction).strip()
    if pred_key == "maintenance_full_body":
        pred_key = {
            "gym": "maintenance_gym_full_body",
            "dumbbells": "maintenance_dumbbell_full_body",
            "home": "maintenance_home_full_body",
        }.get(equipment, pred_key)

    plan = workout_plans.get(pred_key)
    if plan is None:
        # Equipment-aware fallback for any missing templates
        fallback_key = {
            ("maintenance", "gym"): "maintenance_gym_full_body",
            ("maintenance", "dumbbells"): "maintenance_dumbbell_full_body",
            ("maintenance", "home"): "maintenance_home_full_body",
        }.get((goal, equipment))
        plan = workout_plans.get(fallback_key) if fallback_key else None

    if plan is None:
        plan = {"No Plan": ["Try different inputs"]}

    # What we actually served (might differ due to equipment-aware fallback)
    served_plan_label = pred_key if pred_key in workout_plans else prediction

    # Normalize to requested days (up to 7)
    plan = normalize_to_n_days(plan, days)

    # Calculations
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    # Calories: compute baseline + optionally override with ML regressor prediction
    bmr = bmr_mifflin(age, gender, height, weight)
    maintenance = bmr * activity_factor(activity)
    target = maintenance + goal_adjustment(goal)

    if calorie_model is not None:
        try:
            target = float(calorie_model.predict(features)[0])
        except Exception:
            pass

    # Protein heuristic (kept simple for the project)
    protein = weight * (2.0 if goal == "muscle_gain" else 1.6)

    # Unsupervised ML: user segment
    try:
        x_proc = cluster_preprocessor.transform(features)
        cluster_id = int(kmeans.predict(x_proc)[0])
        cluster_text = cluster_label(cluster_id)
    except Exception:
        cluster_id = None
        cluster_text = None

    return jsonify({
        "bmi": round(bmi, 2),
        "bmi_status": bmi_status,
        "bmr": round(bmr),
        "maintenance_calories": round(maintenance),
        "target_calories": round(target),
        "protein": round(protein),
        "predicted_plan_label": served_plan_label,
        "user_cluster_id": cluster_id,
        "user_cluster_label": cluster_text,
        "workout_plan": plan
    })

if __name__ == "__main__":
    app.run(debug=True)