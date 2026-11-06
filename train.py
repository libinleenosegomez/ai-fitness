import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Expected columns:
    # age, gender, height_cm, weight_kg, activity, goal, equipment, level,
    # target_calories, protein_g, workout_plan

    feature_cols = [
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "activity",
        "goal",
        "equipment",
        "level",
    ]

    cat_cols = ["gender", "activity", "goal", "equipment", "level"]
    num_cols = ["age", "height_cm", "weight_kg"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # --- Workout plan classifier ---
    X = df[feature_cols]
    y_plan = df["workout_plan"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_plan, test_size=0.2, random_state=42, stratify=y_plan
    )

    plan_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=350, random_state=42)),
        ]
    )

    plan_model.fit(X_train, y_train)
    y_pred = plan_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Workout plan model accuracy: {acc:.4f}")

    joblib.dump(plan_model, os.path.join(MODELS_DIR, "workout_plan_model.joblib"))

    # --- Calories regressor (ML enhancement) ---
    if "target_calories" in df.columns:
        y_cals = df["target_calories"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cals, test_size=0.2, random_state=42
        )

        cal_model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("reg", RandomForestRegressor(n_estimators=400, random_state=42)),
            ]
        )
        cal_model.fit(X_train, y_train)
        pred = cal_model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        print(f"Calories model MAE: {mae:.1f} kcal")
        joblib.dump(cal_model, os.path.join(MODELS_DIR, "calorie_model.joblib"))

    # --- Clustering (unsupervised ML enhancement) ---
    # Cluster on the same processed features.
    X_proc = preprocessor.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    kmeans.fit(X_proc)
    joblib.dump(
        {"preprocessor": preprocessor, "kmeans": kmeans},
        os.path.join(MODELS_DIR, "user_cluster_model.joblib"),
    )

    print("Models saved to ./models")


if __name__ == "__main__":
    main()