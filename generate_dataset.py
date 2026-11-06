import pandas as pd
import random

data = []

genders = ["male", "female"]
activities = ["low", "medium", "high"]
goals = ["fat_loss", "muscle_gain", "maintenance", "recomposition"]
equipment = ["gym", "home", "dumbbells"]
levels = ["beginner", "intermediate", "advanced"]

def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

def get_level(activity):
    if activity == "low":
        return "beginner"
    elif activity == "medium":
        return "intermediate"
    else:
        return "advanced"

def get_plan(goal, equipment, level):
    if goal == "fat_loss":
        if equipment == "gym":
            return "gym_fatloss_split" if level != "beginner" else "gym_full_body"
        elif equipment == "home":
            return "home_fatloss_beginner"
        else:
            return "dumbbell_fatloss"

    if goal == "muscle_gain":
        if equipment == "gym":
            return "gym_push_pull_legs" if level != "beginner" else "gym_full_body"
        elif equipment == "home":
            return "home_strength_beginner"
        else:
            return "dumbbell_strength_split"

    if goal == "recomposition":
        return "recomp_hybrid_training"

    # Maintenance should differ by equipment so ML can learn it.
    if equipment == "gym":
        return "maintenance_gym_full_body"
    if equipment == "dumbbells":
        return "maintenance_dumbbell_full_body"
    return "maintenance_home_full_body"

def activity_factor(activity: str) -> float:
    return {"low": 1.375, "medium": 1.55, "high": 1.725}[activity]

def bmr_mifflin(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
    # Mifflin–St Jeor. Used only to generate a realistic synthetic target.
    s = 5 if gender == "male" else -161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s

def target_calories_from_goal(tdee: float, goal: str) -> float:
    if goal == "fat_loss":
        return tdee - 400
    if goal == "muscle_gain":
        return tdee + 350
    if goal == "recomposition":
        return tdee - 150
    return tdee


for _ in range(800):   # bigger dataset for ML
    age = random.randint(18, 50)
    gender = random.choice(genders)

    height = random.randint(150, 190)
    weight = random.randint(height - 100, height - 70)

    activity = random.choice(activities)
    goal = random.choice(goals)
    equip = random.choice(equipment)

    level = get_level(activity)
    plan = get_plan(goal, equip, level)

    bmr = bmr_mifflin(age, gender, height, weight)
    tdee = bmr * activity_factor(activity)
    target_cals = target_calories_from_goal(tdee, goal)

    # Add noise to make regression non-trivial but learnable.
    target_cals = max(1200, target_cals + random.gauss(0, 120))

    # Simple protein target heuristic (g/kg) with noise.
    protein_g = weight * (2.0 if goal == "muscle_gain" else 1.6)
    protein_g = max(60, protein_g + random.gauss(0, 10))

    data.append([
        age, gender, height, weight,
        activity, goal, equip, level,
        round(target_cals, 1), round(protein_g, 1),
        plan
    ])

df = pd.DataFrame(data, columns=[
    "age", "gender", "height_cm", "weight_kg",
    "activity", "goal", "equipment", "level",
    "target_calories", "protein_g",
    "workout_plan"
])

df.to_csv("dataset.csv", index=False)

print("Dataset generated")