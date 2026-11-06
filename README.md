# AI Fitness Recommendation (ML Project)

This is a small college-friendly project that recommends a workout plan and nutrition targets using:
- **Supervised ML**: workout plan classifier
- **Supervised ML**: target calories regressor
- **Unsupervised ML**: user segmentation with KMeans

## Setup (Windows / PowerShell)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate dataset (optional)

```bash
python generate_dataset.py
```

## Train ML models

```bash
python train.py
```

This creates a `models/` folder with:
- `workout_plan_model.joblib`
- `calorie_model.joblib`
- `user_cluster_model.joblib`

## Run backend API

```bash
python backend.py
```

Backend runs at `http://127.0.0.1:5000`.

## Run frontend

Open `index.html` in your browser and click **Generate Plan**.

If CORS or file restrictions happen in your browser, run a simple static server:

```bash
python -m http.server 8000
```

Then open `http://127.0.0.1:8000/index.html`.

