# RecOps Demo Runbook

Miraat Gupta | 23070126073 | AIML - A3 | TE7940 MLOps Lab ESE

## Start the system (3 terminals)
1. RecOps serving:   cd D:\stylestore-recops  ->  .\venv\Scripts\Activate  ->  python serving\app.py
2. StyleStore back:  cd "C:\Users\MIRAAT GUPTA\ecommerce-nlp\backend"  ->  .\venv\Scripts\Activate  ->  python main.py
3. StyleStore front: cd "C:\Users\MIRAAT GUPTA\ecommerce-nlp\frontend"  ->  npm run dev
Open http://localhost:5173 -> Login (Developer Login) -> homepage shows
"Recommended for you" with the variant + model version badge.
Demo account maps to simulated user u01615. MLOps dashboard: http://localhost:5173/mlops

## Demo A - Personalization feedback loop (terminal-driven, ~2 min)
Working terminal: D:\stylestore-recops with venv active.
1. Show current shelf + badge version.
2. python src\demo_reset_user.py            (wipes u01615, injects mid-popularity sneaker browsing)
3. dvc add data\raw\events.csv
4. dvc repro                                 (full pipeline, ~30s)
5. python src\promote.py --force
6. curl.exe -X POST http://localhost:8001/reload
7. Hard-refresh homepage (Ctrl+Shift+R): badge +1, shelf now sneakers.
Talking point: user behaviour -> versioned data -> retrain -> registry ->
hot reload; the site changed with zero code changes.

## Demo B - Drift detection + automated retraining (~3 min)
1. Drifted week (festive ethnic surge):
   python src\simulate_events.py --users 2000 --events 30000 --days 7 --end-date <NEXT-DATE> --mix "ethnic:0.55,formal:0.25,accessories:0.20" --out data\incoming\batch_XXX.csv --users-out data\incoming\users_batch_XXX.csv
2. python src\auto_retrain.py --batch data\incoming\batch_XXX.csv
   -> monitor detects drift (exit 42) -> merge -> dvc repro -> promote -> hot reload -> archive.
3. Show reports\drift\drift_summary.html (plain language) and drift_report.html (Evidently).
4. Hard-refresh homepage: badge +1 without any manual retrain command.

## Demo C - Negative test (no drift, ~1 min)
1. Normal week: same simulate_events command with default mix (omit --mix).
2. python src\auto_retrain.py --batch data\incoming\batch_YYY.csv
   -> VERDICT: no significant drift. Model unchanged. Batch archived.
Talking point: monitoring means knowing when NOT to retrain.

## Demo D - A/B champion-challenger (~2 min)
1. Dashboard http://localhost:5173/mlops -> A/B panel: live impressions/clicks/CTR per variant.
2. Click products ON THE SHELF (not the All Products grid) -> refresh dashboard: numbers move.
3. Different accounts hash to different variants: badge shows "champion - v8" (indigo)
   or "challenger - v9" (amber). python src\generate_ab_report.py refreshes the
   downloadable A/B report in the Report Center.
Talking point: live CTR comparison replaces offline metrics, which become
incomparable under drift (stale-champion problem).

## Evidence surfaces
- MLflow UI: mlflow ui -> http://127.0.0.1:5000 -> "Model training" pill -> runs + Model registry. Stop with single Ctrl+C.
- dvc dag, dvc metrics show
- GitHub Actions: github.com/gmiraat-lgtm/stylestore-recops -> Actions (CI model factory)

## Recovery
- Blank page / "recops offline" -> one of the 3 services died; restart it, refresh.
- Gate blocks under drift -> python src\promote.py --force (stale-champion policy, see report).
- Batch numbers: use the next unused batch_XXX; end-date = any later date.
