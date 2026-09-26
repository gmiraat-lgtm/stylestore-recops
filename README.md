# RecOps

**An End-to-End MLOps Pipeline for Personalized E-Commerce Product Recommendation with Interaction-Driven Drift Detection and Continuous Retraining**

TE7940 MLOps Lab ESE Project - Miraat Gupta | 23070126073 | AIML A3 | Symbiosis Institute of Technology, Pune

RecOps is a complete MLOps system built around [StyleStore](https://github.com/gmiraat-lgtm/e-commerce-nlp-stylestore), a self-hosted e-commerce platform with a 31,131-product catalog. User interaction events (views, clicks, add-to-carts, purchases) flow through a versioned, automated pipeline into a personalized recommendation model that is registered, gated, served, monitored for drift, and continuously retrained - with every stage observable from an in-store MLOps dashboard.

## What it does

- **Data and pipeline versioning (DVC):** a five-stage reproducible pipeline - ingest, validate (data quality gate that halts on schema failure), build features (weighted user-item interaction matrix), train, evaluate - over DVC-tracked event data with a local remote.
- **Model management (MLflow):** every run tracked; models registered as numbered versions; a metric-gated promotion script controls the `production` alias, so promotion is the single switch that changes what is served.
- **Serving (Flask):** the recommender is served over HTTP, resolved through the registry alias, with hot reload on promotion. A "Recommended for you" shelf in the storefront renders the results per logged-in user, badge-stamped with the serving model version.
- **Drift detection and automated retraining (Evidently):** weekly event batches are compared against training data; detected drift automatically triggers merge, retrain, gated promotion, and serving reload - while a no-drift batch is archived untouched. Both behaviours are covered by recorded positive and negative tests.
- **CI/CD model factory (GitHub Actions + Docker):** every push runs unit tests, a full pipeline smoke run on a committed sample catalog, a metric sanity gate, a Docker image build with the model baked in (immutable-image pattern), and a live container smoke test.
- **A/B champion-challenger serving:** two registry aliases served simultaneously with deterministic 50/50 user assignment; live shelf impressions and clicks decide the winner by click-through rate - the answer to offline metrics becoming incomparable under drift.
- **Observability and governance:** an `/mlops` dashboard inside the store (production version, event counts, drift verdict, registry history, live A/B scoreboard) and a Report Center of downloadable artifacts - Evidently drift report, plain-language drift summary, per-version model cards with DVC data lineage, quality report, and A/B experiment report.

## Repository layout

    src/        pipeline stages, simulator, drift monitor, orchestrator, promotion gate, report generators
    serving/    Flask serving API (registry-resolved dev server + immutable container variant)
    tests/      unit tests + CI sample catalog
    .github/    CI workflow (test -> pipeline smoke -> metric gate -> docker build -> container smoke)
    dvc.yaml    the five-stage pipeline definition
    params.yaml single control panel for every stage

## Quick start

    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    python src/simulate_events.py --users 2000 --events 200000
    dvc repro
    python src/promote.py
    python serving/app.py          # http://localhost:8001/health

See `DEMO_RUNBOOK.md` for the full demonstration script (personalization feedback loop, drift-triggered automated retraining, the no-drift negative test, and the A/B experiment).
