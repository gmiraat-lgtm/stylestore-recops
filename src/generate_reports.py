"""
RecOps - Report Center generators (model card + quality report).

Produces the two governance documents of the Report Center from live MLflow
registry data, in the same plain-language house style as the drift summary:

  model_card_v<N>.html   - one card per registered model version: identity,
                           training data lineage (DVC hash of events.csv),
                           hyperparameters, offline metrics, promotion status.
                           This is the audit artifact model governance (CO5)
                           asks for: for any model that ever served users, who
                           trained it, on what data, and how good it was.
  quality_report.html    - all versions side by side with the production one
                           highlighted, so quality can be tracked across the
                           retrain history.

Usage:
  python src/generate_reports.py                (card for production + quality report)
  python src/generate_reports.py --version 5    (card for a specific version)
"""

import argparse
import html
from datetime import datetime
from pathlib import Path

import yaml
from mlflow import MlflowClient

MODEL_NAME = "recops-item-cf"
ALIAS = "production"
OUT = Path("reports/generated")

STYLE = """
 body { font-family: Georgia, 'Times New Roman', serif; max-width: 860px;
        margin: 40px auto; color: #1f2937; line-height: 1.6; padding: 0 16px; }
 h1 { font-size: 1.6em; } .muted { color: #6b7280; font-size: 0.9em; }
 table { border-collapse: collapse; width: 100%; margin: 18px 0; }
 td, th { border: 1px solid #d1d5db; padding: 10px; text-align: left; }
 th { background: #f3f4f6; } .mono { font-family: monospace; font-size: 0.9em; }
 .prod { background: #ecfdf5; }
 .badge { display:inline-block; padding:2px 10px; border-radius:999px;
          background:#e0e7ff; color:#3730a3; font-size:0.85em; }
 footer { margin-top: 30px; font-size: 0.85em; color: #6b7280; }
"""

FOOTER = ("<footer>RecOps - TE7940 MLOps Lab ESE Project - "
          "Miraat Gupta | 23070126073 | AIML - A3.</footer>")


def events_dvc_hash():
    try:
        meta = yaml.safe_load(open("data/raw/events.csv.dvc"))
        return meta["outs"][0]["md5"]
    except Exception:
        return "unavailable"


def fetch_versions(client):
    out = []
    try:
        prod_v = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except Exception:
        prod_v = None
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        run = client.get_run(mv.run_id)
        out.append({
            "version": int(mv.version),
            "run_id": mv.run_id,
            "created": datetime.fromtimestamp(mv.creation_timestamp / 1000)
                       .strftime("%d %b %Y, %H:%M"),
            "is_production": str(mv.version) == str(prod_v),
            "params": run.data.params,
            "metrics": run.data.metrics,
        })
    out.sort(key=lambda v: v["version"], reverse=True)
    return out


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def write_model_card(v, dvc_hash):
    m, p = v["metrics"], v["params"]
    status = ("IN PRODUCTION - this version currently serves the storefront"
              if v["is_production"] else
              "Registered, not currently serving")
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Model Card v{v['version']}</title><style>{STYLE}</style></head><body>
<h1>Model Card - recops-item-cf <span class="badge">version {v['version']}</span></h1>
<p class="muted">Generated {datetime.now().strftime('%d %b %Y, %H:%M')}</p>

<h2>What this model is</h2>
<p>An item-based collaborative filtering recommender for the StyleStore catalog.
It learns which products attract the same customers and recommends, for each
user, popular products similar to what that user has engaged with but has not
seen yet. It contains no demographic or personal attributes - only anonymised
interaction strengths between user ids and product ids.</p>

<h2>Identity and lineage</h2>
<table>
<tr><th>Registered version</th><td>{v['version']}</td></tr>
<tr><th>Status</th><td>{status}</td></tr>
<tr><th>Created</th><td>{v['created']}</td></tr>
<tr><th>MLflow run id</th><td class="mono">{v['run_id']}</td></tr>
<tr><th>Training data (DVC md5 of events.csv)</th><td class="mono">{dvc_hash}</td></tr>
<tr><th>Interactions trained on</th><td>{html.escape(str(p.get('n_interactions','-')))}</td></tr>
<tr><th>Users / items (raw)</th>
    <td>{html.escape(str(p.get('n_users','-')))} / {html.escape(str(p.get('n_items_raw','-')))}</td></tr>
</table>

<h2>Hyperparameters</h2>
<table>
<tr><th>top_k_neighbors</th><td>{html.escape(str(p.get('top_k_neighbors','-')))}</td></tr>
<tr><th>min_item_interactions</th><td>{html.escape(str(p.get('min_item_interactions','-')))}</td></tr>
</table>

<h2>Offline evaluation</h2>
<p class="muted">Per-user holdout: hide a few interacted items per user, retrain
on the rest, measure recovery in the top 10 recommendations.</p>
<table>
<tr><th>precision@10</th><td>{fmt(m.get('precision_at_10'))}</td></tr>
<tr><th>recall@10</th><td>{fmt(m.get('recall_at_10'))}</td></tr>
<tr><th>category relevance@10</th><td>{fmt(m.get('category_relevance_at_10'))}</td></tr>
<tr><th>training time (s)</th><td>{fmt(m.get('train_seconds'), 2)}</td></tr>
</table>

<h2>Intended use and limits</h2>
<p>Built for coursework demonstration on simulated interaction data against a
real product catalog. Known limits: cold-start users receive no recommendations
(the storefront hides the shelf); popularity concentration can under-serve
long-tail products; under data drift the offline metrics of older versions are
not comparable with newer ones (see project report, stale-champion discussion).</p>
{FOOTER}</body></html>"""
    path = OUT / f"model_card_v{v['version']}.html"
    path.write_text(doc, encoding="utf-8")
    return path


def write_quality_report(versions):
    rows = []
    for v in versions:
        m = v["metrics"]
        cls = ' class="prod"' if v["is_production"] else ""
        tag = " (production)" if v["is_production"] else ""
        rows.append(f"""<tr{cls}><td>v{v['version']}{tag}</td><td>{v['created']}</td>
        <td>{fmt(m.get('precision_at_10'))}</td><td>{fmt(m.get('recall_at_10'))}</td>
        <td>{fmt(m.get('category_relevance_at_10'))}</td>
        <td><a href="model_card_v{v['version']}.html">card</a></td></tr>""")
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Recommendation Quality Report</title><style>{STYLE}</style></head><body>
<h1>Recommendation Quality Report</h1>
<p class="muted">Generated {datetime.now().strftime('%d %b %Y, %H:%M')} -
every registered version of recops-item-cf, newest first. The highlighted row
is the version currently serving the storefront.</p>
<table>
<tr><th>Version</th><th>Created</th><th>precision@10</th><th>recall@10</th>
<th>category relevance@10</th><th>Model card</th></tr>
{''.join(rows)}
</table>
<p>Reading guide: category relevance@10 measures whether recommendations stay
inside each user's taste segments (the personalization property the drift
monitor protects); precision/recall@10 measure exact recovery of hidden items
and are only comparable between versions evaluated on the same dataset.</p>
{FOOTER}</body></html>"""
    path = OUT / "quality_report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None,
                    help="write a card for this version (default: production)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    versions = fetch_versions(client)
    dvc_hash = events_dvc_hash()

    targets = [v for v in versions
               if (args.version and v["version"] == args.version)
               or (not args.version and v["is_production"])]
    for v in targets:
        print(f"Model card    -> {write_model_card(v, dvc_hash)}")
    print(f"Quality report -> {write_quality_report(versions)}")


if __name__ == "__main__":
    main()
