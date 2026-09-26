"""
RecOps - Drift monitor (Evidently).

Compares an incoming production event batch against the reference data the
production model was trained on. Events are enriched with catalog attributes
(category, gender, price) so the comparison captures WHAT users engage with,
not just raw ids. Produces THREE artifacts: Evidently's full technical HTML
report, a machine-readable drift_status.json, and a plain-language
drift_summary.html written for non-technical readers, with download links to
the underlying data (Report Center artifact #1). Exits 0 (no drift) or 42
(drift detected) so an orchestrator can branch on the verdict.

Usage: python src/monitor.py --batch data/incoming/batch_001.csv
"""

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from evidently import Report
from evidently.presets import DataDriftPreset

FEATURES = ["action", "category", "gender", "price"]

FRIENDLY = {
    "category": ("Product categories browsed",
                 "Which kinds of products (shoes, kurtas, watches...) customers are engaging with."),
    "price":    ("Price range of viewed products",
                 "How expensive the products customers look at are."),
    "gender":   ("Gender section shopped",
                 "Whether activity concentrates in Men's, Women's, Unisex or Kids sections."),
    "action":   ("Shopping behaviour funnel",
                 "The mix of viewing, clicking, adding to cart and purchasing."),
}

NOTICEABLE = 0.1  # heuristic: scores above this read as "changed noticeably"


def enrich(events: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    merged = events.merge(
        catalog[["product_id", "category", "gender", "price"]],
        on="product_id", how="inner",
    )
    return merged[FEATURES]


def parse_results(result_dict: dict):
    """Pull dataset share plus per-column drift scores out of Evidently's result."""
    share, count, columns = None, None, {}
    for metric in result_dict.get("metrics", []):
        mid = str(metric.get("metric_id", ""))
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            share = float(value["share"])
            count = int(value.get("count", -1))
        else:
            m = re.search(r"column=([A-Za-z_]+)", mid)
            if m and isinstance(value, (int, float)):
                columns[m.group(1)] = float(value)
    return share, count, columns


def write_summary_html(path, status, columns, n_ref, n_cur, batch_path, cfg):
    drifted = status["drift_detected"]
    color = "#b91c1c" if drifted else "#166534"
    verdict = ("CHANGE DETECTED - the model's picture of customers is going stale"
               if drifted else
               "NO SIGNIFICANT CHANGE - the current model still matches customer behaviour")
    action = ("Because the change crossed the alert threshold, the system automatically "
              "retrains the recommendation model on data that includes this new behaviour, "
              "checks the new model's quality against the old one, and only then puts it "
              "live on the storefront. No engineer needs to intervene."
              if drifted else
              "Nothing needs to happen. Retraining on every batch would waste computation "
              "without improving recommendations, so the system deliberately stays put.")

    rows = []
    for col, score in sorted(columns.items(), key=lambda x: -x[1]):
        name, meaning = FRIENDLY.get(col, (col, ""))
        word = ("changed noticeably" if score >= NOTICEABLE else "essentially unchanged")
        rows.append(f"""
        <tr>
          <td><b>{html.escape(name)}</b><br><span class="muted">{html.escape(meaning)}</span></td>
          <td class="score">{score:.3f}</td>
          <td>{word}</td>
        </tr>""")

    links = f"""
      <li><a href="../../{batch_path}">Incoming batch events (CSV)</a> - the raw week of traffic that was analysed</li>
      <li><a href="../../{cfg['reference']}">Reference training events (CSV)</a> - what the current model learned from</li>
      <li><a href="drift_status.json">Machine-readable verdict (JSON)</a> - used by the automation</li>
      <li><a href="drift_report.html">Full technical drift report (Evidently)</a> - per-column statistics and distribution charts</li>
    """

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RecOps Drift Summary</title>
<style>
 body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 860px;
        margin: 40px auto; color: #1f2937; line-height: 1.6; padding: 0 16px; }}
 h1 {{ font-size: 1.6em; }} .muted {{ color: #6b7280; font-size: 0.9em; }}
 .verdict {{ border-left: 6px solid {color}; background: #f9fafb;
            padding: 12px 18px; font-weight: bold; color: {color}; }}
 table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
 td, th {{ border: 1px solid #d1d5db; padding: 10px; text-align: left; vertical-align: top; }}
 th {{ background: #f3f4f6; }} .score {{ font-family: monospace; }}
 footer {{ margin-top: 30px; font-size: 0.85em; color: #6b7280; }}
</style></head><body>
<h1>Customer Behaviour Drift Summary</h1>
<p class="muted">RecOps monitoring - generated {datetime.now().strftime('%d %b %Y, %H:%M')}
 - batch: {html.escape(str(batch_path))}</p>

<div class="verdict">{verdict}</div>

<h2>What was checked, in plain terms</h2>
<p>A recommendation model is only as good as its understanding of current customer
behaviour. This check compared <b>{n_cur:,} interaction events from the newest week of
store traffic</b> against <b>{n_ref:,} events the current model was trained on</b>. If
customers now browse different categories, price ranges or sections than before, the
model's suggestions gradually become irrelevant - this is called <i>data drift</i>.</p>

<h2>What changed and what did not</h2>
<p class="muted">The change score measures how different the new week looks from the
training data (0 = identical, higher = more different). Scores above {NOTICEABLE} read
as a noticeable shift.</p>
<table>
<tr><th>Aspect of customer behaviour</th><th>Change score</th><th>Reading</th></tr>
{''.join(rows)}
</table>
<p>Overall, {status['drifted_columns_count']} of 4 monitored aspects drifted beyond
Evidently's statistical test - a share of {status['drifted_columns_share']:.0%}, against
an alert threshold of {status['threshold']:.0%}.</p>

<h2>What happens next</h2>
<p>{action}</p>

<h2>Download the evidence</h2>
<ul>{links}</ul>

<footer>RecOps - TE7940 MLOps Lab ESE Project - Miraat Gupta | 23070126073 | AIML - A3.
This page is a plain-language companion; the Evidently technical report linked above is
the statistical source of truth.</footer>
</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("params.yaml"))["monitor"]

    catalog = pd.read_csv(cfg["catalog"])
    reference = enrich(pd.read_csv(cfg["reference"]), catalog)
    if len(reference) > cfg["reference_sample"]:
        reference = reference.sample(cfg["reference_sample"], random_state=42)
    current = enrich(pd.read_csv(args.batch), catalog)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)

    Path(cfg["report_html"]).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(cfg["report_html"])

    share, count, columns = parse_results(result.dict())
    if share is None:
        print("WARNING: could not parse drift share from Evidently result")
        share, count = -1.0, -1

    drifted = share >= cfg["drift_share_threshold"]
    status = {
        "batch": args.batch,
        "drifted_columns_share": share,
        "drifted_columns_count": count,
        "threshold": cfg["drift_share_threshold"],
        "drift_detected": bool(drifted),
    }
    Path(cfg["status_json"]).write_text(json.dumps(status, indent=2))

    summary_path = cfg.get("summary_html", "reports/drift/drift_summary.html")
    write_summary_html(summary_path, status, columns,
                       len(reference), len(current),
                       args.batch.replace("\\", "/"), cfg)

    print(json.dumps(status, indent=2))
    print(f"Technical report -> {cfg['report_html']}")
    print(f"Plain-language summary -> {summary_path}")

    sys.exit(42 if drifted else 0)


if __name__ == "__main__":
    main()