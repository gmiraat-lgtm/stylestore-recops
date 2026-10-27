"""
RecOps - A/B experiment report generator (Report Center artifact #4).

Summarises the live champion-challenger experiment from the impression and
click log written by the serving API: per-variant impressions, clicks and
click-through rate, with a plain-language verdict. Regenerate at any time;
the page reflects the log as of generation.

Usage: python src/generate_ab_report.py
"""

import csv
from datetime import datetime
from pathlib import Path

LOG = Path("reports/generated/ab_events.csv")
OUT = Path("reports/generated/ab_report.html")

STYLE = """
 body { font-family: Georgia, 'Times New Roman', serif; max-width: 860px;
        margin: 40px auto; color: #1f2937; line-height: 1.6; padding: 0 16px; }
 h1 { font-size: 1.6em; } .muted { color: #6b7280; font-size: 0.9em; }
 table { border-collapse: collapse; width: 100%; margin: 18px 0; }
 td, th { border: 1px solid #d1d5db; padding: 10px; text-align: left; }
 th { background: #f3f4f6; }
 .lead { background: #ecfdf5; }
 footer { margin-top: 30px; font-size: 0.85em; color: #6b7280; }
"""


def main():
    counts = {"champion": {"impressions": 0, "clicks": 0, "version": "-"},
              "challenger": {"impressions": 0, "clicks": 0, "version": "-"}}
    if LOG.exists():
        with open(LOG, newline="") as f:
            for row in csv.DictReader(f):
                v = row["variant"]
                if v in counts:
                    counts[v]["version"] = row["model_version"]
                    key = "impressions" if row["event"] == "impression" else "clicks"
                    counts[v][key] += 1

    for c in counts.values():
        c["ctr"] = (c["clicks"] / c["impressions"]) if c["impressions"] else None

    ch, cl = counts["champion"], counts["challenger"]
    both_measured = ch["ctr"] is not None and cl["ctr"] is not None
    small = min(ch["impressions"], cl["impressions"]) < 100
    if not both_measured:
        verdict = ("The experiment is running but one variant has no traffic yet - "
                   "no comparison is possible.")
    else:
        leader = "challenger" if cl["ctr"] > ch["ctr"] else "champion"
        verdict = (f"On current data the <b>{leader}</b> shows the higher click-through "
                   f"rate. ")
        if small:
            verdict += ("However, with fewer than 100 impressions per variant this "
                        "difference is anecdotal, not statistically meaningful - the "
                        "correct operational decision is to keep the experiment running "
                        "and let more traffic accumulate before promoting either model.")

    def row(name, c, lead):
        cls = ' class="lead"' if lead else ""
        ctr = f"{c['ctr']:.2%}" if c["ctr"] is not None else "-"
        return (f"<tr{cls}><td>{name} (v{c['version']})</td>"
                f"<td>{c['impressions']}</td><td>{c['clicks']}</td><td>{ctr}</td></tr>")

    lead_ch = both_measured and ch["ctr"] >= cl["ctr"]
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>A/B Experiment Report</title><style>{STYLE}</style></head><body>
<h1>A/B Experiment Report - Champion vs Challenger</h1>
<p class="muted">Generated {datetime.now().strftime('%d %b %Y, %H:%M')} from the live
shelf engagement log.</p>

<h2>What this experiment is</h2>
<p>Two recommendation models serve the storefront simultaneously: the production
<b>champion</b> and a <b>challenger</b> candidate. Every user is deterministically
assigned to one variant, so each shopper consistently sees one model's
recommendations. An <i>impression</i> is one rendering of the "Recommended for you"
shelf; a <i>click</i> is a shopper opening a recommended product. Click-through rate
(clicks / impressions) measures which model produces recommendations shoppers
actually want - a live judgement that stays valid even when offline metrics become
incomparable under data drift.</p>

<h2>Results so far</h2>
<table>
<tr><th>Variant</th><th>Impressions</th><th>Clicks</th><th>CTR</th></tr>
{row('Champion', ch, lead_ch)}
{row('Challenger', cl, both_measured and not lead_ch)}
</table>

<h2>Reading and decision</h2>
<p>{verdict}</p>

<footer>RecOps - TE7940 MLOps Lab ESE Project - Miraat Gupta | 23070126073 | AIML - A3.</footer>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"A/B report -> {OUT}")


if __name__ == "__main__":
    main()
