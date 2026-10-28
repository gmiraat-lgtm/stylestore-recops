"""
RecOps - Report Center refresh hook.

Regenerates the model card of EVERY registered version plus the quality
report. promote.py calls this automatically after every registry change,
so a new version has its model card the moment it exists - the dashboard
can never link to a card that was never written.

Usage: python src/refresh_reports.py
"""

from mlflow import MlflowClient

from generate_reports import (OUT, events_dvc_hash, fetch_versions,
                              write_model_card, write_quality_report)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    versions = fetch_versions(client)
    dvc_hash = events_dvc_hash()
    for v in versions:
        write_model_card(v, dvc_hash)
    write_quality_report(versions)
    print(f"Report Center refreshed: {len(versions)} model cards + quality report")


if __name__ == "__main__":
    main()
