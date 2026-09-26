"""
RecOps pipeline - Stage 2: validate.

Data validation gate: checks schema, nulls, allowed action vocabulary and
duplicate rows before anything downstream trains on the data. Writes a JSON
validation report (the pipeline's audit trail) and a cleaned event file.
Exits non-zero on a critical failure so `dvc repro` halts the pipeline -
that is exactly the behaviour a production data gate must have.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = ["user_id", "product_id", "action", "timestamp"]


def main():
    params = yaml.safe_load(open("params.yaml"))["validate"]
    df = pd.read_csv(params["input"])

    report = {"rows_in": int(len(df)), "checks": {}}
    critical_failure = False

    # 1. Schema check
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["checks"]["schema"] = f"FAIL: missing columns {missing}"
        critical_failure = True
    else:
        report["checks"]["schema"] = "PASS"

    if not critical_failure:
        # 2. Null check
        null_count = int(df[REQUIRED_COLUMNS].isna().sum().sum())
        report["checks"]["nulls"] = "PASS" if null_count == 0 else f"WARN: {null_count} nulls dropped"
        df = df.dropna(subset=REQUIRED_COLUMNS)

        # 3. Action vocabulary check
        allowed = set(params["allowed_actions"])
        bad_actions = df[~df["action"].isin(allowed)]
        report["checks"]["actions"] = (
            "PASS" if bad_actions.empty else f"WARN: {len(bad_actions)} rows with unknown actions dropped"
        )
        df = df[df["action"].isin(allowed)]

        # 4. Duplicate check (exact duplicate rows add no information)
        n_dupes = int(df.duplicated().sum())
        report["checks"]["duplicates"] = "PASS" if n_dupes == 0 else f"WARN: {n_dupes} duplicate rows dropped"
        df = df.drop_duplicates()

    report["rows_out"] = int(len(df))

    report_path = Path(params["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if critical_failure:
        print("VALIDATION FAILED - halting pipeline.", file=sys.stderr)
        sys.exit(1)

    out = Path(params["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Validate: wrote {len(df)} clean events -> {out}")


if __name__ == "__main__":
    main()