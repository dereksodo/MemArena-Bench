#!/usr/bin/env python3
"""Local pre-flight check for a Croissant 1.0 JSON-LD file.

Strategy:
  1. If the official `mlcroissant` package is installed, use its validator.
  2. Otherwise, run a conservative JSON-LD + RAI-field check that matches
     what the MLCommons validator and the HuggingFace checker space look at.

Exit codes:
  0 — OK, no errors.
  1 — one or more errors reported.
  2 — cannot read the file.

Usage:
  python -m memarena.tools.validate_croissant docs/memarena-croissant.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_CORE = [
    "@context",
    "@type",
    "name",
    "url",
    "license",
    "conformsTo",
    "distribution",
    "recordSet",
]
REQUIRED_RAI = [
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
]
PROV_FIELDS = ["prov:wasDerivedFrom", "prov:wasGeneratedBy"]
CR_NAMESPACE = "http://mlcommons.org/croissant/"
RAI_NAMESPACE = "http://mlcommons.org/croissant/RAI/"
CONFORMS_TO_VALUE = "http://mlcommons.org/croissant/1.0"


def try_mlcroissant(path: Path) -> list[str] | None:
    """Return list of error strings if mlcroissant is installed, else None."""
    try:
        import mlcroissant as mlc  # type: ignore
    except ImportError:
        return None
    errors: list[str] = []
    try:
        mlc.Dataset(jsonld=str(path))
    except Exception as exc:
        errors.append(f"mlcroissant: {exc}")
    return errors


def basic_check(doc: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_CORE:
        if field not in doc:
            errors.append(f"missing required core field: {field}")

    ctx = doc.get("@context") or {}
    if not isinstance(ctx, dict):
        errors.append("@context must be an object mapping prefix → URI")
    else:
        if ctx.get("cr") != CR_NAMESPACE:
            errors.append(f'@context.cr must be "{CR_NAMESPACE}"')
        if ctx.get("rai") != RAI_NAMESPACE:
            errors.append(f'@context.rai must be "{RAI_NAMESPACE}"')
        if "dct" not in ctx:
            errors.append("@context must define dct prefix")

    if doc.get("@type") not in ("sc:Dataset", "Dataset", "schema.org/Dataset"):
        errors.append(f'@type should be "sc:Dataset", got {doc.get("@type")!r}')

    if doc.get("conformsTo") != CONFORMS_TO_VALUE:
        errors.append(f'conformsTo should be "{CONFORMS_TO_VALUE}"')

    for field in REQUIRED_RAI:
        if field not in doc:
            errors.append(f"missing RAI field: {field}")

    missing_prov = [f for f in PROV_FIELDS if f not in doc]
    if missing_prov:
        errors.append(
            f"missing provenance fields (NeurIPS 2026 guideline): {', '.join(missing_prov)}"
        )

    # distribution checks
    dist = doc.get("distribution") or []
    if not isinstance(dist, list) or not dist:
        errors.append("distribution must be a non-empty array")
    else:
        for i, f in enumerate(dist):
            if f.get("@type") not in ("cr:FileObject", "cr:FileSet"):
                errors.append(
                    f"distribution[{i}] has unexpected @type={f.get('@type')!r}"
                )
            if not f.get("@id"):
                errors.append(f"distribution[{i}] missing @id")
            if not f.get("name"):
                errors.append(f"distribution[{i}] missing name")
            if f.get("@type") == "cr:FileObject" and not f.get("containedIn"):
                # top-level files need contentUrl and sha256
                if "contentUrl" not in f:
                    errors.append(
                        f"distribution[{i}] ({f.get('name')}) top-level FileObject missing contentUrl"
                    )
                if "sha256" not in f and "md5" not in f:
                    errors.append(
                        f"distribution[{i}] ({f.get('name')}) top-level FileObject missing sha256 or md5"
                    )

    # recordSet checks. NeurIPS 2026 lists recordSet and at least one field
    # per RecordSet among the required core Croissant metadata.
    recs = doc.get("recordSet") or []
    if not isinstance(recs, list):
        errors.append("recordSet must be an array")
    elif not recs:
        errors.append("recordSet must be a non-empty array")
    else:
        for i, rec in enumerate(recs):
            if not isinstance(rec, dict):
                errors.append(f"recordSet[{i}] must be an object")
                continue
            if rec.get("@type") not in ("cr:RecordSet", "RecordSet"):
                errors.append(
                    f"recordSet[{i}] has unexpected @type={rec.get('@type')!r}"
                )
            if not rec.get("@id"):
                errors.append(f"recordSet[{i}] missing @id")
            if not rec.get("name"):
                errors.append(f"recordSet[{i}] missing name")
            fields = rec.get("field") or []
            if not isinstance(fields, list) or not fields:
                errors.append(
                    f"recordSet[{i}] ({rec.get('name')}) field must be a non-empty array"
                )
                continue
            for j, fld in enumerate(fields):
                if not isinstance(fld, dict):
                    errors.append(f"recordSet[{i}].field[{j}] must be an object")
                    continue
                if fld.get("@type") not in ("cr:Field", "Field"):
                    errors.append(
                        f"recordSet[{i}].field[{j}] has unexpected @type={fld.get('@type')!r}"
                    )
                if not fld.get("@id"):
                    errors.append(f"recordSet[{i}].field[{j}] missing @id")
                if not fld.get("dataType"):
                    errors.append(f"recordSet[{i}].field[{j}] missing dataType")
                if "source" not in fld and "value" not in fld:
                    errors.append(f"recordSet[{i}].field[{j}] missing source or value")

    # placeholder sniff
    dumped = json.dumps(doc)
    for ph in ("PLACEHOLDER_HF_DATASET_URL", "PLACEHOLDER_SHA256"):
        if ph in dumped:
            errors.append(
                f"placeholder {ph} still present — run patch_croissant_rai.py "
                f"after Hugging Face upload to substitute real values"
            )

    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Do not fail if PLACEHOLDER_* markers are still present (for pre-upload dry runs).",
    )
    args = ap.parse_args()

    try:
        doc = json.loads(args.path.read_text())
    except Exception as exc:
        print(f"ERROR: cannot load {args.path}: {exc}", file=sys.stderr)
        sys.exit(2)

    errors = basic_check(doc)
    mlc_errors = try_mlcroissant(args.path)
    if mlc_errors is not None:
        errors.extend(mlc_errors)
    else:
        print("note: `pip install mlcroissant` for stricter official validation.", file=sys.stderr)

    if args.allow_placeholders:
        errors = [e for e in errors if "placeholder" not in e.lower()]

    if errors:
        print(f"FAIL: {len(errors)} issue(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"OK: 0 errors on {args.path}")


if __name__ == "__main__":
    main()
