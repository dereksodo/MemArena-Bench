#!/usr/bin/env python3
"""Merge host-generated Croissant metadata with our RAI skeleton.

Input:
  --autogen   The JSON exported by the dataset host. For Hugging Face this can
              be fetched from https://huggingface.co/api/datasets/<repo>/croissant.
              It should contain the hosted URL and file distribution metadata.
  --rai       docs/memarena-croissant.json
              Our hand-written skeleton with RAI fields, fallback RecordSets, creators,
              keywords, etc.
  --out       Destination JSON — this is what you upload to OpenReview.

Merge policy:
  • Top-level `url` → from autogen.
  • `distribution` for files we know → take contentUrl + sha256 + contentSize
    from autogen (matched by `name`), but keep our richer `description` +
    `containedIn` + `encodingFormat` if ours is more specific.
  • `distribution` items only in autogen (readme, license) → append.
  • `recordSet` → use the RAI skeleton's validated fallback schemas. Some
    hosted Croissant generators emit nested schemas that are useful for the
    viewer but fail stricter mlcroissant validation.
  • Everything else (RAI, keywords, creator, citeAs) → from RAI.

Usage:
  python scripts/patch_croissant_rai.py \
    --autogen out/hf-autogen-croissant.json \
    --rai     docs/memarena-croissant.json \
    --out     out/memarena-croissant-v1.0.0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.parse import urlparse


def index_by_name(items: list[dict]) -> dict[str, dict]:
    return {it.get("name") or it.get("@id") or "": it for it in items if isinstance(it, dict)}


def merge_file_object(rai_file: dict, autogen_file: dict) -> dict:
    """Start from RAI skeleton, overlay authoritative host-provided fields."""
    out = dict(rai_file)
    for key in ("contentUrl", "sha256", "md5", "contentSize"):
        if key in autogen_file:
            out[key] = autogen_file[key]
    # If RAI did not set encodingFormat, prefer the host's guess.
    if "encodingFormat" not in out and "encodingFormat" in autogen_file:
        out["encodingFormat"] = autogen_file["encodingFormat"]
    return out


def _is_relative_url(value: str) -> bool:
    parsed = urlparse(value)
    return not parsed.scheme and not value.startswith("/")


def absolutize_content_urls(items: list[dict], dataset_url: str) -> list[dict]:
    """Make FileObject contentUrl values standalone-downloadable.

    OpenReview/NeurIPS checkers validate the Croissant JSON as an uploaded file,
    not as a file sitting inside the Hugging Face repo. Relative contentUrl
    values are valid in-repo, but record generation has no folder context and
    cannot resolve them. Convert those paths to HF resolve URLs.
    """
    if not dataset_url.startswith("https://huggingface.co/datasets/"):
        return items

    out: list[dict] = []
    base = dataset_url.rstrip("/")
    for item in items:
        item = dict(item)
        content_url = item.get("contentUrl")
        if isinstance(content_url, str) and _is_relative_url(content_url):
            escaped = quote(content_url, safe="/")
            item["contentUrl"] = f"{base}/resolve/main/{escaped}"
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autogen", type=Path, required=True)
    ap.add_argument("--rai", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    autogen = json.loads(args.autogen.read_text())
    rai = json.loads(args.rai.read_text())

    merged = dict(rai)

    # 1. Top-level dataset URL / identifier from the hosting platform.
    if autogen.get("url"):
        merged["url"] = autogen["url"]
    if autogen.get("identifier"):
        merged["identifier"] = autogen["identifier"]
    if autogen.get("datePublished"):
        merged["datePublished"] = autogen["datePublished"]

    # 2. Distribution merge.
    rai_dist = rai.get("distribution") or []
    auto_dist = autogen.get("distribution") or []
    auto_by_name = index_by_name(auto_dist)

    new_dist: list[dict] = []
    matched_auto_names: set[str] = set()
    for rf in rai_dist:
        name = rf.get("name") or rf.get("@id") or ""
        if name in auto_by_name:
            new_dist.append(merge_file_object(rf, auto_by_name[name]))
            matched_auto_names.add(name)
        else:
            # Keep placeholder RAI entry — user will notice via validator.
            new_dist.append(rf)

    # Append autogen-only entries, e.g. README or LICENSE files exposed by the host.
    for name, af in auto_by_name.items():
        if name not in matched_auto_names:
            extra = dict(af)
            if not extra.get("name") and (extra.get("@id") or name):
                extra["name"] = extra.get("@id") or name
            new_dist.append(extra)

    merged["distribution"] = new_dist
    if merged.get("url"):
        merged["distribution"] = absolutize_content_urls(new_dist, merged["url"])

    # 3. Structure schemas. Use the validated RAI fallback schemas. HF can
    # emit nested recordSets with arrayShape/isArray inconsistencies, so do not
    # let hosted schemas make the OpenReview Croissant invalid.
    rai_record_sets = rai.get("recordSet") or []
    if rai_record_sets:
        merged["recordSet"] = rai_record_sets

    # 4. Safety: make sure RAI namespaces survived.
    ctx = merged.get("@context") or {}
    assert ctx.get("rai") == "http://mlcommons.org/croissant/RAI/", \
        "RAI namespace lost during merge"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(
        f"wrote {args.out} "
        f"({len(new_dist)} distribution entries, "
        f"{len(merged.get('recordSet') or [])} record sets)"
    )

    # Quick sniff report.
    still_placeholders = [
        it for it in new_dist
        if str(it.get("contentUrl", "")).startswith("PLACEHOLDER")
        or it.get("sha256") == "PLACEHOLDER_SHA256"
    ]
    if still_placeholders:
        print(
            f"warning: {len(still_placeholders)} distribution entries still have "
            f"placeholders — names not found in host-generated Croissant:",
            file=sys.stderr,
        )
        for it in still_placeholders:
            print(f"  - {it.get('name')}", file=sys.stderr)


if __name__ == "__main__":
    main()
