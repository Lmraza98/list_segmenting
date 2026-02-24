import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import segment_leads
import standardize


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_batch(
    incoming_dir: Path,
    state_dir: Path,
    standardized_dir: Path,
    segmented_dir: Path,
    archive_dir: Path,
    max_per_company: int,
    title_column: str | None,
    similarity_threshold: float,
    split_by_region: bool,
) -> Path:
    ensure_dirs([incoming_dir, state_dir, standardized_dir, segmented_dir, archive_dir])

    titles_path = state_dir / "titles.csv"
    aliases_path = state_dir / "aliases.csv"
    industry_aliases_path = state_dir / "industry_aliases.csv"

    input_files = sorted(p for p in incoming_dir.glob("*.csv") if p.is_file())
    if not input_files:
        raise ValueError(f"No CSV files found in incoming folder: {incoming_dir}")

    batch_manifest = standardized_dir / f"batch_manifest_{utc_stamp()}.csv"
    manifest_rows = []

    for input_file in input_files:
        stem = input_file.stem
        standardized_output = standardized_dir / f"{stem}_standardized.csv"
        audit_output = standardized_dir / f"{stem}_audit.csv"
        industry_audit_output = standardized_dir / f"{stem}_industry_audit.csv"
        segment_output_dir = segmented_dir / stem

        row_count, new_buckets, detected_col, output_target, aliases_out, audit_out = standardize.standardize_titles(
            leads_path=input_file,
            titles_path=titles_path,
            aliases_path=aliases_path,
            audit_path=audit_output,
            output_path=standardized_output,
            title_column=title_column,
            similarity_threshold=similarity_threshold,
        )

        total_rows, segment_count, holdout_count = segment_leads.segment_leads(
            input_path=output_target,
            output_dir=segment_output_dir,
            max_per_company=max_per_company,
            industry_aliases_path=industry_aliases_path,
            industry_audit_path=industry_audit_output,
            split_by_region=split_by_region,
        )

        archive_target = archive_dir / f"{utc_stamp()}__{input_file.name}"
        input_file.replace(archive_target)

        manifest_rows.append(
            {
                "source_file": str(input_file.name),
                "archived_source": str(archive_target),
                "standardized_output": str(output_target),
                "audit_output": str(audit_out),
                "segments_dir": str(segment_output_dir),
                "rows_processed": str(row_count),
                "rows_segmented": str(total_rows),
                "segments_created": str(segment_count),
                "holdout_rows": str(holdout_count),
                "new_buckets_added": str(new_buckets),
                "title_column": detected_col,
                "titles_state": str(titles_path),
                "aliases_state": str(aliases_out),
                "industry_aliases_state": str(industry_aliases_path),
                "industry_audit_output": str(industry_audit_output),
            }
        )

    with batch_manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file",
            "archived_source",
            "standardized_output",
            "audit_output",
            "segments_dir",
            "rows_processed",
            "rows_segmented",
            "segments_created",
            "holdout_rows",
            "new_buckets_added",
            "title_column",
            "titles_state",
            "aliases_state",
            "industry_aliases_state",
            "industry_audit_output",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    return batch_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: standardize + segment all CSVs dropped in an incoming folder."
    )
    parser.add_argument("--incoming-dir", default="data/incoming", help="Drop input CSVs here.")
    parser.add_argument("--state-dir", default="data/state", help="Persistent mapping files live here.")
    parser.add_argument("--standardized-dir", default="data/output/standardized", help="Standardized output CSVs.")
    parser.add_argument("--segmented-dir", default="data/output/segmented", help="Segment output root folder.")
    parser.add_argument("--archive-dir", default="data/archive", help="Processed source CSVs are moved here.")
    parser.add_argument("--max-per-company", type=int, default=2, help="Max contacts/company per segment.")
    parser.add_argument("--title-column", default=None, help="Optional explicit title column name.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Fuzzy bucket threshold in standardization.",
    )
    parser.add_argument(
        "--split-by-region",
        action="store_true",
        help="Split segments by region as an extra key dimension.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = run_batch(
        incoming_dir=Path(args.incoming_dir),
        state_dir=Path(args.state_dir),
        standardized_dir=Path(args.standardized_dir),
        segmented_dir=Path(args.segmented_dir),
        archive_dir=Path(args.archive_dir),
        max_per_company=args.max_per_company,
        title_column=args.title_column,
        similarity_threshold=args.similarity_threshold,
        split_by_region=args.split_by_region,
    )
    print(f"Batch completed. Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()