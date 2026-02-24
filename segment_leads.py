import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


CONFIG_DIRNAME = "config"
PERSONA_BUCKETS_FILENAME = "persona_buckets.json"
INDUSTRY_RULES_FILENAME = "industry_rules.json"
STATE_REGION_MAP_FILENAME = "state_region_map.json"

INDUSTRY_ALIASES_COLUMNS = ["normalized_industry", "industry_bucket", "seen_count", "last_seen_utc"]
INDUSTRY_AUDIT_COLUMNS = ["issue_type", "raw_industry", "normalized_industry", "suggested_bucket"]


def config_dir() -> Path:
    return Path(__file__).resolve().parent / CONFIG_DIRNAME


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_label_pattern_rules(path: Path) -> List[Tuple[str, str]]:
    payload = load_json(path)
    rules: List[Tuple[str, str]] = []
    for item in payload:
        label = (item.get("label") or "").strip()
        pattern = (item.get("pattern") or "").strip()
        if label and pattern:
            rules.append((label, pattern))
    if not rules:
        raise ValueError(f"No valid rules loaded from {path}")
    return rules


PERSONA_BUCKETS = load_json(config_dir() / PERSONA_BUCKETS_FILENAME)
EXEC_TITLES = set(PERSONA_BUCKETS.get("Executive", []))
VP_TITLES = set(PERSONA_BUCKETS.get("VP", []))
OWNER_FOUNDER_TITLES = set(PERSONA_BUCKETS.get("OwnerFounder", []))
PRODUCT_OPS_TITLES = set(PERSONA_BUCKETS.get("ProductOps", []))
INDUSTRY_RULES = load_label_pattern_rules(config_dir() / INDUSTRY_RULES_FILENAME)
STATE_TO_REGION = load_json(config_dir() / STATE_REGION_MAP_FILENAME)


@dataclass
class IndustryAlias:
    normalized_industry: str
    industry_bucket: str
    seen_count: int
    last_seen_utc: str

    def to_row(self) -> Dict[str, str]:
        return {
            "normalized_industry": self.normalized_industry,
            "industry_bucket": self.industry_bucket,
            "seen_count": str(self.seen_count),
            "last_seen_utc": self.last_seen_utc,
        }


def detect_csv_dialect(path: Path) -> csv.Dialect:
    sample_size = 8192
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(sample_size)
    first = sample.splitlines()[0] if sample else ""
    if first.count("\t") > first.count(","):
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.get_dialect("excel")


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def classify_persona(standardized_title: str) -> str:
    title = (standardized_title or "").strip()
    if title in EXEC_TITLES:
        return "Executive"
    if title in VP_TITLES:
        return "VP"
    if title in OWNER_FOUNDER_TITLES:
        return "OwnerFounder"
    if title in PRODUCT_OPS_TITLES:
        return "ProductOps"
    return "Other"


def classify_region(state: str, country: str) -> str:
    state_norm = normalize_text(state)
    country_norm = normalize_text(country)
    if country_norm and country_norm not in {"us", "usa", "united states"}:
        return "International"
    if state_norm in STATE_TO_REGION:
        return STATE_TO_REGION[state_norm]
    return "UnknownRegion"


def classify_size_band(employees_raw: str) -> str:
    text = re.sub(r"[^0-9]", "", employees_raw or "")
    if not text:
        return "UnknownSize"
    n = int(text)
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 200:
        return "51-200"
    if n <= 1000:
        return "201-1000"
    return "1001+"


def clean_industry(industry: str) -> str:
    text = (industry or "").strip()
    return text if text else "UnknownIndustry"


def build_segment_key(persona: str, industry: str, region: str) -> str:
    return f"{slugify(persona)}__{slugify(industry)}__{slugify(region)}"


def build_segment_key_no_region(persona: str, industry: str) -> str:
    return f"{slugify(persona)}__{slugify(industry)}"


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str], csv.Dialect]:
    dialect = detect_csv_dialect(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect, restkey="__extra__")
        rows: List[Dict[str, str]] = []
        for row in reader:
            if "__extra__" in row:
                del row["__extra__"]
            if None in row:
                del row[None]
            rows.append(row)
        headers = list(reader.fieldnames or [])
    return rows, headers, dialect


def write_rows(path: Path, headers: List[str], rows: List[Dict[str, str]], delimiter: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            delimiter=delimiter,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_industry_aliases(path: Path) -> Dict[str, IndustryAlias]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    aliases: Dict[str, IndustryAlias] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_text(row.get("normalized_industry", ""))
            bucket = (row.get("industry_bucket") or "").strip()
            seen_raw = (row.get("seen_count") or "0").strip()
            if not key or not bucket:
                continue
            aliases[key] = IndustryAlias(
                normalized_industry=key,
                industry_bucket=bucket,
                seen_count=int(seen_raw) if seen_raw.isdigit() else 0,
                last_seen_utc=(row.get("last_seen_utc") or "").strip() or utc_now_iso(),
            )
    return aliases


def write_industry_aliases(path: Path, aliases: Dict[str, IndustryAlias]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDUSTRY_ALIASES_COLUMNS)
        writer.writeheader()
        for key in sorted(aliases.keys()):
            writer.writerow(aliases[key].to_row())


def classify_industry_bucket(raw_industry: str, aliases: Dict[str, IndustryAlias]) -> Tuple[str, bool]:
    cleaned = clean_industry(raw_industry)
    normalized = normalize_text(cleaned)
    alias = aliases.get(normalized)
    if alias:
        return alias.industry_bucket, True
    for bucket, pattern in INDUSTRY_RULES:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return bucket, True
    return "Other Industrial/Electrical", False


def write_industry_audit(
    path: Path,
    unmatched_rows: List[Dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDUSTRY_AUDIT_COLUMNS)
        writer.writeheader()
        for row in unmatched_rows:
            writer.writerow(row)


def segment_leads(
    input_path: Path,
    output_dir: Path,
    max_per_company: int,
    industry_aliases_path: Path | None = None,
    industry_audit_path: Path | None = None,
    split_by_region: bool = False,
) -> Tuple[int, int, int]:
    rows, headers, dialect = read_rows(input_path)
    if not headers:
        raise ValueError("Input CSV has no headers.")
    if "Standardized Titles" not in headers:
        raise ValueError("Input CSV is missing 'Standardized Titles'. Run standardize.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    if industry_aliases_path is None:
        industry_aliases_path = output_dir / "industry_aliases.csv"
    if industry_audit_path is None:
        industry_audit_path = output_dir / "industry_audit.csv"

    aliases = load_industry_aliases(industry_aliases_path)
    unmatched_industries: List[Dict[str, str]] = []

    enriched_rows: List[Dict[str, str]] = []
    now = utc_now_iso()
    for row in rows:
        persona = classify_persona(row.get("Standardized Titles", ""))
        raw_industry = row.get("Industry", "")
        industry_bucket, matched = classify_industry_bucket(raw_industry, aliases)
        normalized_industry = normalize_text(clean_industry(raw_industry))
        if normalized_industry:
            alias = aliases.get(normalized_industry)
            if not alias:
                aliases[normalized_industry] = IndustryAlias(
                    normalized_industry=normalized_industry,
                    industry_bucket=industry_bucket,
                    seen_count=1,
                    last_seen_utc=now,
                )
            else:
                alias.industry_bucket = industry_bucket
                alias.seen_count += 1
                alias.last_seen_utc = now
        if not matched and normalized_industry:
            unmatched_industries.append(
                {
                    "issue_type": "unmatched_industry",
                    "raw_industry": raw_industry or "",
                    "normalized_industry": normalized_industry,
                    "suggested_bucket": industry_bucket,
                }
            )
        region = classify_region(row.get("Company State", ""), row.get("Company Country", ""))
        size_band = classify_size_band(row.get("# Employees", ""))
        if split_by_region:
            segment_key = build_segment_key(persona, industry_bucket, region)
        else:
            segment_key = build_segment_key_no_region(persona, industry_bucket)
        row["Campaign Persona"] = persona
        row["Industry Bucket"] = industry_bucket
        row["Region"] = region
        row["Company Size Band"] = size_band
        row["Segment Key"] = segment_key
        enriched_rows.append(row)

    company_counts: Dict[str, Counter] = defaultdict(Counter)
    accepted_by_segment: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    holdout_rows: List[Dict[str, str]] = []

    for row in enriched_rows:
        company = normalize_text(row.get("Company", ""))
        if not company:
            company = "__unknown_company__"
        segment_key = row["Segment Key"]
        if company_counts[segment_key][company] >= max_per_company:
            holdout = dict(row)
            holdout["Holdout Reason"] = f"company_cap_exceeded_{max_per_company}"
            holdout_rows.append(holdout)
            continue
        company_counts[segment_key][company] += 1
        accepted_by_segment[segment_key].append(row)

    extra_cols = [
        "Campaign Persona",
        "Industry Bucket",
        "Region",
        "Company Size Band",
        "Segment Key",
    ]
    out_headers = headers + [c for c in extra_cols if c not in headers]
    holdout_headers = out_headers + ["Holdout Reason"]

    manifest_rows: List[Dict[str, str]] = []
    for segment_key, segment_rows in sorted(accepted_by_segment.items()):
        segment_path = segments_dir / f"{segment_key}.csv"
        write_rows(segment_path, out_headers, segment_rows, dialect.delimiter)
        unique_companies = len({normalize_text(r.get("Company", "")) for r in segment_rows})
        persona = segment_rows[0]["Campaign Persona"] if segment_rows else ""
        industry = segment_rows[0]["Industry Bucket"] if segment_rows else ""
        region = segment_rows[0]["Region"] if segment_rows else ""
        size_mix = Counter(r["Company Size Band"] for r in segment_rows)
        manifest_rows.append(
            {
                "segment_key": segment_key,
                "persona": persona,
                "industry": industry,
                "region": region,
                "lead_count": str(len(segment_rows)),
                "company_count": str(unique_companies),
                "size_mix": "|".join(f"{k}:{v}" for k, v in sorted(size_mix.items())),
                "file": str(segment_path.name),
            }
        )

    holdout_path = output_dir / "holdout_over_company_cap.csv"
    write_rows(holdout_path, holdout_headers, holdout_rows, dialect.delimiter)

    manifest_path = output_dir / "segment_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "segment_key",
            "persona",
            "industry",
            "region",
            "lead_count",
            "company_count",
            "size_mix",
            "file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    write_industry_aliases(industry_aliases_path, aliases)
    dedup_unmatched = {
        (r["normalized_industry"], r["suggested_bucket"]): r for r in unmatched_industries
    }
    write_industry_audit(industry_audit_path, list(dedup_unmatched.values()))

    return len(rows), len(manifest_rows), len(holdout_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment standardized leads into campaign-ready CSVs with per-company caps."
        )
    )
    parser.add_argument(
        "--input",
        default="leads.csv",
        help="Input leads CSV with Standardized Titles column (default: leads.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="segmented",
        help="Output folder for segment files (default: segmented)",
    )
    parser.add_argument(
        "--max-per-company",
        type=int,
        default=2,
        help="Maximum contacts per company per segment (default: 2)",
    )
    parser.add_argument(
        "--industry-aliases-path",
        default=None,
        help="Optional persistent industry alias mapping CSV path.",
    )
    parser.add_argument(
        "--industry-audit-path",
        default=None,
        help="Optional path to write unmatched industry audit CSV.",
    )
    parser.add_argument(
        "--split-by-region",
        action="store_true",
        help="Split segment files by region in addition to persona+industry.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    total_rows, segment_count, holdout_count = segment_leads(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        max_per_company=args.max_per_company,
        industry_aliases_path=Path(args.industry_aliases_path) if args.industry_aliases_path else None,
        industry_audit_path=Path(args.industry_audit_path) if args.industry_audit_path else None,
        split_by_region=args.split_by_region,
    )
    print(f"Input rows processed: {total_rows}")
    print(f"Segments created: {segment_count}")
    print(f"Holdout rows (company cap): {holdout_count}")
    print(f"Output folder: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
