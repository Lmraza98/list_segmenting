import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BUCKET_COLUMNS = [
    "bucket_key",
    "standardized_title",
    "variants",
    "seen_count",
    "last_seen_utc",
]
TITLES_FILENAME = "titles.csv"
ALIASES_FILENAME = "aliases.csv"
AUDIT_FILENAME = "title_audit.csv"
ALIASES_COLUMNS = ["variant", "standardized_title", "seen_count", "last_seen_utc"]
AUDIT_COLUMNS = ["issue_type", "variant", "current_bucket", "suggested_bucket", "matched_rules"]

CONFIG_DIRNAME = "config"
TITLE_RULES_FILENAME = "title_rules.json"
TITLE_FALLBACK_RULES_FILENAME = "title_fallback_rules.json"


def config_dir() -> Path:
    return Path(__file__).resolve().parent / CONFIG_DIRNAME


def load_label_pattern_rules(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rules: List[Tuple[str, str]] = []
    for item in payload:
        label = (item.get("label") or "").strip()
        pattern = (item.get("pattern") or "").strip()
        if label and pattern:
            rules.append((label, pattern))
    if not rules:
        raise ValueError(f"No valid rules loaded from {path}")
    return rules


RULES = load_label_pattern_rules(config_dir() / TITLE_RULES_FILENAME)
FALLBACK_RULES = load_label_pattern_rules(config_dir() / TITLE_FALLBACK_RULES_FILENAME)
COMPILED_RULES = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in RULES]
COMPILED_FALLBACK_RULES = [
    (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in FALLBACK_RULES
]


def pick_first(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row.get(key):
            return (row.get(key) or "").strip()
    return ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_titles_path() -> Path:
    # Keep the bucket store next to this script so users never have to pass it in.
    return Path(__file__).resolve().parent / TITLES_FILENAME


def default_aliases_path() -> Path:
    return Path(__file__).resolve().parent / ALIASES_FILENAME


def default_audit_path() -> Path:
    return Path(__file__).resolve().parent / AUDIT_FILENAME


def detect_csv_dialect(csv_path: Path) -> csv.Dialect:
    sample_size = 8192
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(sample_size)
    first_line = sample.splitlines()[0] if sample else ""
    if first_line.count("\t") > first_line.count(","):
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.get_dialect("excel")


def normalize_title(title: str) -> str:
    text = (title or "").strip().lower()
    text = re.sub(r"[\|/,_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9&+ ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_title_column(headers: List[str]) -> Optional[str]:
    if not headers:
        return None

    preferred = [
        "Title",
        "title",
        "Job Title",
        "job_title",
        "job title",
        "Position",
        "position",
    ]
    header_lookup = {h.lower(): h for h in headers}
    for candidate in preferred:
        if candidate.lower() in header_lookup:
            return header_lookup[candidate.lower()]

    for header in headers:
        if "title" in header.lower():
            return header
    return None


def parse_variants(variants: str) -> List[str]:
    if not variants:
        return []
    parts = re.split(r"[|;,]+", variants)
    return [x.strip() for x in parts if x.strip()]


def stringify_variants(variants: List[str]) -> str:
    seen = set()
    ordered = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return "|".join(ordered)


@dataclass
class Bucket:
    bucket_key: str
    standardized_title: str
    variants: List[str]
    seen_count: int
    last_seen_utc: str

    def to_row(self) -> Dict[str, str]:
        return {
            "bucket_key": self.bucket_key,
            "standardized_title": self.standardized_title,
            "variants": stringify_variants(self.variants),
            "seen_count": str(self.seen_count),
            "last_seen_utc": self.last_seen_utc,
        }


@dataclass
class AliasEntry:
    variant: str
    standardized_title: str
    seen_count: int
    last_seen_utc: str

    def to_row(self) -> Dict[str, str]:
        return {
            "variant": self.variant,
            "standardized_title": self.standardized_title,
            "seen_count": str(self.seen_count),
            "last_seen_utc": self.last_seen_utc,
        }


def load_buckets(titles_path: Path) -> Dict[str, Bucket]:
    if not titles_path.exists() or titles_path.stat().st_size == 0:
        return {}

    buckets: Dict[str, Bucket] = {}
    with titles_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        for row in reader:
            bucket_key_raw = pick_first(
                row,
                ["bucket_key", "bucket", "title_bucket", "normalized_title", "key"],
            )
            bucket_key = normalize_title(bucket_key_raw)
            standardized_title = pick_first(
                row,
                [
                    "standardized_title",
                    "standardized_titles",
                    "standard_title",
                    "canonical_title",
                ],
            )

            variants_raw = pick_first(row, ["variants", "variant_titles", "raw_titles"])
            variants = [normalize_title(v) for v in parse_variants(variants_raw)]
            variants = [v for v in variants if v]
            seen_count_raw = pick_first(row, ["seen_count", "count", "frequency"]) or "0"
            seen_count = int(seen_count_raw) if seen_count_raw.isdigit() else 0
            last_seen = pick_first(row, ["last_seen_utc", "last_seen", "updated_at"]) or utc_now_iso()

            if not bucket_key:
                continue

            if bucket_key not in variants:
                variants.append(bucket_key)

            buckets[bucket_key] = Bucket(
                bucket_key=bucket_key,
                standardized_title=standardized_title or bucket_key,
                variants=variants,
                seen_count=seen_count,
                last_seen_utc=last_seen,
            )
    return buckets


def load_aliases(aliases_path: Path) -> Dict[str, AliasEntry]:
    if not aliases_path.exists() or aliases_path.stat().st_size == 0:
        return {}
    aliases: Dict[str, AliasEntry] = {}
    with aliases_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        for row in reader:
            variant = normalize_title(pick_first(row, ["variant", "title", "raw_title"]))
            standardized = pick_first(
                row, ["standardized_title", "standard_title", "bucket", "mapped_title"]
            )
            if not variant or not standardized:
                continue
            seen_raw = pick_first(row, ["seen_count", "count", "frequency"]) or "0"
            last_seen = pick_first(row, ["last_seen_utc", "last_seen", "updated_at"]) or utc_now_iso()
            aliases[variant] = AliasEntry(
                variant=variant,
                standardized_title=standardized,
                seen_count=int(seen_raw) if seen_raw.isdigit() else 0,
                last_seen_utc=last_seen,
            )
    return aliases


def write_aliases(aliases_path: Path, aliases: Dict[str, AliasEntry]) -> None:
    with aliases_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALIASES_COLUMNS)
        writer.writeheader()
        for key in sorted(aliases.keys()):
            writer.writerow(aliases[key].to_row())


def write_buckets(titles_path: Path, buckets: Dict[str, Bucket]) -> None:
    with titles_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BUCKET_COLUMNS)
        writer.writeheader()
        for key in sorted(buckets):
            writer.writerow(buckets[key].to_row())


def best_bucket_match(
    normalized: str,
    bucket_keys: List[str],
    similarity_threshold: float,
) -> Optional[str]:
    if not normalized or not bucket_keys:
        return None

    best_key = None
    best_ratio = 0.0
    for key in bucket_keys:
        ratio = SequenceMatcher(None, normalized, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key

    if best_key and best_ratio >= similarity_threshold:
        return best_key
    return None


def choose_initial_standardized_title(raw_title: str, normalized: str) -> str:
    cleaned = (raw_title or "").strip()
    if cleaned:
        return cleaned
    return normalized


def canonicalize_bucket_variants(
    buckets: Dict[str, Bucket],
    variants_index: Dict[str, str],
    allowed_bucket_keys: set[str],
) -> None:
    for key in list(buckets.keys()):
        if key not in allowed_bucket_keys:
            del buckets[key]

    for key, bucket in buckets.items():
        cleaned: List[str] = []
        for variant in bucket.variants:
            if variants_index.get(variant) != key:
                continue
            if variant in cleaned:
                continue
            cleaned.append(variant)
        if key not in cleaned:
            cleaned.insert(0, key)
        bucket.variants = cleaned


def remap_all_variants_to_rules(buckets: Dict[str, Bucket]) -> Dict[str, Bucket]:
    remapped: Dict[str, Bucket] = {}
    for label, _ in RULES:
        key = normalize_title(label)
        existing = buckets.get(key)
        remapped[key] = Bucket(
            bucket_key=key,
            standardized_title=label,
            variants=[key],
            seen_count=(existing.seen_count if existing else 0),
            last_seen_utc=(existing.last_seen_utc if existing else utc_now_iso()),
        )

    all_variants = set()
    for key, bucket in buckets.items():
        all_variants.add(key)
        for variant in bucket.variants:
            if variant:
                all_variants.add(variant)

    for variant in all_variants:
        if not variant:
            continue
        target_label = match_rule_title(variant) or fallback_bucket_title(variant)
        target_key = normalize_title(target_label)
        bucket = remapped[target_key]
        if variant not in bucket.variants:
            bucket.variants.append(variant)
    return remapped


def rule_matches(raw_title: str) -> List[str]:
    title = (raw_title or "").strip().lower()
    if not title:
        return []
    matches = []
    for label, pattern in COMPILED_RULES:
        if pattern.search(title):
            matches.append(label)
    return matches


def match_rule_title(raw_title: str) -> Optional[str]:
    title = (raw_title or "").strip().lower()
    if not title:
        return None
    for label, pattern in COMPILED_RULES:
        if pattern.search(title):
            return label
    return None


def fallback_bucket_title(raw_title: str) -> str:
    t = (raw_title or "").lower()
    for label, pattern in COMPILED_FALLBACK_RULES:
        if pattern.search(t):
            return label
    return "President"


def add_variant_mapping(
    variants_index: Dict[str, str],
    variant: str,
    bucket_key: str,
    conflict_mode: str = "error",
) -> None:
    existing = variants_index.get(variant)
    if existing and existing != bucket_key:
        if conflict_mode == "keep_existing":
            return
        if conflict_mode == "prefer_new":
            variants_index[variant] = bucket_key
            return
        raise ValueError(
            f"Variant conflict: '{variant}' is mapped to both '{existing}' and '{bucket_key}'. "
            "A variant can only belong to one bucket."
        )
    variants_index[variant] = bucket_key


def write_audit_report(
    audit_path: Path,
    buckets: Dict[str, Bucket],
) -> None:
    rows: List[Dict[str, str]] = []
    for bucket in buckets.values():
        for variant in bucket.variants:
            matches = rule_matches(variant)
            if not matches:
                rows.append(
                    {
                        "issue_type": "unmatched",
                        "variant": variant,
                        "current_bucket": bucket.standardized_title,
                        "suggested_bucket": fallback_bucket_title(variant),
                        "matched_rules": "",
                    }
                )
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["issue_type"], r["variant"])):
            writer.writerow(row)


def standardize_titles(
    leads_path: Path,
    titles_path: Path,
    aliases_path: Path,
    audit_path: Path,
    output_path: Optional[Path],
    title_column: Optional[str],
    similarity_threshold: float,
) -> Tuple[int, int, str, Path, Path, Path]:
    buckets = remap_all_variants_to_rules(load_buckets(titles_path))
    aliases = load_aliases(aliases_path)
    allowed_bucket_keys = {normalize_title(label) for label, _ in RULES}
    variants_index: Dict[str, str] = {}
    # Pass 1: bucket key always maps to itself (authoritative).
    for key, _bucket in buckets.items():
        add_variant_mapping(variants_index, key, key, conflict_mode="prefer_new")
    # Pass 2: load all other variants, keeping the first established mapping.
    for key, bucket in buckets.items():
        for variant in bucket.variants:
            if variant == key:
                continue
            add_variant_mapping(variants_index, variant, key, conflict_mode="keep_existing")

    leads_dialect = detect_csv_dialect(leads_path)
    with leads_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=leads_dialect, restkey="__extra__")
        rows = []
        for row in reader:
            if "__extra__" in row:
                del row["__extra__"]
            if None in row:
                del row[None]
            rows.append(row)
        headers = list(reader.fieldnames or [])

    if not headers:
        raise ValueError("Lead CSV has no headers.")
    if not rows and leads_path.stat().st_size > 100:
        raise ValueError(
            "Lead CSV parsed to zero rows. Aborting to prevent overwrite; "
            "check delimiter/format."
        )

    detected = title_column or detect_title_column(headers)
    if not detected:
        raise ValueError(
            "Could not detect a title column. Pass --title-column explicitly."
        )
    if detected not in headers:
        raise ValueError(f"Title column '{detected}' not found in lead CSV.")

    additions = 0
    now = utc_now_iso()
    for row in rows:
        raw_title = (row.get(detected) or "").strip()
        normalized = normalize_title(raw_title)

        if not normalized:
            row["Standardized Titles"] = ""
            continue

        existing_bucket_key = variants_index.get(normalized)
        alias_entry = aliases.get(normalized)
        rule_title = match_rule_title(raw_title)
        rule_bucket_key = normalize_title(rule_title) if rule_title else None

        bucket_key = existing_bucket_key
        if alias_entry:
            alias_bucket_key = normalize_title(alias_entry.standardized_title)
            if alias_bucket_key in allowed_bucket_keys:
                bucket_key = alias_bucket_key
        elif rule_bucket_key:
            if rule_bucket_key not in buckets:
                buckets[rule_bucket_key] = Bucket(
                    bucket_key=rule_bucket_key,
                    standardized_title=rule_title or rule_bucket_key,
                    variants=[rule_bucket_key],
                    seen_count=0,
                    last_seen_utc=now,
                )
            # Migrate legacy mappings to the current rule bucket.
            if existing_bucket_key and existing_bucket_key != rule_bucket_key:
                old_bucket = buckets.get(existing_bucket_key)
                if old_bucket and normalized in old_bucket.variants:
                    old_bucket.variants = [v for v in old_bucket.variants if v != normalized]
                add_variant_mapping(
                    variants_index,
                    normalized,
                    rule_bucket_key,
                    conflict_mode="prefer_new",
                )
                bucket_key = rule_bucket_key
            elif not bucket_key:
                bucket_key = rule_bucket_key
        elif not bucket_key:
            if normalized in buckets:
                bucket_key = normalized

            if not bucket_key:
                bucket_key = best_bucket_match(
                    normalized,
                    list(buckets.keys()),
                    similarity_threshold=similarity_threshold,
                )

        if not bucket_key or bucket_key not in allowed_bucket_keys:
            fallback_title = fallback_bucket_title(raw_title)
            bucket_key = normalize_title(fallback_title)
            if bucket_key not in buckets:
                buckets[bucket_key] = Bucket(
                    bucket_key=bucket_key,
                    standardized_title=fallback_title,
                    variants=[bucket_key],
                    seen_count=0,
                    last_seen_utc=now,
                )
                additions += 1

        bucket = buckets[bucket_key]
        if normalized not in bucket.variants:
            bucket.variants.append(normalized)
        if existing_bucket_key and existing_bucket_key != bucket_key:
            old_bucket = buckets.get(existing_bucket_key)
            if old_bucket and normalized in old_bucket.variants:
                old_bucket.variants = [v for v in old_bucket.variants if v != normalized]
            add_variant_mapping(
                variants_index,
                normalized,
                bucket_key,
                conflict_mode="prefer_new",
            )
        else:
            add_variant_mapping(variants_index, normalized, bucket_key)
        bucket.seen_count += 1
        bucket.last_seen_utc = now
        row["Standardized Titles"] = bucket.standardized_title
        alias = aliases.get(normalized)
        if not alias:
            aliases[normalized] = AliasEntry(
                variant=normalized,
                standardized_title=bucket.standardized_title,
                seen_count=1,
                last_seen_utc=now,
            )
        else:
            alias.standardized_title = bucket.standardized_title
            alias.seen_count += 1
            alias.last_seen_utc = now

    output_target = output_path or leads_path
    output_headers = headers.copy()
    if "Standardized Titles" not in output_headers:
        output_headers.append("Standardized Titles")

    with output_target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_headers,
            delimiter=leads_dialect.delimiter,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    canonicalize_bucket_variants(buckets, variants_index, allowed_bucket_keys)
    write_buckets(titles_path, buckets)
    write_aliases(aliases_path, aliases)
    write_audit_report(audit_path, buckets)

    return len(rows), additions, detected, output_target, aliases_path, audit_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize lead titles using a persistent bucket map in titles.csv, "
            "and add a 'Standardized Titles' column to the lead CSV."
        )
    )
    parser.add_argument(
        "--leads",
        default="leads_example.csv",
        help="Path to lead list CSV (default: leads_example.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for updated lead CSV. Defaults to overwrite --leads.",
    )
    parser.add_argument(
        "--title-column",
        default=None,
        help="Explicit title column name in lead CSV. If omitted, script auto-detects.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Fuzzy match threshold for assigning unknown titles to existing buckets.",
    )
    parser.add_argument(
        "--titles-path",
        default=None,
        help="Optional explicit path for persistent titles bucket CSV.",
    )
    parser.add_argument(
        "--aliases-path",
        default=None,
        help="Optional explicit path for persistent aliases CSV.",
    )
    parser.add_argument(
        "--audit-path",
        default=None,
        help="Optional explicit path for audit CSV output.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    leads_path = Path(args.leads)
    titles_path = Path(args.titles_path) if args.titles_path else default_titles_path()
    aliases_path = Path(args.aliases_path) if args.aliases_path else default_aliases_path()
    audit_path = Path(args.audit_path) if args.audit_path else default_audit_path()
    output_path = Path(args.output) if args.output else None

    row_count, new_buckets, title_col, output_target, aliases_out, audit_out = standardize_titles(
        leads_path=leads_path,
        titles_path=titles_path,
        aliases_path=aliases_path,
        audit_path=audit_path,
        output_path=output_path,
        title_column=args.title_column,
        similarity_threshold=args.similarity_threshold,
    )
    print(f"Processed rows: {row_count}")
    print(f"Detected title column: {title_col}")
    print(f"New buckets added: {new_buckets}")
    print(f"Lead CSV written: {output_target}")
    print(f"Titles bucket CSV written: {titles_path}")
    print(f"Aliases CSV written: {aliases_out}")
    print(f"Audit CSV written: {audit_out}")


if __name__ == "__main__":
    main()
