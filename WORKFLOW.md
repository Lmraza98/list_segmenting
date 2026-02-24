# Lead Pipeline Workflow

## Folder Structure

- `data/incoming/`
  - Drop new lead CSV files here.
- `samples/`
  - Optional reference/example files. Not processed automatically.
- `data/state/`
  - Persistent mapping memory across runs:
  - `titles.csv`
  - `aliases.csv`
  - `industry_aliases.csv`
- `data/output/standardized/`
  - Standardized lead outputs (`*_standardized.csv`)
  - Per-file audits (`*_audit.csv`)
  - Batch manifests (`batch_manifest_*.csv`)
- `data/output/segmented/`
  - One folder per processed file, containing:
  - `segments/*.csv`
  - `segment_manifest.csv`
  - `holdout_over_company_cap.csv`
- `data/archive/`
  - Source files moved here after successful processing.

## One Command Run

```powershell
python process_batch.py
```

Default behavior:
- Reads all `*.csv` in `data/incoming/`
- Standardizes titles using persistent state in `data/state/`
- Segments each standardized file
- Moves processed source file to `data/archive/`
- Writes output artifacts to `data/output/...`

## Common Options

```powershell
python process_batch.py --max-per-company 2 --title-column "Title"
```

- `--max-per-company`: max contacts from one company per segment.
- `--title-column`: force which column is treated as title.
- `--split-by-region`: optional extra split by geography. If omitted, segments are persona+industry only.

## Direct Script Use (Optional)

- Standardize only:
```powershell
python standardize.py --leads my.csv --output my_standardized.csv
```

- Segment only:
```powershell
python segment_leads.py --input my_standardized.csv --output-dir segmented/my_run --max-per-company 2
```
