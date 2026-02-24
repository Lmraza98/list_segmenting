# Lead List Tool (Simple Guide)

This tool turns a raw lead CSV into clean, campaign-ready segment files.

You only need to do 2 things:
1. Drop your CSV in `data/incoming/`
2. Run `python process_batch.py`

## Quick Start

1. Put your new lead file(s) in:
   - `data/incoming/`
2. Open PowerShell in this project folder.
3. Run:

```powershell
python process_batch.py
```

Done.

## What Happens Automatically

For each file in `data/incoming/`, the tool:

1. Standardizes job titles (example: many CEO variations -> `CEO`)
2. Standardizes industries into industry buckets
3. Creates campaign segments
4. Limits each segment to max 2 people from the same company
5. Moves extra people to a holdout file
6. Moves the original source file to `data/archive/`

## Where Files Go

- Input:
  - `data/incoming/`
- Saved mapping memory (reused every run):
  - `data/state/titles.csv`
  - `data/state/aliases.csv`
  - `data/state/industry_aliases.csv`
- Rule configuration files:
  - `config/title_rules.json`
  - `config/title_fallback_rules.json`
  - `config/industry_rules.json`
  - `config/persona_buckets.json`
  - `config/state_region_map.json`
- Standardized outputs:
  - `data/output/standardized/`
- Segmented outputs:
  - `data/output/segmented/<input_file_name>/`
- Archived source files:
  - `data/archive/`
- Optional reference files:
  - `samples/`

## Definitions (Plain English)

- `Standardized Titles`:
  - The cleaned title bucket used for campaigns (example: `VP Engineering`).

- `segment_manifest.csv`:
  - A summary table of all segment files for one input file.
  - One row = one segment CSV.

- `segment_key`:
  - The internal segment name used in the filename.

- `persona`:
  - The audience type (example: `Executive`, `VP`, `OwnerFounder`).

- `industry`:
  - The standardized industry bucket used for segmenting.

- `region`:
  - Geography bucket used by the segment (if region splitting is enabled).

- `lead_count`:
  - Number of lead rows in that segment file.

- `company_count`:
  - Number of unique companies in that segment file.

- `size_mix`:
  - Breakdown of company sizes inside that segment.

- `holdout_over_company_cap.csv`:
  - Leads not included in primary segments because the company cap was reached.
  - Cap rule: max 2 people from the same company per segment.

## Rules Are Now in JSON (No Code Edits Needed)

The logic is controlled by files in `config/`:

- `config/title_rules.json`
  - Main title mapping rules (example: many CEO title variations -> `CEO`)
- `config/title_fallback_rules.json`
  - Backup title mapping when no main title rule matches
- `config/industry_rules.json`
  - Industry bucket rules
- `config/persona_buckets.json`
  - Which standardized titles belong to each persona group (`Executive`, `VP`, etc.)
- `config/state_region_map.json`
  - State -> region mapping used in segmentation

### Important

- You can change behavior by updating these JSON files.
- Keep labels consistent (same spelling/case across files) so segments stay stable.
- If you are unsure, ask before editing rules directly.

## How To Use `segment_manifest.csv`

Open `segment_manifest.csv` first.

Use it to decide:
1. Which segments are large enough to send now (`lead_count`)
2. Which segments fit your campaign audience (`persona`, `industry`)
3. Which exact file to send (`file`)

## Important Check: No Lost Leads

To confirm all leads are accounted for:

`total input leads = all segment files + holdout file`

## Common Questions

### Can I process multiple files at once?

Yes. Put multiple CSVs in `data/incoming/` and run once.

### Does it get smarter over time?

Yes. Mapping files in `data/state/` are reused each run.

### What if something looks wrong?

1. Check `data/output/standardized/*_audit.csv`
2. Share examples that look wrong
3. Re-run after mapping updates
