# Data, privacy, and third-party notice

## Public evidence boundary

This repository does not distribute raw reviewer records, reviewer names,
review text, product URLs, image URLs, raw McAuley user histories, or raw
Amazon product/review tables. It also omits the source-derived bootstrap CSV
catalogs used for full campaign replay.

Included JSONL, JSON, CSV, and NPY files under `evidence/campaigns/**/outputs/` are
generated scientific records, aggregate analysis tables, exact decomposition
records, or timing arrays. They are included to make the reported evidence
auditable. The `inputs/results/` files are frozen historical result artifacts,
not raw consumer records. The only file under an `inputs/data/processed/` path
is a hash and schema manifest; source catalog rows are absent.

## Obtaining source data

Researchers who repeat the full computation must obtain the relevant source
data independently and comply with the source provider's current access terms,
license, privacy requirements, and citation instructions. This repository does
not grant or imply any rights in third-party datasets.

## License scope

The MIT License applies to original repository code for which the authors hold
copyright. It does not relicense:

- third-party datasets or database contents;
- frozen third-party or historical provenance whose own terms control;
- journal templates, publisher material, or the manuscript;
- names, marks, or metadata owned by data providers; or
- any artifact for which the authors do not hold redistribution rights.

Generated evidence files are provided for scholarly inspection and result
verification. Users remain responsible for determining whether their planned
reuse requires additional permission or attribution.

## Privacy controls

The evidence staging source was screened for direct identifiers, local machine
paths, execution logs, raw user/item files, and manuscript/reviewer materials.
The corresponding reports are retained under `evidence/privacy/`. The top-level
`verify_evidence.py` also fails if forbidden raw-data filenames, project-state
directories, caches, or known absolute local paths appear in the committed
snapshot.
