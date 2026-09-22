# Artifacts and SQL

Schema version 1 is stored in `PRAGMA user_version`; unsupported versions fail
rather than silently rewrite existing tables. A future migration must explicitly
handle the old schema. Inserts for one experiment are transactional. Run IDs are
unique; timestamps and IDs change between runs, while seeded statistical results
and hashes of identical inputs/configuration are reproducible.

The manifest records normalized input hashes and archived CSVs, membership hash,
configuration hash, Python module hashes, git revision/dirty state, dependency
versions, command, tickers, date range, source/adjustment policy and timestamps.
CSV ingestion additionally records the original price-file hash. The immutable
configuration keeps the evaluation boundary independent of later data availability.
Eligibility is stored per ticker/date, including observations, membership, signal
and eligibility; missing coverage is retained for names that never trade.

Tables: `experiments`, `eligibility`, `holdings`, `asset_returns`, `monthly_returns`,
`costs`, `results`. The key joins align formation-date eligibility with return-date
asset and portfolio returns. `attribution.csv` demonstrates the join and retains
NULL returns. Portfolio costs repeat on each asset row: **do not sum that column**.

```python
from store import connect, comparison, coverage, attribution

with connect("outputs/experiments.sqlite") as db:
    run_id = db.execute(
        "SELECT run_id FROM experiments ORDER BY rowid DESC LIMIT 1"
    ).fetchone()[0]
    print(comparison(db, run_id, "2023-01-31", 10))
    print(coverage(db, run_id))
    print(attribution(db, run_id, "long_short", 10).head())
```

All user-supplied query values use placeholders, including cost and date filters.
Comparisons return NULL annual means if even one portfolio month is missing.
`evaluation.csv` records sample/observed counts, turnover, mean intervals,
p-values, blocks, costs and inference status. Short/no-variation samples remain
in the output. `comparison.csv` is the compact 10-bps SQL summary.

Each return artifact has an adjacent metadata JSON containing its hash, frequency,
return convention, mode and cost assumption. To prepare factors from an authorized
provider export (convert percent units to decimals first):

```python
import hashlib, json
from pathlib import Path

p = Path("/path/monthly_factors.csv")
p.with_suffix(".json").write_text(
    json.dumps(
        {
            "frequency": "monthly",
            "units": "decimal",
            "source": "Describe the actual provider, dataset, retrieval time and transformations",
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
```

This metadata is a declaration, not proof of data quality. Exact monthly date
alignment is enforced after normalization to month-end. No inner join may hide
missing strategy months. Factor regression is conditional model inference; trying
many factor models and reporting only the best would require a separate testing policy.
