# Regional expert ensemble contract

Checked August 29, 2026.

This experiment predicts next-quarter state employment year-over-year log
growth in a point-in-time historical backtest. It covers the 50 states and
Washington, DC and excludes territories. It is not a causal, recession,
trading, or financial-advice system.

The shared contract is
`contracts/regional-expert-ensemble.v1.json`. Generated source bytes,
normalized observations, panels, folds, predictions, fitted state, and
manifests belong under ignored `data/` or `artifacts/` directories.

Run Python from the shared, hash-verified source bundle:

```bash
uv run --locked regional-expert-ensemble \
  --source-bundle data/regional/regional-source-bundle.v1.json \
  --output-dir artifacts/regional-ensemble/python/v1
```

Generate the ignored synthetic acceptance fixture:

```bash
uv run --locked python -m scripts.generate_regional_fixture \
  --output-dir artifacts/regional-ensemble/synthetic-source
```

On August 29, 2026, both implementations consumed the same synthetic
multi-vintage bundle spanning 2015 Q1 through 2025 Q4. The Elixir no-write
verifier reported `MATCH` for exact panel/fold bytes, deterministic predictions
within `1.0e-6`, exact stack weights, and neural structural invariants. This is
contract evidence, not a real-world performance claim.
