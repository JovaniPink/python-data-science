# BLS Public Data API source record

_Verified August 16, 2026._

## Decision

This experiment retrieves two first-party U.S. Bureau of Labor Statistics
(BLS) series directly from the BLS Public Data API at runtime:

| Series ID | Observed measure | Repository-derived treatment |
|---|---|---|
| `CUUR0000SA0` | CPI-U, U.S. city average, all items, not seasonally adjusted, monthly | Convert the index to its 12-month percentage change |
| `LNS14000000` | Civilian unemployment rate, seasonally adjusted, monthly | Use the published rate as a descriptive feature |

The default sample is January 2006 through December 2025. The first 12 months
supply the within-sample CPI lag and do not become model observations. Data is
retrieved when the CLI or notebook runs; raw BLS data is not committed.

## Access and permitted use

- API endpoint: <https://api.bls.gov/publicAPI/v2/timeseries/data/>
- Developer documentation: <https://www.bls.gov/developers/>
- API signature: <https://www.bls.gov/developers/api_signature_v2.htm>
- API limits FAQ: <https://www.bls.gov/developers/api_FAQs.htm>
- Current terms: <https://www.bls.gov/developers/termsOfService.htm>
- CPI series-code explanation: <https://www.bls.gov/cpi/factsheets/cpi-series-ids.htm>
- October 2025 CPI notice: <https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm>
- October 2025 CPS notice: <https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm>

The BLS terms observed on August 16, 2026 state that data accessed through
BLS.gov should not include controls on end use. They require API users to cite
the retrieval date and clearly state:

> BLS.gov cannot vouch for the data or analyses derived from these data after
> the data have been retrieved from BLS.gov.

The terms prohibit falsely representing modified content as BLS content and
allow BLS to impose or enforce access limits. The implementation therefore:

- records a timezone-aware UTC retrieval timestamp;
- preserves source series IDs and the exact endpoint;
- retains unavailable monthly records and BLS footnotes rather than imputing;
- makes anonymous requests in at most 10-year inclusive windows;
- does not bypass request limits or require a registration key;
- labels transformations and model output as repository-derived; and
- does not use the BLS logo.

This is an engineering record of reviewed terms, not legal advice. Recheck the
terms before a materially different use.

## Measurement and analysis boundaries

- CPI-U measures average price change for an urban consumer population. BLS
  says that population covers over 90 percent of the U.S. population but
  excludes rural nonmetropolitan residents, farm households, military
  installations, religious communities, and institutions.
- CPI-U is not a complete cost-of-living measure and need not match an
  individual household's inflation experience.
- CPI is not seasonally adjusted; unemployment is. A 12-month CPI change
  reduces recurring seasonality but does not make the series methodologically
  identical.
- BLS series may be revised. A rerun records its timestamp and can legitimately
  produce different values.
- BLS marks October 2025 unavailable in both selected series. Official BLS CPI
  and Current Population Survey notices attribute the gap to the 2025 lapse in
  appropriations. The experiment excludes that month and does not impute it.
- K-means is fit to standardized inflation and unemployment. It is sensitive
  to years, features, cluster count, initialization, and algorithm.
- Monthly observations overlap in their 12-month CPI windows and are serially
  correlated. The experiment does not treat them as independent evidence.
- Silhouette measures geometric separation and ARI measures assignment
  stability across seeds; neither establishes economic meaning or causality.
- Output is ex-post description, not causal inference, a recession classifier,
  a forecast, a trading signal, or financial advice.

## Why FRED is not the model-training source

FRED is useful for discovery and reference, but its
[legal terms](https://fred.stlouisfed.org/legal/terms/) reviewed August 16,
2026 prohibit using FRED services or content in connection with developing or
training machine-learning systems. This repository does not retrieve the
selected series from FRED and does not train on FRED-delivered data.
