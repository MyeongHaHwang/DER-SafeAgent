# External raw sources

This file lists every external corpus or feed mixed into the fine-tuning
dataset, with license, expected hash, and any redistribution restriction.

| Source | URL / DOI | License (SPDX) | Expected SHA-256 of release tarball | Redistributable? |
|--------|-----------|----------------|------------------------------------|------------------|
| MITRE ATT&CK for ICS (STIX bundle) | TBD | Apache-2.0 | TBD | yes |
| CISA ICS-CERT advisories (selected) | TBD | Public domain (US gov) | TBD | yes |
| HAI testbed dataset (subset) | TBD | TBD — license check pending | TBD | TBD |
| SWaT (Singapore) (subset) | TBD | restricted research use | TBD | no — link only |
| Internal synthetic from `code/simulation/` | n/a | this repo's license | (per-shard hash in manifest) | yes |

Rows that resolve to **Redistributable? = no** are referenced by URL only;
users must obtain them upstream and run `build_dataset.py` with their local
copy. The build script verifies hashes before ingesting.

## Rejected rows
See `REJECTED.md` for examples that failed schema validation. The list is
written by `build_dataset.py` and committed for transparency.
