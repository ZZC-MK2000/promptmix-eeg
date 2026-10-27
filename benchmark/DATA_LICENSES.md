# Data licenses and redistribution audit

**No raw EEG, audio, or model checkpoints are redistributed by AEB v0.1.** The
repository contains code, split indices, metadata, aggregate results, and derived figures.

| Dataset | License / status | Raw data in AEB? | Action |
|---|---|---|---|
| MUSIN-G (OpenNeuro ds003774) | CC0; verified from dataset_description.json | no | users download from OpenNeuro S3 |
| ds002725 (OpenNeuro) | CC0; verified from dataset_description.json | no | users download from OpenNeuro |
| SparrKULee | CC BY-NC 4.0; access approved separately | **no** | download script + split only; non-commercial use |
| CIRE | **unverified in this checkout** | no | obtain from provider and confirm license |
| KUL AAD | **unverified in this checkout** | no | obtain from provider and confirm license |
| DTU AAD | **unverified in this checkout** | no | obtain from provider and confirm license |
| NMED-T | research-use dataset; stimuli copyrighted | no | use under the original terms; no stimulus audio release |

## Checklist before public data release

1. Obtain written redistribution terms for CIRE, KUL and DTU.
2. Keep SparrKULee to download instructions and split manifests; do not mirror raw files.
3. Do not publish NMED-T audio; publish only derived features or aggregate statistics.
4. Re-run this table if a new dataset is added.
