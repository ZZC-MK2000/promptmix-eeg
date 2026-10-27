# Ethics and responsible release

- **Privacy.** EEG can carry identity and health-adjacent information. AEB releases no raw
  EEG, no subject-level embeddings, and no checkpoints; only split indices and aggregate
  results are public.
- **Consent and licenses.** SparrKULee is used under CC BY-NC 4.0 and is not redistributed.
  CIRE/KUL/DTU users must obtain data from the original providers and follow their terms.
- **Non-clinical scope.** The benchmark measures naturalistic auditory decoding; it is not a
  diagnostic tool and must not be presented as one.
- **Non-commercial scope.** The reference cache is for non-commercial research. A commercial
  user must replace or re-license the restricted datasets.
- **Evaluation integrity.** Every query-conditioned method must report zero-EEG, shuffled-EEG,
  correct-pairwise and EEG-only controls. Random-row and adjacent-window results are upper
  bounds, never cross-subject evidence.
- **No MOS / no perceptual claim.** The downstream audio demo is illustrative; it is not a
  perceptual quality benchmark.
