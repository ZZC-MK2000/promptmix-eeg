# Third-party notice: LP-MusicCaps model code

`lpmc_model/bart.py` and `lpmc_model/modules.py` are copied from the official
LP-MusicCaps implementation:

- SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam.
  *LP-MusicCaps: LLM-Based Pseudo Music Captioning.* ISMIR 2023.
- Upstream: `seungheondoh/lp-music-caps` (MIT license).
- The files are included only so `code/information/caption_lp.py` can import the
  architecture. No model weights are included; the checkpoint is external
  (`Ostixe360/lp-music-caps` / `seungheondoh/lp-music-caps`).
