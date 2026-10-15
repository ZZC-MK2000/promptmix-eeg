# Project naming decision (2026-10-15)

**Old names**: `CLaBraM` (upstream pre-trained EEG encoder), `mainv3` (workspace version), `EEG2Music` (generic).

**New public-facing name**: **PromptMix-EEG**

**Rationale**:
- Descriptive: "Prompt" (frozen prompt library) + "Mix" (mixture weights) + "EEG" (input modality).
- Matches the paper title: *PromptMix-EEG: Frozen-Generator Music Synthesis from Brain Signals via Prompt-Library Mixture Weights*.
- Distinguishes from the earlier `mixed2` / `ep16` arms, which used free-form CLAP embeddings and LoRA fine-tuning.
- Easy to remember, search, and cite.

**Alternatives considered**:
- NeuroPrompt (too generic, "neuro" could mean anything)
- BrainPrompt (same issue)
- EEG-PromptMixer (verbose)
- FrozenGen-EEG (emphasizes the frozen generator but not the mixture weights)
- Manifold-EEG (too abstract)
- LibraryCond-EEG (jargon)

**Workspace directory**: `mainv3/` is **retained unchanged** to avoid breaking 30+ scripts with hardcoded paths (`ROOT = Path("D:/Projects/CLaBraM/mainv3")`) and existing caches/checkpoints. A new directory `promptmix-eeg/` serves as the public-facing entry point with a README pointing to `mainv3/`.

**When to fully rename**: after the paper is submitted and the mode (A)/(B) experiments are complete, if there is a need to release the artifact publicly. At that point, batch-replace paths with `sed` or refactor to relative paths / environment variables.

**Action items**:
- [x] Create `promptmix-eeg/README.md` (done 2026-10-15).
- [x] Use "PromptMix-EEG" in the paper outline (`paper/outline_promptmix_eeg.md`).
- [ ] Update any external-facing documentation (slides, reports, repo README if/when made public).
- [ ] After submission: decide whether to fully rename the workspace directory.

---

*Decision made autonomously per the goal objective: "此外现在项目的命名已经不合适了，换个新的".*
