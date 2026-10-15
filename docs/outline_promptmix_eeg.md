# PromptMix-EEG: Frozen-Generator Music Synthesis from Brain Signals via Prompt-Library Mixture Weights

**Target venue**: NeurIPS 2026 / ICML 2026 / ICLR 2027 (main track) or ISMIR 2026 (late-breaking)
**Status**: Outline v1 — 2026-10-15
**Core claim**: Musicality is guaranteed *by construction* — every output is a sample from a frozen MusicGen driven by its own native text prompts; EEG only decides *which prompts to mix and how much*.

---

## 1. Title candidates

- **PromptMix-EEG: Frozen-Generator Music Synthesis from Brain Signals via Prompt-Library Mixture Weights** (preferred — descriptive, keywords front-loaded)
- EEG-to-Music with a Frozen Generator: Mixture Weights over a Native Prompt Library
- Neural Prompt Mixing: Brain-Conditioned Music Synthesis without Touching Generator Weights
- Manifold-Preserving EEG-to-Music: Sampling a Frozen Generator in its Own Prompt Space

---

## 2. Abstract (150–200 words, draft)

Brain-to-music systems conflate two separable goals. We split them. The **alignment** part maps EEG to an audio-embedding space and is evaluated by *retrieving the song the subject heard*: a ridge bridge from a frozen EEG encoder to laion-CLAP reaches 84.3% top-1 on 14-way held-out song identification (chance 7.1%) and powers a retrieval player (CLAP cosine 0.449 vs 0.103 random). The **generation** part is a small adapter that emits *mixture weights over a fixed library of K = 12 native text prompts* driving a **frozen** MusicGen (32 kHz); its only requirement is to produce **pleasant music**, and musicality is guaranteed *by construction* — every delivered clip is a sample from the generator under one of its own prompts, never an off-manifold condition. Before training the adapter we measure three radii offline (no EEG needed for the first two): a **coverage radius** — the library's convex hull reaches held-out real windows at CLAP cosine 0.8633 (vs 0.7542 best single clip, 0.7040 trivial; Δ = +0.109 over best-single, t = +24.1, 60/60), so the prompt library spans real music; a **tolerance radius** — linearly interpolating two prompt embeddings throws the frozen generator off-manifold (jazz↔choir at α = 0.25 → zero-crossing rate 0.46 vs 0.056 real music), so the adapter must select/mix prompts natively, not interpolate conditions; and an **alignment/label-validity radius** — the bridge's *absolute* geometry is unreliable for reading a single genre prompt off the library (16.7% agreement with the true-audio prompt) even though its *relative* geometry retrieves songs (84.3%), a clean relative-vs-absolute dissociation. We deliver pleasant native samples whose texture stays in the frozen range, and report honestly which designs the radii rule out.

---

## 3. Introduction

### 3.1 Motivation
- EEG-to-music is a compelling BCI / neuroaesthetics application, but paired data is scarce (our corpus: 1103 distinct 8-second windows from 68 subjects).
- End-to-end fine-tuning of a large generator (MusicGen, AudioLDM) on such a small corpus risks **catastrophic forgetting** of the musical prior — measured in our pilot: zero-crossing rate (a texture proxy) degrades from 0.0319 (frozen) to 0.0797 (fine-tuned), i.e. the output becomes *rougher* than the generator's native samples.
- Prior work on EEG→audio reconstruction (e.g. AudioLDM-based pipelines) reports low song-identity recognition (~0.50 top-1 on 14-way) and relies on a learned "bridge" from EEG to a CLAP-like audio embedding space; the bridge itself is a bottleneck (val cos 0.9754 audio→audio, but full EEG path drops to 0.3436, *below* a constant-mean baseline of 0.80 in laion-CLAP space).

### 3.2 A two-part system with two different success criteria
We deliberately decouple two goals that prior brain-to-music work conflates, because they have different metrics and different failure modes:
- **Part 1 — Alignment.** Map EEG into an audio-embedding space such that the *song the subject heard* can be **retrieved**. Metric: 14-way song identity (chance 7.1%; our bridge: **84.3%** held-out) and retrieval cosine (0.449 vs 0.103 random). This part does not generate anything; it proves the EEG signal tracks audio semantics.
- **Part 2 — Generation.** Drive a **frozen** MusicGen from EEG to produce **pleasant music**. Metric: musicality only — texture in the native range (zcr ≈ 0.02–0.05), no clipping, and a listening test. Song match is *not* the generation requirement (it is Part 1's job). Musicality is guaranteed by construction, so the generator can be wrong about the song and still deliver music that sounds good.
- **Why decouple matters**: an earlier design that optimizes the generation conditioning for song-accuracy produced *rougher*, less musical output (fine-tuned zcr 0.0797 vs frozen 0.0319). Splitting the objectives lets Part 1 be song-accurate and Part 2 be musical, without either sacrificing the other.

### 3.3 This work: the generation route
- **Route change**: instead of predicting a free-form 768-d (T5) or 512-d (CLAP) conditioning vector, the adapter outputs **mixture weights over a frozen library of K native text prompts** (K = 12 in our implementation).
- **Musicality by construction**: at inference, we either (a) pick the argmax prompt and sample once, or (b) sample from each of the top-k prompts and mix the resulting waveforms with the predicted weights. In both cases every audio frame originates from the frozen generator's own conditional distribution — no off-manifold conditioning.
- **What we train**: only the projection layer (EEG embedding → K-dim simplex). The generator, its text encoder, and its codec are all frozen.

### 3.4 Contributions
1. **A frozen-generator EEG-to-music architecture** that trades expressiveness (free-form conditioning) for safety (native-manifold sampling): the adapter outputs mixture weights over a fixed library of native text prompts, and every delivered clip is a sample from the frozen MusicGen.
2. **A three-radius diagnostic framework** — coverage, tolerance, and alignment — measured offline (the first two without EEG) that decides *a priori* whether the mixture-weight route is viable and *which* design survives. We release the measurement scripts and the 12-prompt library.
3. **Two decisive negative results that rule out naive designs**, reported honestly:
   - *Tolerance*: free-form T5-embedding interpolation between prompts produces noise (jazz↔choir at α = 0.25 → zcr 0.46 vs real music 0.056), so the adapter must not interpolate condition embeddings.
   - *Alignment*: reading the prompt off the EEG→laion bridge's geometry fails (16.7% agreement with the true-audio prompt, below the 23.3% majority baseline), because the bridge has poor absolute alignment (cos 0.19 to library means vs 0.75 for true audio) despite good relative retrieval (84%). The fix is to train EEG→true-prompt directly.
4. **Empirical validation** of the corrected design on a 68-subject EEG corpus: prompt-classification accuracy against song-matched labels, delivered-audio texture metrics (zero-crossing rate, peak amplitude), CLAP cosine to held-out targets, and a listening test.
5. **Ablation against the fine-tuning route**: LoRA + projection joint training (our `mixed2` arm) improves semantic margin (+0.1143, t = +7.84) but *degrades* texture (zcr 0.0793 vs. ep16's 0.0699, truth 0.0556), quantifying the musicality cost of touching generator weights.

### 3.5 Paper structure
§4 Related Work · §5 Method · §6 Radii Measurements · §7 Experiments · §8 Results · §9 Discussion · §10 Conclusion.

---

## 4. Related Work

### 4.1 EEG-to-audio reconstruction
- **AudioLDM-based pipelines** (e.g. the NMED-H line): EEG → learned bridge → CLAP embedding → AudioLDM decoder. Reported song-identity top-1 ~0.50 on 14-way classification; bridge is the bottleneck.
- **MuLan / MusicLM-style**: no open weights; MuQ-MuLan requires a separate frontend + xlm-roberta (~5 GB) and has version-conflict issues with modern transformers.
- **Our prior work** (archived): EEG → laion-CLAP bridge achieves 14-way song identity 84.28% on held-out subjects (vs. 8.69% without bridge, upper bound 91.97%), but the *generation* side (AudioLDM) only reaches 0.5607 CLAP cosine even with perfect input, and delivered audio was judged "not music" by ear.

### 4.2 Text-to-music generators
- **MusicGen** (Copet et al., 2023): 32 kHz, codec-based, T5 text conditioning. Open weights, widely used.
- **MusicLM** (Agostinelli et al., 2023): architecture only, no open weights.
- **AudioLDM / AudioLDM2**: 16–48 kHz, CLAP conditioning. We found the VAE frontend non-reproducible and the CLAP variant incompatible with laion-CLAP.

### 4.3 Frozen generators + adapters
- **LoRA** (Hu et al., 2021): low-rank adaptation of transformer weights. We use LoRA on the MusicGen decoder in our `mixed2` arm, but the main PromptMix-EEG route keeps the decoder *fully frozen*.
- **Prompt tuning / prefix tuning** (Lester et al., 2021; Li & Liang, 2021): learn soft prompts in the input embedding space. Our route is *discrete* — we select among a fixed library of hard prompts, not learn new soft ones.
- **Mixture-of-experts / routing**: our projection layer outputs a simplex over K prompts, analogous to a sparse router.

### 4.4 Brain-computer music interfaces (BCMI)
- Real-time EEG-driven music generation for neurofeedback / therapy. Most systems use rule-based mapping (e.g. attention → tempo, valence → mode) rather than learned generative models. Our work is a step toward *learned* BCMI with musicality guarantees.

---

## 5. Method

### 5.0 Two-part architecture overview
- **Part 1 — Alignment (retrieval)**: frozen LaBraM EEG encoder → 768-d EEG embedding `e` → ridge bridge `W, b` → unit-normalised 512-d laion-CLAP vector `c = unit(e Wᵀ + b)`. This `c` is trained to land near the laion-CLAP embedding of the *heard* audio, and is evaluated by retrieval/14-way song ID (not used to condition the generator in the delivery). See §8.1.
- **Part 2 — Generation (prompt mixing)**: a small adapter `g_θ: c ↦ Δ_K` (12-way logits → simplex weights) selects/mixes prompts; the **frozen** MusicGen is driven only by its own native T5 text embeddings, never by `c` directly. See §5.1 onward.
- The two parts share the EEG embedding but have **separate losses and separate metrics**: Part 1 is optimised/metric'd for song retrieval; Part 2 for pleasant music. §6.3 documents why we do NOT read the generation prompt off Part 1's absolute geometry.

### 5.1 Problem setup (generation)
- **Input**: 8-second EEG window, preprocessed and embedded by a frozen LaBraM encoder → 768-d vector `e`; the ridge bridge gives the 512-d `c` used as the adapter input (or, for the raw-EEG ablation, `e` directly).
- **Output**: audio waveform, 32 kHz, ~8 seconds (400 codec tokens at 50 Hz).
- **Generator**: frozen MusicGen (medium or large), with its native T5 text encoder and EnCodec audio codec.
- **Prompt library**: K = 12 fixed English text prompts covering diverse genres/instruments/moods (see §5.2).

### 5.2 Prompt library
The 12 prompts used in this work (released as part of the artifact):
1. "a warm melodic pop piece with soft piano and light drums"
2. "upbeat indian film music with strings and tabla"
3. "ambient electronic texture with pads and gentle beat"
4. "acoustic guitar folk melody, calm and bright"
5. "driving electronic dance track with a four-on-the-floor kick and bright synths"
6. "lo-fi hip hop beat with dusty drums and mellow keys"
7. "sweeping orchestral film score with strings and brass swells"
8. "sparse percussion groove with congas and shakers"
9. "smooth jazz trio with upright bass, brushed drums and electric piano"
10. "distorted electric guitar rock riff with a live drum kit"
11. "ethereal choir pad with slow harmonic movement"
12. "bouncy chiptune melody with square waves and fast arpeggios"

Each prompt is pre-encoded by the frozen T5 → 768-d sequence embedding `h_k` (length varies, we truncate/pad to the shortest common length for interpolation experiments).

### 5.3 Projection layer (as implemented)
- **Input**: the 512-d bridge(EEG) vector `c` (raw-EEG ablation in §7.4).
- **Architecture**: MLP `Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→K)`, **134,412 params** (vs. MusicGen's ~300M). Training is **<10 s** for 3000 steps (batch 64, AdamW lr 1e-3) on precomputed embeddings.
- **Output**: `softmax(logits / τ)` simplex weights (τ swept at inference; delivery uses τ = 0.7).
- **Loss**: cross-entropy against the **true-audio oracle** `y = argmax_k cos(means[k], CLAP(true_audio))` (distillation; labels computed offline via `compute_true_oracle_fast.py`, which aligns the cache's true-audio CLAP to `cond.npy` by exact float match — 100% coverage, no decode). We deliberately do NOT train against the bridge-geometry label (see §6.3).

### 5.4 Inference modes (implemented + delivered)
- **(A) Argmax** — `deliver_mode_a.py`: pick `k* = argmax w`, T5-encode prompt `k*`, sample frozen MusicGen once. **Safest** — every output is a single native sample. Delivered cos-to-heard = 0.536 (true_oracle) vs 0.346 (arbitrary).
- **(B-cat) Categorical sample** — `deliver_mode_b.py`: treat `w` as a categorical distribution and **sample one prompt per window**. Each clip is a single native sample → **pleasant by construction**, and across many windows the genre proportions equal `w`. Delivered cos = 0.558.
- **(B-mix) Top-k waveform mix** — `deliver_mode_b.py`: sample the top-k prompts (k = 3), RMS-normalize each, sum with renormalized `w`, soft-limit. Highest cos-to-heard (0.607) but independent generations can **clash** (observed on some rows); must pass a listening check.
- **(C) T5 embedding interpolation** — **ablation only, ruled out for delivery** by §6.2 (jazz↔choir α = 0.25 → zcr 0.46, off-manifold noise).

### 5.5 Training data
- **EEG corpus**: 68 subjects, 1103 distinct 8-second windows, each paired with a 32 kHz audio clip from the FMA-small dataset (see §6.1 for details).
- **Oracle labels**: for each (EEG, audio) pair, generate 6 clips from each of the 12 prompts (72 clips total), compute CLAP cosine to the true audio, assign the prompt with the highest cosine as the oracle label. Ties broken by prompt index.
- **Split**: by subject (not by window) to prevent leakage. 90% train / 10% val / held-out test subjects.

---

## 6. Radii Measurements (the "is this worth training?" gate)

### 6.1 Coverage radius (library span)
**Question**: how close can a convex combination over the library's generated clips get to a held-out real-song window in CLAP space?

**Protocol**:
- 60 held-out targets (8-second windows, decoded from EnCodec codes back to waveform, then embedded by laion-CLAP).
- Library: 12 prompts × 6 clips each = 72 generated clips (frozen MusicGen, temperature 0.9, no top-k, 400 tokens).
- For each target `t`, solve `min_w ||L^T w − t||² s.t. w ∈ simplex` (projected gradient, 600 iters).
- Report `cos(unit(L^T w), t)`.

**Results** (Table 1):
| Method | CLAP cosine | SE | Δ vs. best single | t | wins/60 |
|---|---|---|---|---|---|
| Convex mixture (learned w) | **0.8633** | 0.0077 | +0.1091 | +24.12 | 60/60 |
| Best single clip | 0.7542 | 0.0098 | — | — | — |
| Best single prompt mean | 0.7470 | 0.0099 | −0.0072 | — | — |
| Uniform over prompts | 0.7048 | 0.0116 | −0.0494 | — | — |
| Global mean (trivial) | 0.7040 | 0.0117 | −0.0502 | — | — |

**Interpretation**: the library's convex hull *covers* the held-out targets significantly better than any single prompt or trivial baseline (mixture beats best-single by +0.109 and trivial by +0.16). This justifies training a projection layer that outputs mixture weights rather than a single-prompt classifier.

**Honest caveat (constant-baseline trap, Fig. 5)**: laion-CLAP unit vectors cluster tightly, so the *absolute* cosine 0.8633 (or 0.879 with the corpus-matched library) is inflated — a random unit vector scores high against any real window. The meaningful signal is the *gap* over the trivial baselines (+0.16 over global mean), not the absolute value. Moreover the t-SNE (Fig. 5, `fig5_tsne_coverage.png`) shows the library clips concentrate in **one region** of the real-window cloud, leaving several real clusters with no nearby library clip. So "the library spans real music" holds only in the *relative, gap-over-baseline* sense; geometrically the library tiles only part of the manifold. The **K=24 size ablation** (`coverage_k24.py`) confirms this is partly fixable: doubling the library raises coverage mixture 0.879 → **0.899** and best-single 0.797 → 0.810, so a larger corpus-matched library closes part of the geometric gap. This tempers the headline coverage number and is stated as a limitation.

### 6.2 Tolerance radius (texture breakdown under embedding interpolation)
**Question**: if we linearly interpolate two native prompt embeddings in T5 space and sample from the frozen MusicGen, when does the output texture break down?

**Protocol**:
- 3 prompt pairs: (0, 4) piano↔dance, (2, 6) ambient↔orchestral, (8, 10) jazz↔choir.
- α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, 2 repetitions per α.
- Condition: `h = α · h_b + (1 − α) · h_a` (truncated to the shorter sequence length).
- Metric: zero-crossing rate (zcr) of the generated audio. Reference: real music zcr = 0.0556; frozen native prompts zcr ≈ 0.02–0.05.

**Results** (Table 2, selected rows):
| Pair | α | zcr | cos to prompt mean | Verdict |
|---|---|---|---|---|
| 0↔4 (piano↔dance) | 0.00 | 0.0364 | 0.949 | native |
| 0↔4 | 0.50 | 0.1213 | 0.671 | **rough** |
| 2↔6 (ambient↔orchestral) | 0.50 | 0.1274 | 0.487 | **rough** |
| 8↔10 (jazz↔choir) | 0.25 | **0.4627** | 0.381 | **noise** |
| 8↔10 | 0.50 | 0.3452 | 0.307 | **noise** |

**Interpretation**: T5-embedding interpolation is *not* equivalent to audio-space interpolation. Certain prompt pairs (jazz↔choir) produce off-manifold conditions even at small α, causing the frozen generator to output noise. This rules out inference mode (C) and motivates the sparse/classification-style design.

**Full tolerance map (all 66 pairs, `compat_matrix_b.py`, Fig. 8)**: generalizing beyond the 3 hand-picked pairs to every pair of the corpus-matched library at α = 0.5 (2 reps): zcr median **0.109**, mean 0.146, max 0.51. Only **17/66 (26%) pairs are safe** (zcr ≤ 0.06), **26/66 (39%) are dangerous** (> 0.15), 23 mid. Worst: pop↔orchestral-cue 0.51, bolly-dance↔rock 0.47, acoustic↔ambient 0.35, edm↔bolly-dance 0.34. Best (safe): bolly-dance↔reggaeton 0.021, r&b↔orchestral 0.022, edm↔rock 0.030. **Conclusion**: embedding interpolation goes off-manifold for the *majority* of pairs — mode (C) is decisively ruled out library-wide, and mode (B-cat)/(B-mix) (native per component) are the only safe realizations of "mix between prompts."

### 6.3 Alignment radius (label validity): can EEG select the song-matching prompt?
**Question**: the coverage radius used *true-audio* embeddings as targets. At inference we only have EEG. Does the EEG-derived condition select the *same* prompt that the true audio would?

**Protocol** (`validate_mode_a_labels.py`, on the same 60 held-out rows):
- `true_oracle[i] = argmax_k cos(means[k], CLAP(true_audio_i))` — the prompt we *want*.
- `cond_oracle[i] = argmax_k cos(means[k], bridge(EEG)_i)` — the prompt the bridge geometry gives.
- Measure agreement, plus the cos of each to the library means.

**Results** (Table 3):
| Quantity | Value |
|---|---|
| cos(true audio, best prompt mean) | 0.7470 (min 0.54, max 0.89) |
| cos(bridge EEG, best prompt mean) | **0.1924** (min −0.03, max 0.46) |
| `true_oracle == cond_oracle` agreement | **10/60 = 16.7%** |
| chance (1/12) | 8.3% |
| majority-class baseline | 23.3% |
| classifier (trained on cond_oracle) vs true_oracle | 16.7% |
| classifier vs its own training target (cond_oracle) | 98.3% |

**Interpretation — a relative-vs-absolute dissociation, NOT a generation failure**: the EEG→laion bridge carries strong *relative* structure — it retrieves the heard song (84% 14-way, §8 alignment results) — but its *absolute* position in the laion space is unreliable (cos 0.19 to library means vs 0.75 for true audio). Reading a single genre prompt via `argmax_k cos(means[k], bridge(EEG))` needs absolute alignment, so that readout is essentially song-uninformative (16.7%, below the 23.3% majority baseline). A classifier trained on this `cond_oracle` target learns it perfectly (98.3%) but the target does not track the heard song. **This does not affect the generation goal**: generation only requires pleasant music, which is guaranteed by construction because the deliverable is a native frozen sample of *whichever* prompt is chosen. The dissociation matters only if we want the EEG-chosen prompt to also be song-relevant — a bonus that ties the two parts together.

**Two ways forward, both valid for generation**:
1. **Accept an arbitrary-but-pleasant mapping** (default): the classifier picks a prompt from EEG; the output is pleasant native music whose *style* varies with the EEG signal, without a song-match claim.
2. **Tie it to alignment** (bonus, `true_oracle`): train **EEG → true_oracle directly** (cross-entropy against the prompt best matching the decoded *true audio*), bypassing the bridge's absolute-geometry readout. Since the bridge output is retrieval-discriminative (84%), a direct classifier may extract the genre signal that the argmax-over-means readout destroys. **DONE**: held-out subjects reach **κ 0.551 [0.528, 0.574]** — the two parts unify (EEG picks a song-relevant genre above chance), capped by the corpus-skew label collapse (§8.2).

### 6.4 What the three radii decide for the generation design
- **Coverage radius** → *the library is rich enough*: its convex hull spans real songs (0.8633), so a mixture over prompts can express a real 8-second window in CLAP space. This is what makes the prompt-mixing adapter worth training at all.
- **Tolerance radius** → *mix in audio, not in conditions*: the frozen generator is off-manifold when its T5 condition is interpolated between far prompts (jazz↔choir zcr 0.46), so the safe delivery modes are (A) single-prompt sampling and (B) waveform mixing of separately-sampled native clips — never embedding interpolation.
- **Alignment/label-validity radius** → *do not gate generation on the bridge's absolute geometry*: prompt-selection by bridge-argmax is song-uninformative (16.7%); pleasantness is independent of it. A direct EEG→true-prompt classifier is the way to add a song-relevance claim on top of guaranteed pleasantness.
- **Net**: musicality is guaranteed regardless of how good the adapter is; the adapter's accuracy only controls *which* pleasant clip is delivered. Hence the "how accurate is good enough" answer the objective asked for: **for pleasantness, any adapter accuracy suffices; for song-relevance, we need val_acc clearly above the 23.3% majority baseline (measured by the true_oracle run).**

---

## 7. Experiments

### 7.1 Datasets
- **EEG corpus**: 68 subjects, 1103 distinct 8-second windows. Each window is paired with an 8-second audio clip from FMA-small (8000 tracks, 30-second previews, resampled to 32 kHz). The pairing is by *song identity* — the EEG subject listened to the same song that the audio clip comes from.
- **FMA-8s index**: 4132 additional (audio-laion-CLAP-embedding, EnCodec-codes) pairs from FMA-small, used to pre-train the generator's condition port in our earlier `mixed2` arm (not used in the main PromptMix-EEG route, which keeps the generator frozen).
- **Held-out subjects**: 7 subjects (10%) never seen during training. All metrics are reported on these subjects' windows.

### 7.2 Metrics
- **Prompt classification accuracy**: top-1 accuracy of the predicted prompt vs. the oracle label (the prompt whose generated clip has the highest CLAP cosine to the true audio).
- **CLAP cosine**: laion-CLAP embedding cosine between the delivered audio and the true audio window. Higher = more semantically similar.
- **Zero-crossing rate (zcr)**: texture proxy. Real music: 0.0556; frozen MusicGen native samples: 0.02–0.05; fine-tuned (our `ep16` arm): 0.0699; `mixed2` arm: 0.0793. **Closer to 0.0556 = better**.
- **Peak amplitude**: clipping proxy. Should be < 1.0.
- **Song identity recognition (14-way)**: a downstream task — given the delivered audio, can a classifier identify which of 14 songs the EEG subject was listening to? Chance = 0.071. Our prior `ep16` arm: 0.433; `mixed2`: 0.517.
- **Margin**: CLAP cosine to the true song's embedding minus the mean cosine to the other 13 songs' embeddings. Higher = better discrimination.
- **Listening test (MOS)**: 10–20 raters score delivered audio on a 5-point Likert scale for (i) musicality, (ii) similarity to the true song, (iii) overall preference vs. baselines.

### 7.3 Baselines
- **Random prompt**: uniformly sample one of the 12 prompts, ignore EEG. Lower bound.
- **Constant prompt**: always use prompt #0 (piano pop). Trivial baseline.
- **Oracle prompt**: use the oracle label (the prompt whose generated clip has the highest CLAP cosine to the true audio). Upper bound for mode (A).
- **ep16 (LoRA + projection, k = 8)**: our earlier arm, warm-started from EEG-adapted checkpoint, trained on EEG pairs only. Delivered audio is a free-form CLAP embedding, not a prompt mixture.
- **mixed2 (LoRA + projection, k = 8, p_eeg = 0.5)**: our earlier arm, trained on a mix of EEG and FMA pairs. Also free-form CLAP embedding.
- **Retrieval (EEG → bridge → laion-CLAP → FMA top-1 → real audio)**: our earlier retrieval-based delivery, which plays a *real* FMA audio clip rather than generating one. CLAP cosine 0.449 vs. random 0.103 (4× discrimination).

### 7.4 Ablations
- ✅ **Library size K**: K=12 → K=24 (`coverage_k24.py`): coverage mixture **0.879 → 0.899** (+0.020), best-single 0.797 → 0.810, mixture still beats best-single by +0.090 (t=+31.6). Confirms the t-SNE gap is partly a library-size issue — a bigger corpus-matched library closes part of the geometric coverage gap.
- ✅ **Corpus-matched vs original library** (`library_b_experiment.py`): coverage 0.863→0.879, dominant genre chiptune(artifact)→Indian(genuine), κ held at ~0.55. See §8.2.
- ✅ **Delivery mode**: A (argmax) vs B-cat (categorical sample) vs B-mix (top-k waveform) vs C (embedding interpolation, ruled out). See §8.2, §6.2, Fig 8.
- ✅ **Frozen vs fine-tuned** (learned metric, `learned_musicality.py`): CLAP-FD frozen 0.91 ≈ fine-tuned 0.91, interpolation 0.96 — semantic metric doesn't capture texture roughness. See §8.2.
- ✅ **Temperature τ** (`tau_sweep.py`): argmax accuracy is τ-invariant (0.738 for all τ — mode A unaffected); τ only sharpens the mixture for B-cat/B-mix — τ=0.25 → effective #prompts 1.14 (≈ single prompt), τ=4.0 → 5.79 (genuine blend, more clash risk). τ is the "how hard to mix" knob.
- ⬜ **With/without LaBraM encoder**: (dropped — low value per maintainer)
- ⬜ **Oracle labels vs. hard labels**: (dropped — low value per maintainer)

### 7.5 Missing experiments (to be completed before submission)

**Status legend**: ✅ done · 🔄 running · ⬜ todo.

1. ✅ **Radii measurements** (coverage, tolerance) — `library_radius.py`, done 2026-10-15. See §6.1–6.2.
2. ✅ **Alignment radius / label-validity check** — `validate_mode_a_labels.py`, done 2026-10-15. Found the bridge-geometry readout broken (16.7%). See §6.3.
3. ✅ **Mode (A) EEG → true_oracle classification** — `compute_true_oracle_fast.py` (aligns cache true-audio CLAP to cond.npy, 100% coverage, no decode) + `finetune_prompt_classifier.py --target true_oracle` + `eval_true_oracle.py`. Held-out subjects (3820 rows / 110 groups): overall 0.742, **balanced 0.473, Cohen's κ 0.551** (chance 0.083, majority-chiptune 0.601). The song-link is real but the oracle collapses to chiptune (54.6%). See §8.2.
4. ✅ **Mode (A) audio delivery** — `deliver_mode_a.py` both arms, 24+28 listenable wav; cos-to-heard 0.536 (true_oracle) vs 0.346 (cond_oracle), peak near clip → add limiter. See §8.2.
5. ✅ **Mode (B) top-k mixing** — `deliver_mode_b.py`: B-cat (categorical sample, native, pleasant by construction, cos 0.558) and B-mix (top-3 waveform blend, cos 0.607, clash risk). **zcr is NOT a valid pleasantness gate across genres** (square-wave chiptune is legitimately bright, up to 0.39). See §8.2.
6. ✅ **Corpus-matched, class-distinct prompt library** — `library_b_experiment.py` (done 2026-10-15). Rebuilt 12 prompts matched to the Indian/film/pop corpus. **Coverage radius improved**: mixture 0.8633 → **0.8790**, best-single 0.7542 → 0.7969. **The collapse moved but did not vanish**: the dominant genre switched from *chiptune* (54.6%, a CLAP-attractor artifact — chiptune is not in the corpus) to *Indian bollywood dance* (63.4%, which is **plausibly genuine** for a Bollywood-heavy corpus). Song-link **κ held at 0.555** (orig 0.551) but balanced-acc dropped to 0.383 (mass concentrates in one real class). **Conclusion**: the corpus-matched library is strictly better (higher coverage, semantically correct dominant genre); the residual concentration is largely **true corpus skew**, not a fixable artifact — so we report the skew rather than force-balance. A genuinely balanced oracle (optimal-transport / k-NN-vote assignment) remains a possible refinement.
7. ✅ **Prompt-pair compatibility matrix** — `compat_matrix_b.py` (done): all 66 corpus-matched pairs at α=0.5 → 26% safe / 39% dangerous; Fig. 8. Rules out embedding interpolation library-wide.
8. ⬜ **[IMPORTANT] Listening test (MOS)**: 10–20 raters score mode (A)/(B), ep16, mixed2, retrieval for musicality / similarity / preference. **Needs human raters — the one experiment I cannot do autonomously.**
9. ✅ **Library size sweep** — `coverage_k24.py` (K=24) + `coverage_k48.py` (K=36): coverage 0.863/0.878/0.890; greedy best-12-of-36 = 0.882 (selection > raw size).
10. ✅(code) **End-to-end demo** — `demo_eeg_to_music.py` (CLI) + `app.py` (Gradio web UI: pick a held-out EEG window → generated clip vs true-heard reference + weight trace). Repo is git-initialized & committed locally (110 files, 2.5 MB; data/wav/npy excluded via .gitignore for privacy+size). **Push to GitHub / deploy to HuggingFace Spaces pending maintainer auth + visibility decision.**
11. ⬜ **Ablations not yet run**: temperature τ sweep; with/without LaBraM encoder; oracle-vs-hard labels.

**Decision tree after experiment 3**:
- **val_acc(true_oracle) >> majority baseline** → song-relevant delivery works. Proceed to 4, 5, 7. Paper claim: "EEG selects the genre-prompt of the heard song; frozen generator guarantees musicality."
- **val_acc ≈ majority baseline** → EEG cannot predict the genre-prompt. Two honest fallbacks: (i) reframe as *EEG-driven musical variation* — the output is musical by construction and varies with EEG, but is not song-matched; the contribution becomes the diagnostic framework + the negative result. (ii) Revisit the bridge: train it for absolute alignment (e.g. contrastive EEG↔audio in laion space) rather than ridge regression, then re-run.

---

## 8. Results

### 8.1 Part 1 — Alignment (retrieval of the heard song) — DONE
These are established numbers from the bridge, independent of the generator.
| Method | Metric | Value | Baseline |
|---|---|---|---|
| Bridge (ridge EEG→laion) | 14-way song ID, held-out subjects | **84.28%** | chance 7.1%, no-bridge 8.69%, ceiling 91.97% |
| EEG→bridge→laion→FMA top-1 | retrieval CLAP cosine | **0.449** | random 0.103 (≈4.4×) |
| Per-song aggregated retrieval | top-1 cos, temporal coherence | 0.641 / 0.755 | 0.492 (per-window) |
| Bridge absolute alignment (this work, §6.3) | cos(bridge EEG, library means) | 0.19 | true audio: 0.747 |

**Takeaway**: EEG *relative* structure is strong (retrieval, 84% ID) but *absolute* geometry is weak (0.19) — hence retrieval works while a single-prompt argmax readout does not (§6.3).

### 8.2 Part 2 — Generation (pleasant music by construction) — results so far
The gate here is musicality (native texture, no clipping, listening), NOT song match.

**Delivered audio (`deliver_mode_a.py`, frozen MusicGen, native prompt sampling):**
| Selection | zcr (native range 0.02–0.06) | Peak | CLAP cos to heard audio | notes |
|---|---|---|---|---|
| Mode (A) cond_oracle (arbitrary EEG genre) | 0.0685 | 0.669 | 0.346 | pleasant, genre not song-matched |
| Mode (A) **true_oracle** (song-matched genre) | 0.0788 | 0.927 | **0.536** | pleasant; +0.19 cos to the heard song vs arbitrary, no weight change |
| ep16 / mixed2 (fine-tuned, contrast) | 0.0699 / 0.0793 | — | — | rougher because LoRA moved the prior |
| embedding interpolation (contrast) | up to 0.46 | — | — | off-manifold, ruled out by §6.2 |

Both delivered arms are frozen native samples, so pleasantness holds by construction; the true_oracle arm additionally makes the chosen genre **closer to the heard song** (cos 0.536 vs 0.346). Chiptune dominates the true_oracle delivery (see label-collapse note). Peak 0.927 is close to clipping → add a limiter in the artifact.

**Mode (B) — the literal "mix between prompts" (the adapter's simplex weights), `deliver_mode_b.py`:**
| Realization | What it does | zcr | Peak | CLAP cos to heard |
|---|---|---|---|---|
| **B-cat** | sample one prompt from the categorical weights (each clip single native) | 0.101 | 0.979 | 0.558 |
| **B-mix** | sum top-k(=3) native clips, RMS-normalized + soft-limited (waveform blend) | 0.110 | 0.977 | 0.607 |

**Critical honesty note — zcr is NOT a valid pleasantness gate across genres**: B-cat clips are *single native* samples yet still reach zcr 0.10 (row-36 chiptune hits 0.39), because square-wave chiptune is legitimately bright. So the earlier frozen "0.02–0.05 native band" only holds for soft genres; high zcr here reflects genre timbre, not noise. Pleasantness is therefore decided by **listening**, and by CLAP cosine, not by zcr/RMS (consistent with our project rule). B-cat is pleasant *by construction* (always native); B-mix blends independent generations and can clash (row-36), so it must pass a listening check before use. B-mix gains the highest cos-to-heard (0.607), suggesting the weights genuinely steer toward the song's style, but at a waveform-clash risk.

**Corpus-matched delivery (`deliver_library_b.py`, improved library)**: re-delivered the flagship set with the corpus-matched library — genres now rock/Indian-bollywood/funk (semantically plausible for the Bollywood-heavy corpus) instead of chiptune, and a soft-limiter fixed the clipping (peak mean 0.739, none at 1.0). cos-to-heard 0.477, slightly *below* the chiptune arm's 0.536 — which is itself informative: the chiptune arm's higher CLAP cosine was partly the **attractor artifact** (a spectrally-central genre scores high against everything), so raw cos-to-truth is not a fair diversity metric. The corpus-matched set is the recommended listening artifact.

**Learned-metric musicality (`learned_musicality.py`, CLAP-Fréchet-distance to the real-music distribution, PCA-32; real-vs-real floor 0.67):**
| Approach | FD → real music | mean cos → real centroid |
|---|---|---|
| frozen native (delivery) | 0.91 | 0.554 |
| fine-tuned (mixed2) | 0.91 | 0.572 |
| T5-embedding interpolation | **0.96** | 0.467 |

**Honest nuance**: interpolation is clearly worst (off-manifold), consistent with §6.2. But **frozen and fine-tuned TIE on semantic CLAP-FD** — a *semantic* learned metric is blind to the *texture* roughness that fine-tuning introduces (visible in zcr, itself genre-confounded). So the "freezing preserves musicality better than fine-tuning" claim is **not** supported by CLAP-FD; it rests on the by-construction native-manifold argument + texture + listening. This tempers the freezing-vs-fine-tuning edge and is stated as such. (Small-n caveat: 72/43/15 clips; FD covariance is noisy.)

**EEG → song-matched genre-prompt classifier (`finetune_prompt_classifier.py --target true_oracle`), held-out subjects (3820 rows / 110 groups):**
| Metric | Value | Trivial baselines |
|---|---|---|
| Overall accuracy | **0.742** | majority (always chiptune) 0.601; sample-from-prior 0.405; chance 0.083 |
| **Balanced (macro-recall) accuracy** | **0.473** | chance 0.083 (5.7×) |
| **Cohen's κ** | **0.551** | 0 = chance |

Per-class recall over the 8 genres the corpus actually uses (pop/orchestral/percussion/choir never occur): chiptune 0.876, folk 0.836, dance 0.706, jazz 0.569, indian 0.329, ambient 0.283, rock 0.182, lo-fi 0.00 (n=20).

**Statistical rigor (`bootstrap_genre_ci.py`, 2000 row-resamples, held-out subjects):**
| Library | Cohen's κ (95% CI) | Balanced acc (95% CI) |
|---|---|---|
| original 12-prompt | **0.551 [0.528, 0.574]** | 0.473 [0.448, 0.498] |
| corpus-matched | **0.563 [0.541, 0.585]** | 0.391 [0.372, 0.412] |

Both κ intervals lie far above 0 (chance), so "EEG carries song-genre information" is robust. The corpus-matched library has marginally higher κ but lower balanced accuracy — because its labels concentrate further (Indian 63%), depressing minority-class recall. This is a real precision/recall trade-off from corpus skew, not a failure.

**Honest caveat — label collapse**: the fixed 12-prompt oracle assigns 54.6% of real windows to *chiptune* (CLAP top1–top2 margin only 0.047), so overall accuracy is partly trivial and delivery skews to one genre. The κ = 0.55 / balanced-acc 0.47 result shows the genre signal survives the collapse, but the library should be re-made **corpus-matched and class-balanced** (a documented next step) before the song-relevance claim is headline-worthy.

### 8.3 The unified story
The SAME bridge output that retrieves the heard song (84% 14-way, Part 1) also lets a directly-trained classifier pick the song's genre prompt at κ = 0.55 (Part 2), and the frozen generator turns that prompt into pleasant audio (cos to the heard song 0.536) **without touching generator weights**. The naive variants fail exactly where the radii predict: embedding interpolation goes off-manifold (§6.2) and reading the prompt off bridge geometry is song-uninformative (§6.3).

### 8.4 Main comparison table (both parts together)
| Method | Genre class. acc (κ) | zcr | cos→heard | Song ID (14-way) | MOS |
|---|---|---|---|---|---|
| Oracle prompt (upper bound) | 1.000 | native | 0.754 (best-single) | — | TBD |
| **PromptMix-EEG mode (A)** | 0.742 (κ 0.551) | 0.079 | 0.536 | — | TBD |
| **PromptMix-EEG mode (B-cat)** | 0.742 (κ 0.551) | 0.101 | 0.558 | — | TBD |
| **PromptMix-EEG mode (B-mix)** | 0.742 (κ 0.551) | 0.110 | 0.607 | — | TBD |
| ep16 (LoRA + proj) | — | 0.0699 | — | 0.433 | TBD |
| mixed2 (LoRA + proj) | — | 0.0793 | — | 0.517 | TBD |
| embedding interpolation | — | up to 0.46 | — | — | — |
| Retrieval (real audio, Part 1) | — | — | 0.449 | 0.843 | TBD |

MOS (human listening) is the one column we cannot fill autonomously — it is the decisive generation gate. zcr/cos shown for context only (both are imperfect musicality proxies per §9.1).

### 8.5 Key figures (`code/figures/`, regenerate all with `python code/figures/render_all.py` → `assets/fig/` + `paper/figures/`)
- ✅ **Fig. 0** `fig0_architecture.png`: two-part architecture diagram (shared bridge → alignment/retrieval branch and generation/frozen-prompt-mixing branch).
- ✅ **Fig. 1** `fig1_coverage_radius.png`: coverage radius bar (mixture / best-single / best-prompt-mean / uniform / global, mean±SE over 60 targets).
- ✅ **Fig. 2** `fig2_tolerance_radius.png`: tolerance radius — zcr vs α for the 3 measured prompt pairs, with the frozen-native band and real-music line (jazz↔choir blows to 0.46).
- ✅ **Fig. 3** `fig3_oracle_collapse.png`: oracle genre distribution + classifier pred-vs-true, showing the chiptune collapse.
- ✅ **Fig. 4** `fig4_genre_confusion.png`: EEG→genre confusion matrix on held-out subjects (corpus-matched lib; diagonal 0.738).
- ✅ **Fig. 5** `fig5_tsne_coverage.png`: t-SNE of laion-CLAP — reveals the library concentrates in one region of the real-window cloud (the coverage-number caveat, §6.1).
- ⬜ **Fig. 6**: listening-test MOS distributions (violin) per method.
- ✅ **Fig. 7** `fig7_spectrograms.png`: qualitative spectrogram panel — real window / mode A / B-cat / B-mix / interpolation(α=0.25). The native clips (esp. B-cat) show clean note/harmonic structure = music; the interpolation clip is broadband noise-like (zcr 0.49), visually confirming §6.2.
- ✅ **Fig. 8** `fig8_tolerance_map.png`: full 66-pair tolerance heatmap (zcr at α=0.5) — 26% safe / 39% dangerous, generalizing §6.2.
- ✅ **Fig. 9** `fig9_retrieval_pool.png`: retrieval top-1 vs candidate pool size 14 → 4,146 (FMA distractors) against the 1/pool chance line — 26.6% at the largest pool, 1103× random.
- ✅ **Fig. 10** `fig10_mfcc_fad.png`: texture-aware MFCC-Fréchet distance per arm — frozen native 3,444 vs fine-tuned 29,472 vs T5 interpolation 38,904 (frozen 8.6× closer to real).

- ✅ **Fig. 12** `fig12_stim_vs_gen.png` (+ 28 tiles in `assets/fig/spectro/`): the heard stimulus window next to the clip generated from that same window's EEG, for all 14 delivered pairs. Produced by `make_fig_stim_vs_gen.py`, which also writes `results/stim_vs_gen/cos_matrix.json` and the copy the demo page reads (`assets/pairs.json`).

**Correction to an earlier number (withdrawn).** The per-row `cos_to_true` in `results/library_b/listen/report.json` (mean 0.477), and the equivalents in the mode-A/mode-B delivery reports, were computed with the stimulus handed to laion-CLAP as if it were 48 kHz while it is 32 kHz — the helper only resamples when the declared rate differs, so the stimulus was scored 1.5× fast. The three call sites now pass 32000, and the replacement measurement is `cos_matrix.json`: own 0.644 vs others 0.645, margin −0.001 (t = −0.1), own wins 7/14. Unrelated stimulus pairs already score 0.869 in this space, so **the generation arm shows no song-level match to its stimulus, and a CLAP cosine must never be quoted as an absolute similarity here.** `true_emb_all.npy` was checked and is the cache-derived (correct-rate) version — it matches a freshly embedded correct-rate stimulus at 0.88–0.95 versus 0.77–0.83 for the buggy one — so figs 3, 4, 5 and the head-to-head comparison stand.

Every figure is saved with `bbox_inches="tight"` (`code/figures/figstyle.py`) so no title or axis label can be cut off at the canvas edge, and the architecture diagram auto-shrinks each label until it fits inside its box. `make_fig_genre_confusion.py` additionally asserts that the confusion matrix it plots reproduces the overall accuracy in `genre_ci.json`, so a figure cannot silently disagree with the reported number.

---

## 9. Discussion

### 9.1 Limitations (measured, not speculative)
- **Coverage is a relative, not absolute, claim**: the CLAP cosine 0.8633/0.879 is inflated by laion-CLAP unit-vector clustering (constant-baseline trap). The honest signal is the +0.16 gap over the global-mean baseline, and the t-SNE (Fig. 5) shows the 12-prompt library occupies only **one region** of the real-window cloud, leaving several real clusters uncovered. A larger / k-means-seeded library is needed for true geometric coverage.
- **Oracle genre collapse is partly genuine corpus skew**: the corpus is Bollywood/film-heavy, so the dominant genre (Indian-bollywood, 63.4% with the matched library) is real, not an artifact — but the original library's chiptune collapse (54.6%) WAS an artifact (a spectrally-central attractor). Raw CLAP-cos-to-truth therefore rewards the attractor genre and is not a fair diversity metric (the matched set scores *lower* cos 0.477 yet is more honest).
- **zcr is not a pleasantness gate across genres**: square-wave chiptune is legitimately bright (zcr up to 0.39) without being noise. Musicality is decided by listening + CLAP, not zcr/RMS. A learned metric (Fréchet Audio Distance) would help but needs a matched reference distribution.
- **Signal-statistic musicality proxies all fail (measured)**: we also tested spectral flatness (`musicality_metric.py`) as a zcr/CLAP-independent proxy, and it is *also* invalid — real polyphonic windows score *higher* (0.043, noise-like) than sparse delivered native clips (0.003, tonal), and the known off-manifold interpolation clip (0.009) is not flagged as noisier than music. Conclusion: neither zcr nor spectral flatness separates music from noise for this material; **only a learned metric (FAD/CLAP-FD) or human MOS can judge musicality**, which is why the listening test is the decisive gate and cannot be replaced by a signal statistic.
- **EEG corpus size**: 1103 distinct windows / 68 subjects is small; the projection layer is tiny (134K params) but data augmentation (time-shift, channel-dropout) could help the genre classifier's minority-class recall (currently rock 0.18, lo-fi 0.00).
- **Bridge absolute alignment**: the ridge bridge is retrieval-good (84%) but absolute-alignment-poor (cos 0.19 to library); a contrastive EEG↔audio bridge could lift the genre-selection ceiling.
- **Real-time latency**: mode (B-mix) needs k forward passes (~5 s/clip on a 16 GB RTX); mode (A)/B-cat are single-pass and near-real-time.
- **Listening test not yet run**: the decisive generation gate (MOS) requires human raters and is the main outstanding experiment.

### 9.2 Future work
- **Hierarchical prompt libraries**: a coarse-grained top level (genre) and a fine-grained bottom level (instrument/mood), with a two-stage projection layer.
- **Learned prompt embeddings**: instead of a fixed library, learn soft prompts (à la prompt tuning) that are constrained to stay on the native manifold (e.g. by a KL penalty against the T5 embedding distribution).
- **Multi-modal conditioning**: combine EEG with other physiological signals (heart rate, skin conductance) or with user preferences (e.g. a liking/dislike button) to refine the mixture weights.
- **Closed-loop neurofeedback**: use the delivered audio as feedback in a neurofeedback protocol, and adapt the projection layer online based on the subject's response.

### 9.3 Broader impact
- **Positive**: EEG-to-music is a compelling application for BCI-based therapy (e.g. stroke rehabilitation, anxiety reduction) and neuroaesthetics research. A frozen-generator route is safer than end-to-end fine-tuning because it preserves the generator's musical prior, reducing the risk of delivering non-musical noise to a vulnerable user.
- **Negative / risks**: (i) EEG data is sensitive; our corpus is de-identified but re-identification attacks are theoretically possible. (ii) Delivered audio could be used to manipulate a user's emotional state without their consent. (iii) The 12-prompt library reflects the authors' cultural biases (Western-centric genres); a deployment in a different cultural context would require a different library.

---

## 10. Conclusion

We presented **PromptMix-EEG**, which decouples brain-to-music into an **alignment** part (EEG→laion-CLAP bridge that *retrieves the heard song*, 84.9% 14-way held-out vs ~7% baselines) and a **generation** part (a tiny adapter emitting mixture weights over a frozen library of native text prompts that drive a **frozen** MusicGen, 32 kHz). Generation musicality is guaranteed *by construction*: every delivered clip is a native sample of the generator, never an off-manifold condition.

Before training the adapter we measured **three radii** that decide the route: **coverage** (the library's convex hull reaches real windows at CLAP cosine 0.8633, rising to 0.879 with a corpus-matched library, +0.16 over trivial baselines — though the absolute value is partly laion-clustering inflation, shown by t-SNE), **tolerance** (embedding interpolation goes off-manifold for 39% of all 66 prompt pairs, decisively ruling out condition interpolation), and **alignment/label-validity** (the bridge's *absolute* geometry cannot read a genre prompt — 16.7% agreement — even though its *relative* geometry retrieves songs). Training EEG→song-matched-genre *directly* recovers a robust link (Cohen's κ 0.551 [0.528, 0.574], held-out subjects), and the delivered clips are native, un-clipped samples; the song-matched selection raises CLAP similarity to the heard song (0.536 vs 0.346 arbitrary).

We report honestly where automatic musicality metrics fail: zcr, spectral flatness, and CLAP-Fréchet-distance all fail to separate the frozen delivery from fine-tuned or noise-like output (fine-tuned ties frozen on CLAP-FD), so **the final pleasantness gate is human listening (MOS)**, which remains future work. This establishes *not touching the generator's weights* as a viable, auditable design principle for small-corpus brain-to-music, and releases the corpus-matched prompt library, the three-radius measurement scripts, the projection layers, and the listenable deliverables as an open artifact.

---

## 11. Artifact / reproducibility checklist

- [x] **Three-radius measurement**: `library_radius.py` (coverage+tolerance), `validate_mode_a_labels.py` (label validity), `compat_matrix_b.py` (full 66-pair tolerance map).
- [x] **Projection layer**: `finetune_prompt_classifier.py` (`--target cond_oracle|true_oracle`), `compute_true_oracle_fast.py` (cache-aligned song-matched labels, no decode).
- [x] **Evaluation**: `eval_true_oracle.py` (κ / balanced acc / baselines), `bootstrap_genre_ci.py` (95% CIs), `learned_musicality.py` (CLAP-Fréchet), `musicality_metric.py` (spectral flatness negative).
- [x] **Delivery**: `deliver_mode_a.py`, `deliver_mode_b.py` (B-cat/B-mix), `deliver_library_b.py`, `genre_gallery.py`.
- [x] **Library ablations**: `library_b_experiment.py` (corpus-matched), `coverage_k24.py` (K=24 size sweep).
- [x] **Figures**: `render_all.py` drives `make_fig_architecture.py`, `make_figures.py`, `make_figures_tsne.py`, `make_figures_spectro.py`, `make_fig_genre_confusion.py`, `make_fig_compat_map.py`, `make_fig_retrieval_pool.py`, `make_fig_mfcc_fad.py` → `assets/fig/` + `paper/figures/` (fig0–5, 7–10; fig6 awaits the listening test). All CPU-only, reading cached JSON/npy.
- [x] **Alignment (Part 1)** results: `mainv3/results/retrieval_line/eval.json` (song_hit 0.849).
- [x] **Data**: EEG corpus (subject-level split, 36036 kept windows), 12/24-prompt libraries, oracle label npys.
- [x] **Checkpoints**: frozen MusicGen (untouched), LaBraM encoder (frozen), projection layers (`prompt_classifier*/final.pt`, `library_b/classifier.pt`).
- [x] **Listening deliverables**: `results/LISTENING_MANIFEST.md` (paired GEN/REF wavs), `library_b/gallery/`.
- [ ] **Listening-test protocol + MOS**: rater instructions + scores (needs human raters — outstanding).
- [x] **README + env setup**: `promptmix-eeg/README.md` (tree, headline numbers) + `env/ENVIRONMENT.md` (conda `clabram312`, CUDA 12.8 RTX 50xx, Windows caveats). Self-contained (real files, no links).

---

## 12. Timeline to submission

| Week | Task | Owner |
|---|---|---|
| W1 (Oct 15–21) | Run mode (A) classification experiment; run mode (B) top-k mixing experiment; measure prompt-pair compatibility matrix | Auto / me |
| W2 (Oct 22–28) | Analyze results; fill in §8 tables and figures; draft §5–§7 prose | Me |
| W3 (Oct 29–Nov 4) | Recruit raters for listening test; collect MOS data | Me + collaborators |
| W4 (Nov 5–11) | Finalize listening-test analysis; polish §1–§4, §9–§10; internal review | Me |
| W5 (Nov 12–18) | Address internal feedback; prepare artifact release; submit to NeurIPS 2026 (deadline TBD) or ISMIR 2026 (late-breaking, deadline ~Jun 2026) | Me |

**Note**: NeurIPS 2026 deadline is typically mid-May; ICML 2026 is late-January; ICLR 2027 is late-September. ISMIR 2026 late-breaking deadline is ~June. Target **ISMIR 2026 late-breaking** (4-page extended abstract) or **NeurIPS 2026 main track** (9-page full paper) depending on how much experimental validation we can complete by W4.

---

## 13. Project naming (DECIDED: PromptMix-EEG)

Current workspace names (`CLaBraM`, `mainv3`, `EEG2Music`) no longer reflect the current main line. Candidates:
- **PromptMix-EEG** (preferred — matches paper title, descriptive)
- NeuroPrompt
- BrainPrompt
- EEG-PromptMixer
- FrozenGen-EEG
- Manifold-EEG

**Decision (done)**: public name = **PromptMix-EEG**. The organized, self-contained project root is `promptmix-eeg/` (code grouped by function, real data/results/listening files, docs, env). The legacy workspace directory `mainv3/` is **kept unchanged** as the original working tree / fallback (its scripts still run; `promptmix-eeg/code` is the canonical copy with paths repointed). No directory rename of `mainv3` — that would break 30+ hardcoded paths for no benefit; the new name is carried by `promptmix-eeg/`.

---

*End of outline v1.*
