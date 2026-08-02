# 03 — Key Innovations

Six ideas recur across the corpus and carry weight in the final model. For each: where it came from, what problem it addresses, and — honestly — how strong the evidence for its individual contribution actually is. Not every idea here has a clean isolated ablation in the notebooks; where that's true, it's stated directly rather than implied.

---

## 1. Per-sensor normalization

**Origin:** `24_traffic_flow_pems08_v5_persensor (1).ipynb`, Mar 4 — the filename names the change directly.

**Problem it addresses:** PEMS08 has 170 sensors on different road segments — a highway sensor and a side-street sensor see very different traffic volumes. A single global mean/std computed across all sensors and all time steps under-normalizes high-volume sensors and over-normalizes low-volume ones.

**Implementation** (verified from code):
```python
# Per-sensor normalisation — one mean & std per sensor, shape (170,)
sensor_mean = train_raw.mean(axis=0)
sensor_std  = train_raw.std(axis=0) + 1e-8
flow_norm = (flow - sensor_mean) / sensor_std
```
Computed once per sensor (axis=0), fit only on the training split, then broadcast over time — standard practice, correctly implemented (no leakage from val/test into the fit).

**Evidence of improvement:** the motivation is sound and the implementation is correct, but the notebooks don't contain a clean isolated ablation (same architecture, global-norm vs. per-sensor-norm, all else equal) that I found. Per-sensor normalization appears from notebook 24 onward and is never removed in any later architecture — it becomes a permanent fixture of the preprocessing pipeline — which is suggestive but not a substitute for a controlled comparison. Treat this as "adopted and never revisited," not as "measured and quantified."

---

## 2. Physics-informed derivative feature

**Origin:** `PI_GWM_PEMS08.ipynb`, Apr 21 (`PIGWM` = Physics-Informed Graph WaveNet-Mamba).

**What it actually is** — worth being precise, since "physics-informed" can imply something more elaborate than what's here: a first-order finite difference of the (already per-sensor-normalized) flow channel.
```python
diff = np.zeros_like(flow_n)
diff[1:] = flow_n[1:] - flow_n[:-1]      # Δflow; diff[0] = 0
self.features = np.concatenate([data_normed, diff], axis=-1)
```
This is not a PDE-constrained loss term or a physical conservation law baked into the architecture — it's a rate-of-change feature appended as an extra input channel. The framing ("physics-informed") is reasonable in the loose sense that traffic-flow rate-of-change carries physically meaningful information (acceleration/deceleration of the traffic stream), but the implementation is a simple, cheap feature-engineering addition, not a physics-constrained model.

**Evidence of improvement:** it enters the pipeline in `PI_GWM_PEMS08.ipynb` and persists into the final `GWMamba` — the model's `start_conv` explicitly takes `in_features + 1` with the code comment *"physics derivative appended in Dataset"*. As with per-sensor normalization, I found no isolated with/without ablation for this specific channel. It's a kept, low-cost feature, not a measured one.

---

## 3. Three-support diffusion graph convolution

**Origin:** Graph WaveNet family, stabilizes by `gwnet_v21_strong_full.ipynb` (Apr 11), `n_supports=3`.

**What the three supports are** (from `GWMamba.get_supports()` in the final architecture):
1. `A_fwd` — a fixed forward random-walk transition matrix, built from the sensor distance graph.
2. `A_bwd` — the corresponding backward/transpose transition matrix.
3. `A_adp` — a **learned** adaptive adjacency: `softmax(relu(E1 @ E2.T))`, where `E1`/`E2` are trainable per-node embedding matrices.

This is the standard Graph WaveNet self-adaptive-adjacency design: two fixed, direction-aware diffusion supports (capturing the known physical road topology) plus one fully learned support (able to pick up correlations the physical graph doesn't encode — e.g., sensors that behave similarly at rush hour despite not being adjacent). Each support runs its own order-2 diffusion (`order=2` in `DiffusionGCN`), and results are concatenated before projection.

**Evidence of improvement:** unlike features 1 and 2, this one has *directional* evidence, if not a clean isolated ablation: `gwnet_ab1/ab2/ab3.ipynb` (Apr 11–12) are explicitly named ablation notebooks in the Graph WaveNet line, consistent with this configuration being deliberately tested against simpler alternatives. I did not open all three ablation notebooks to extract their specific deltas — flagged here rather than assumed. What's solidly confirmed is that `n_supports=3` is the configuration that ships in every Graph WaveNet and GW-Mamba notebook from `gwnet_v21_strong_full` onward, including the final frozen architecture.

---

## 4. TING — Temporal Identity Noise Gate

**Origin:** `tignet-pems.ipynb`, Apr 8, part of the TI-GNet line.

**What it does**, from the module's own docstring:
> Denoises a noisy input stream using noise-free temporal + spatial embeddings.
> `gate = sigmoid(Linear([s_emb ‖ tod_emb ‖ dow_emb]))`
> `prior = Linear([tod_emb ‖ dow_emb])` — a noise-free temporal estimate
> `out = gate · proj(x_noisy) + (1 − gate) · prior`

The idea: a sensor's identity (which sensor it is) and the time (hour of day, day of week) are known with certainty and are noise-free, even when the actual traffic reading is noisy or missing. TING learns a gate that blends the raw noisy measurement with a "prior" estimate derived purely from those clean identity signals — effectively letting the model fall back on "what traffic usually looks like at this sensor at this time" when the measurement itself is unreliable.

**Evidence of improvement:** this is the most rigorously *planned* ablation in the entire corpus — the notebook contains an explicit 6-variant ablation scaffold (`w/o TING`, `w/o GCN`, `w/o t-MLP`, `w/o s-MLP`, `w/o hour`, full model), with the code comment *"required for a publishable paper"* and a dedicated `TIGNetNoTING` class (TING replaced by a plain linear projection). **However, the notebook has no saved execution outputs** — no error cells, but also no metric lines and no final output captured. The ablation was designed correctly but I found no evidence it was actually run and recorded in this file. TING itself doesn't carry forward into `GWMamba` by name, but its core idea — blending a noisy signal with a prior derived from clean identity embeddings — is conceptually close to how the final model's TOD/DOW embeddings are used (fixed, clean per-timestep identity signals combined with the learned spatial-temporal representation).

---

## 5. Curriculum loss (Huber → MAE)

**Origin:** `PI_GWM_PEMS08.ipynb`, Apr 21, alongside the physics-informed feature.

**Implementation:**
```python
class CurriculumLoss(nn.Module):
    def __init__(self, warmup_epochs=5, huber_delta=1.0):
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mae = nn.L1Loss()
    def forward(self, pred, target):
        return self.huber(pred, target) if self.current_epoch <= self.warmup_epochs else self.mae(pred, target)
```
Train with Huber loss (quadratic near zero, linear for large errors — robust to the noisy/outlier readings common early in training when predictions are far off) for a 5-epoch warm-up, then switch to plain MAE — which is both the actual evaluation metric and what the earlier `gwnet_v21_strong_full` composite loss (`MAE + Huber + RMSE`, blended throughout training rather than phased) was already approximating with a fixed weighted sum. Curriculum loss reframes that as a two-phase schedule instead of a constant blend.

**Evidence of improvement:** no isolated ablation (same architecture, curriculum vs. fixed-loss, all else equal) found in the notebooks. The design logic is sound — warm up robustly, then optimize the actual metric directly — but this is a motivated design choice, not a measured one, in the material available.

---

## 6. Mamba refinement block

**Origin:** first appears inside `PI_GWM_PEMS08.ipynb` (Apr 21, 20:13) as `SelectiveSSM` + `TemporalMambaBlock`, then gets its own dedicated line 46 minutes later in `GW_Mamba_PEMS08.ipynb`.

**What it does:** a single `MambaRefinementBlock`, applied once after the full stack of WaveBlocks, operating on the accumulated skip-connection representation (not on the raw input, and not repeated per-layer):
```
LayerNorm → input projection (split into gate z, pass-through g)
  → depthwise causal 1D conv on z → SiLU
  → SelectiveSSM(z)   # discretized selective state-space scan over time
  → z * SiLU(g)  → output projection → residual add
```
`SelectiveSSM` implements a simplified selective scan: input-dependent step size (`Δt`, via a softplus-activated linear projection), per-channel state transition matrices, and a sequential recurrence over the time axis — the core Mamba mechanism, at a scale appropriate for this problem (`d_state=8`) rather than a full language-model-scale implementation.

**Why add this to a Graph WaveNet backbone specifically:** WaveBlock's dilated convolutions have a large but *finite* receptive field (the `[1,2,4,8,1,2,4,8]` dilation schedule covers the full 24-step input, but each output position still aggregates through a fixed, architecture-determined window). A selective SSM scan processes the entire sequence with an input-dependent state update, which is a structurally different way of modeling long-range dependence — added here as a refinement layer rather than a replacement for the graph-temporal backbone.

**Evidence of improvement — the closest thing to a real before/after in the whole corpus:**

| Notebook | Date | Model | Test MAE | vs. MD-GRTN baseline (13.114) |
|---|---|---|---|---|
| `gwnet-v18.ipynb` | Apr 4 | GWNet (no Mamba) | 14.923 | +1.809 |
| `gw-mamba-pems08_baseline.ipynb` | May 13 | GWNet + Mamba refinement | 13.494 | +0.380 |

A ~1.4 MAE improvement between an early Graph WaveNet checkpoint and the final Mamba-refined model. **This is directional evidence, not a controlled ablation** — five weeks and many other changes (per-sensor norm already present by then, physics-derivative feature, curriculum loss, further hyperparameter tuning, longer/different training runs) separate these two notebooks, so the gap can't be attributed to the Mamba block alone. No notebook in the corpus runs the *exact* same final config with the Mamba block removed as a controlled comparison. Framed honestly: the project's overall trajectory (all improvements combined, Mamba included) closed about 79% of the gap to the MD-GRTN baseline (from +1.809 down to +0.380 MAE), and the Mamba refinement was the last major architectural change made before that final number was reached.
