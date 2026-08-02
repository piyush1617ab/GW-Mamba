# 02 — Architecture Families

Nine distinct architectural approaches show up across the corpus (by code-verified keyword and class-name search, not filename guessing — see the frequency table at the end of this chapter). This chapter profiles each: why it was tried, what it was good at, what broke, and what — if anything — survived into the final model.

---

## 1. LSTM / GRU / CNN + attention ("LLM") hybrids
**Where:** Phase 0–1 · **Kept in the end:** No, but set the harness everything else reuses.

The starting point for any sequence-forecasting problem. Plain LSTM (`1_traffic flow prediction.ipynb`) established the data-loading and training harness (`TrafficDataset`, sequence windowing) that essentially every later notebook reuses in some form. Attention was added almost immediately (notebook 3's `TimeSeriesLLM`) — reasonable, since even a small Transformer block gives the model a way to weigh different historical timesteps instead of compressing everything through a fixed-size recurrent state.

**Problem:** none of these have any notion of *space*. PEMS08 is 170 sensors on a real road network — adjacent sensors are correlated, and a model that treats each sensor as an independent series (or flattens them into one undifferentiated vector) throws that away. This is the reason the project moves to graph-based models as soon as it touches the real benchmark (Phase 1, notebook 27).

## 2. TCN (Temporal Convolutional Networks)
**Where:** Phase 1 (`23_..._v4_TCN`) · **Kept in the end:** Yes, structurally — as dilated causal convolution inside every WaveBlock-based model from here on.

`TemporalBlock`-style dilated causal convolutions are a cheaper, more parallelizable alternative to recurrence for capturing multi-scale temporal patterns. This one experiment is short-lived as a standalone architecture, but the *technique* (dilated causal conv) is exactly what Graph WaveNet's `WaveBlock` is built from two phases later — so in practice, TCN never gets fully abandoned, it gets absorbed.

## 3. GCN / GAT (graph convolution and graph attention)
**Where:** First appears Phase 1 (`27_..._GCN_Transformer`), then contested throughout Phase 2 · **Kept in the end:** GCN-style diffusion convolution, yes (as `DiffusionGCN`); GAT, no.

Two competing ways to aggregate a sensor's neighbors: plain graph convolution (fixed weights from the adjacency matrix) vs. graph *attention* (learned, input-dependent weights per edge). The MD-GRTN line runs this contest directly — `GCN_GRU_Layer` is swapped for `GAT_GRU_Layer`/`GraphAttention` around Mar 26. GAT is more expressive per edge but far more expensive: it's exactly the GAT-based attention matrix that later causes the OOM crash in the GATSTCNet family (`(B,H,4080,8160)` — see [`04_failed_experiments_and_debugging.md`](04_failed_experiments_and_debugging.md)). The version that ultimately survives (Graph WaveNet's `DiffusionGCN`) uses fixed forward/backward diffusion matrices plus one *learned adjacency*, which gets most of GAT's flexibility (the adjacency itself is learned) without paying attention's quadratic cost per forward pass.

## 4. MD-GRTN (the baseline being targeted)
**Where:** Phase 2, 66 notebooks · **Kept in the end:** The multi-period framing (recent/hourly/daily context), no; specific modules, no; the paper's numbers as a permanent comparison target, yes — through the very last notebook in the corpus.

MD-GRTN combines a diffusion-style denoiser (`BackNet`, an MLP "backward process"), multi-head attention fusion across multiple time periods (`MDAFModule`), a GRU/GCN spatial-temporal backbone (`MGRCModule`), and Transformer layers on top. It's a legitimate, capable architecture — six weeks of iteration on it (Phase 2) is the largest single time investment in the whole project — but it's also structurally heavy (five-plus distinct submodules stacked), and the `nodiff` ablation (removing the denoiser) suggests some of that weight wasn't earning its keep (detail in the next chapter's failed-experiments discussion).

Its real legacy in this project isn't architectural — it's the **target number**. `MAE=13.114, RMSE=22.623, MAPE=8.471%` on PEMS08 gets printed as a literal comparison constant in dozens of notebooks all the way through the final `gw-mamba-pems08_baseline.ipynb`. Every later architecture decision in this project was implicitly made against this yardstick.

## 5. Graph WaveNet (GWNet)
**Where:** Phase 3, 21 notebooks, first appears Apr 4 · **Kept in the end:** Yes — this is the backbone of the final model.

Dilated causal convolution (from the TCN experiment) stacked with diffusion graph convolution and a **self-adaptive adjacency matrix**: two node-embedding matrices whose product, softmax'd, produces a learned graph alongside the two fixed (forward/backward) diffusion supports. `gwnet-v18` is the first notebook in the entire corpus to beat the MD-GRTN baseline on *any* metric (MAPE). It's simpler than MD-GRTN (four core modules: `DiffusionGCN`, `WaveBlock`, the adaptive-adjacency embeddings, and an output head) and reaches comparable numbers — the first sign that the project's next real gain would come from refining this architecture rather than adding more submodules to MD-GRTN.

## 6. The wide-search zoo (STID, TI-GNet, STAGNet, GATSTCNet, and others)
**Where:** Phase 4, ~25 named variants across 30 notebooks · **Kept in the end:** One module (TING) migrates conceptually into the final model's identity-embedding design; the rest don't survive as standalone lines.

With GWNet established as a strong, simpler backbone, this phase tests a wide range of published ideas against it in quick succession — spatial-MLP-based STID, TI-GNet's gated denoising (`TING`, see next chapter), GAT+TCN hybrids (GATSTCNet), multi-scale variants (MSDTN), and several others. None of these individually beat what GWNet was already achieving by this point, but they're not wasted motion — TING's approach to fusing "clean" identity signals (time-of-day, day-of-week, sensor ID) with a noisy measurement stream is conceptually close to how the final `GWMamba` handles its own TOD/DOW embeddings, even though the literal `TING` class isn't reused.

## 7. FUSION (GWNet × STAEformer × MD-GRTN)
**Where:** Phase 4, Apr 19 · **Kept in the end:** No.

The explicit "combine everything" hypothesis — `FUSIONModel`'s own docstring names the three architectures it merges. It takes three separate input windows (recent/hourly/daily) and routes them through components borrowed from all three prior families. This is the most structurally ambitious model in the corpus, and it's set aside within two days of being written (`fusion_traffic.ipynb` → `fusion_traffic_fixed.ipynb`, Apr 19, then no further iteration). The project's direction from here is the opposite of FUSION's premise: not "combine every good idea," but "take the single strongest backbone (GWNet) and add one well-targeted refinement."

## 8. Physics-informed features + curriculum loss
**Where:** Phase 4, `PI_GWM_PEMS08.ipynb`, Apr 21 · **Kept in the end:** Yes, both — they carry directly into `GWMamba`.

"Physics-informed" here is intentionally simple and worth being precise about: it's a first-order finite difference of the flow signal (`diff[t] = flow[t] − flow[t−1]`), appended as an extra input channel — not a PDE-constrained loss or anything more elaborate. `CurriculumLoss` is equally simple and equally effective in design: train with Huber loss (robust to noisy/outlier readings) for a short warm-up (default 5 epochs), then switch to MAE (matching the actual evaluation metric) for the rest of training. Both ideas are small, well-motivated, and — notably — this is also the notebook where `SelectiveSSM` and `TemporalMambaBlock` first appear, 46 minutes before the dedicated GW-Mamba notebook. Full detail in [`03_key_innovations.md`](03_key_innovations.md).

## 9. Mamba / Selective State-Space Models
**Where:** Phase 5 onward · **Kept in the end:** Yes — this is the second half of the final model's name.

A single `MambaRefinementBlock` (LayerNorm → gated input projection → depthwise causal conv → SiLU → selective SSM scan → gated output) applied once, after the WaveBlock stack, to the accumulated skip connections. This isn't Mamba replacing the graph-temporal backbone — it's a lightweight sequential refinement layer added on top of it, targeting the kind of long-range temporal dependency that dilated convolution's finite receptive field can under-serve. `GW_Mamba_v3` (Apr 24) tries a bidirectional version (`BiMambaBlock`) before the project settles back to the unidirectional design for the final architecture.

---

## Frequency across the corpus

Code-verified mentions (word-boundary matched against actual notebook source, not filename guessing):

| Keyword | Notebooks | Keyword | Notebooks |
|---|---|---|---|
| Transformer | 97 | Graph WaveNet | 23 |
| MultiHeadAttention | 88 | TransformerEncoder | 19 |
| GCN | 83 | LSTM | 17 |
| GRU | 74 | Mamba | 12 |
| MDGRTN / MD-GRTN | 53 / 36 | SSM | 12 |
| GWNet | 49 | STAEformer | 11 |
| TCN | 44 | Physics-informed | 11 |
| GAT | 44 | CNN | 8 |
| dilated (conv) | 43 | curriculum | 5 |
| WaveNet | 37 | STID | 4 |

Transformer/attention components show up across nearly every family (97 notebooks) — not because Transformers were the main architecture, but because attention layers get used as a component inside almost everything (spatial attention, temporal attention, period-fusion attention), independent of whatever the "headline" architecture of a given notebook is.
