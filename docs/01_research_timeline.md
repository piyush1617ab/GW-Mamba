# 01 — Research Timeline

154 notebooks, February 14 → May 25, 2026. This chapter walks through all seven phases in order, with the evidence (filenames, class names, dates, printed metrics) for each transition. Where a claim is based on a notebook I opened and read in full, it's stated as fact. Where it's inferred from filenames/dates across a large cluster I didn't individually re-verify (mainly inside the 66-notebook MD-GRTN phase), it's flagged as such.

---

## Phase 0 — Toy dataset prototyping
**Feb 14–22, 2026 · 18 notebooks · folder: `other models/model on small dataset/`**

The dataset here is `1st_dataset/traffic.csv` — 48,121 rows of `DateTime, Junction, Vehicles, ID`, hourly counts across a small number of road junctions. This is a standard small Kaggle traffic-counts dataset, not PEMS.

The first notebook, `1_traffic flow prediction.ipynb` (Feb 14, 23:29 — the earliest timestamp in the whole corpus), builds an LSTM directly with Keras-style layers (comments: *"building the lstm model"*, *"First LSTM layer with 64 memory units"*) and reaches MAE ≈ 16–19, RMSE ≈ 22–26 on this dataset.

From notebook 3 onward, a naming convention appears that's worth clarifying up front: **"LLM" in these filenames does not mean large language model.** It's Piyush's own label for a Transformer/attention-based sequence block — confirmed directly from code: `3_traffic_flow_pred_llm.ipynb` defines `class TimeSeriesLLM` with a comment *"TimeSeries LLM model"*, built from `PositionalEncoding` + `TransformerEncoder`. Later notebooks extend the same convention to other backbones — `CNN_LLM`, `GRU_LLM`, `LSTM_LLM` (notebook 7), i.e. "backbone + attention block," not an actual language model integration.

Through notebooks 4–7, the small dataset is sliced differently (per-junction vs. global), and hybrid CNN/GRU/LSTM + attention variants are tried. Notebook 6 (`6_traffic_flow_gnn.ipynb`) is the **first GNN attempt** (`GNNLayer`, `GNNModel`). Note: MAE on notebooks 6–7 drops to the 1–4 range — this is *not* a 5–10x model improvement over notebook 1; it reflects a different, lower-magnitude data slice (junction-level counts vs. the full series), not an apples-to-apples comparison. This is the first of several places in the corpus where raw MAE isn't comparable across notebooks without checking what data/scale it was computed on.

From notebook 8 onward, filenames start referencing `pems08` even though the files are still in the "small dataset" folder — the first contact with the eventual target benchmark, still using the small-dataset harness. Notebook 19 (`19_traffic_flow_gnn+llm_v3.ipynb`, Feb 22) introduces `AdaptiveGraphConv` — the first appearance of a *learned* (as opposed to fixed) adjacency matrix, an idea that becomes central much later in Graph WaveNet's self-adaptive adjacency.

Note: notebook **11** does not exist in the archive — a gap in the numbering, not a missing file on our end (confirmed against the full zip listing).

**Why the phase ends here:** a 9-day gap follows (Feb 22 → Mar 3), and the next notebooks move to a new top-level folder (`other models/`, not `.../small dataset/`) with clean, PEMS08-only filenames starting at `21_`. That's a deliberate checkpoint — graduating from toy data to the real benchmark.

---

## Phase 1 — PEMS08 direct baselines
**Mar 3–5, 2026 · 8 notebooks**

Numbered 21–27 (with a couple of parallel `21_` branches), all on PEMS08 directly:

| # | Notebook | What changed |
|---|---|---|
| 21 | `..._lstm_colab_gpu (1)` | LSTM baseline, moved to Colab GPU |
| 21 | `..._v3_llm+lstm_fix_overfit (1)` | Overfitting fix branch off the same LSTM |
| 22 | `..._fixed_lstm+llm_v2 (1)` | Second overfitting-fix iteration |
| 23 | `..._v4_TCN (1)` | **TCN introduced** (`TemporalBlock`) |
| 24 | `..._v5_persensor (1)` | **Per-sensor normalization introduced** — see [`03_key_innovations.md`](03_key_innovations.md) |
| 25 | `..._v6_fixed (1)` | Fix pass on the per-sensor version |
| 26 | `..._CNN_LSTM_v2` | CNN+LSTM hybrid |
| 27 | `..._GCN_Transformer (1)` | **First graph-based model** (`GCNTransformer`) |

Notebook 27 is the real pivot of this phase: the first time a graph convolution enters the pipeline on the real benchmark, setting up everything that follows.

---

## Phase 2 — The MD-GRTN reproduction effort
**Mar 13 – Apr 11, 2026 · 66 notebooks (43% of the entire corpus)**

By notebook count, this is the single largest phase — nearly as many notebooks as every other phase combined. MD-GRTN is the published baseline being targeted throughout the project; its reported PEMS08 numbers (printed as a literal constant in dozens of later notebooks) are **MAE = 13.114, RMSE = 22.623, MAPE = 8.471%**.

**What MD-GRTN's architecture actually is**, per `mdgrtn_phase1.ipynb` (Mar 13, the first reproduction attempt): a `BackNet` (an MLP "denoiser," commented as the *backward process* of a diffusion-style scheme) feeding an `MDModule` → `MDAFModule` ("Multi-period Diffusion Attention Fusion" — the comment literally spells this out), followed by an `MGRCModule` and spatial/temporal Transformer layers. "Diffusion" here means denoising in the diffusion-model sense, not graph diffusion convolution — an important distinction that matters for the next paragraph.

**The reproduction-and-search sub-arc** (evidence: filenames + class-name deltas across the cluster):
- Mar 13–20: `mdgrtn_phase1` → `phase2` → `phase3` → `final` → `phase1_v2/v3` — early phase-numbered iterations.
- Mar 23: `mdgrtn_paper_exact.ipynb` — an explicit attempt at a literal, by-the-book reproduction.
- Mar 24–26: the **`nodiff` ablation** — `BackNet` and `MDModule` (the diffusion-denoiser) are removed; `MDAFModule` becomes plain `MAFModule`, and a new `MultiGraphFusion` module replaces it. This directly tests whether the denoising step is pulling its weight. Detail in [`04_failed_experiments_and_debugging.md`](04_failed_experiments_and_debugging.md).
- Mar 26–28: the **GAT swap** — `GCN_GRU_Layer` is replaced with `GAT_GRU_Layer` / `GraphAttention` across a run of `phase1_v2 gat`, `phase1_v3 2seq gat` variants, alongside sequence-length experiments (`2seq` vs `3seq` — how many historical periods feed the multi-period fusion module).
- Mar 28 – Apr 3: fast version churn, `v4` through `v21`, converging on `AdjacencyFusion` + `GATLayer` as the stable spatial block.
- **Apr 3, `mdgrtn_v14_14_9_current_best.ipynb`** — explicitly marked as the best checkpoint at the time. Its training log is real and legible: val MAE starts at 20.334 (epoch 1) and improves to 16.417 by epoch 15 — genuine convergence, still well above the 13.114 target.
- **Mar 30, `mdgrtn-v11 actual.ipynb`** — has a full paper-style final test block: **test MAE 15.164 vs. baseline 13.114 (Δ = +2.050, still behind)**, RMSE 24.561 vs 22.623, MAPE 8.543% vs 8.471%.
- Apr 1–7: `mdgrtn_gwn.ipynb` and the `mdgwnet-*` notebooks start blending MD-GRTN ideas with Graph WaveNet components — the first sign of the two lines cross-pollinating.
- Apr 4, `pems08_beat_mdgrtn.ipynb` and Apr 11, `superior_mdgrtn_beater.ipynb` / `_vram_fixed` — explicitly named attempts to close the gap, ending with a VRAM fix (the smallest file in the cluster, `mdgrtn_beater_user_paths.ipynb` at 6KB, looks like a final path-configuration tweak rather than a new architecture).

**Across every MD-GRTN notebook I opened and checked in depth, none reached the 13.114 MAE baseline outright** — the closest documented test-set result is the 15.164 above. I did not individually re-open all 66 notebooks to this level of depth (the appendix catalogs each by the signals the extraction pass found), so this is reported as "true of the notebooks inspected," not as an exhaustive claim about all 66.

**Why the phase (eventually) ends:** the return on continued MD-GRTN-specific tuning was flattening out relative to a competing line of work — Graph WaveNet, developing in parallel — that reached comparable numbers with a structurally simpler model (next phase).

---

## Phase 3 — The Graph WaveNet family
**Apr 4–16, 2026 · 21 notebooks · overlaps Phase 4**

`gwnet-v18.ipynb` (Apr 4, 21:50) is the first Graph WaveNet notebook: `DiffusionGCN` + `WaveBlock` + `GWNet`. It's also the **first notebook in the whole corpus to beat the MD-GRTN baseline on any metric**: MAPE 8.371% vs. baseline 8.471% (Δ = −0.100%, a win), even though MAE (14.923 vs 13.114) and RMSE (24.023 vs 22.623) were still behind.

From there: `v20` through `v29` (Apr 4–8, including `-o`-suffixed branches that look like optimizer/config sweeps), then GPU-specific variants `gwnet-rtx1/2/3` (Apr 8–9), explicit ablations `gwnet_ab1/2/3` (Apr 11–12), and `gwnet-vk1/2/3` (Apr 16) adding `AdaptiveTemporalPool` and `DropPath` regularization.

**`gwnet_v21_strong_full.ipynb`** (Apr 11) is generated by `generate_full_ipynb.py` — a Python script that programmatically builds an entire notebook as JSON rather than hand-editing cells (see the callout in [`07_lessons_learned.md`](07_lessons_learned.md)). Its config is the clearest single snapshot of "mature Graph WaveNet" in this project: `d_model=96`, `n_layers=10`, dilations cycling `[1,2,4,8,1,2,4,8]`, **`n_supports=3`** (two fixed forward/backward diffusion supports plus one learned adaptive adjacency — this is the "three-support graph convolution" that survives into the final architecture), sparse top-k graph construction (`topk_graph=16`), and a composite MAE+Huber+RMSE loss with EMA weights and a OneCycleLR schedule.

`mdgwnet-v18/21/117` (Apr 7) hybridize MD-GRTN's multi-period fusion idea onto the GWNet backbone — a bridge between Phase 2 and Phase 3 rather than a clean break.

---

## Phase 4 — Wide architecture search
**Apr 6–21, 2026 · 30 notebooks · overlaps Phase 3**

With Graph WaveNet established as a strong backbone, this phase casts a wide net of named architectures, mostly on PEMS08:

`STID` (Apr 8) · `TI-GNet` with its `TING` module (Apr 8, see [`03_key_innovations.md`](03_key_innovations.md)) · `STAGNet`, `STGTNet`, `STWaveNet` (Apr 6–8) · `MSDTN` / `MSDTN_GWNNet` (Apr 11) · `MPSTF` + fixed (Apr 11) · `PASTANet` (Apr 12) · `GWGRUNet`, `WaveGRU`, `WaveST` (Apr 12–13) · `PEMS08Net` / `V2` (Apr 13–14) · `final_model.ipynb` (Apr 14 — named `FinalModel`, superseded within days, a reminder that "final" in a filename is a snapshot, not a guarantee) · `STAEformerPlus` (Apr 14) · `GWSTNet` (Apr 15) · the `GATSTCNet` family, including the OOM crash and fix (Apr 18–19, detailed in [`04_failed_experiments_and_debugging.md`](04_failed_experiments_and_debugging.md)) · **`FUSION`** (Apr 19 — see below) · `MPSTAEformer`, `MGSTformer` (Apr 20) · `MSTAGWNet` + debugged (Apr 20) · `GWMPCNet` (Apr 21) · **`PI_GWM_PEMS08`** (Apr 21, 20:13).

Two notebooks in this phase matter more than the rest for what comes next:

**`fusion_traffic.ipynb`** (Apr 19) defines `class FUSIONModel`, whose own docstring states the design directly: *"FUSION: GWNet × STAEformer × MD-GRTN."* It takes three separate input windows (24-step recent, 12-step hourly context, 12-step daily context) and combines all three prior architecture families into one model. This is the "combine everything" hypothesis, tested explicitly before being set aside — see [`04_failed_experiments_and_debugging.md`](04_failed_experiments_and_debugging.md) for why.

**`PI_GWM_PEMS08.ipynb`** (Apr 21, 20:13) is the physics-informed line — `CurriculumLoss`, a physics-derivative input feature (detailed in the next chapter), and, notably, it **already contains `SelectiveSSM` and `TemporalMambaBlock`** — 46 minutes before the dedicated GW-Mamba notebook appears. The Mamba idea was born inside this physics-informed experiment, then extracted into its own line almost immediately.

---

## Phase 5 — GW-Mamba emerges
**Apr 21–May 16, 2026 · 7 notebooks (still in `other models/`)**

`GW_Mamba_PEMS08.ipynb` (Apr 21, 20:59) is the first notebook dedicated to the Mamba+GWNet combination: `DiffusionGCN` + `WaveBlock` (kept from Graph WaveNet) + `MambaRefinementBlock` + `SelectiveSSM` (new). `GW_Mamba_v3_PEMS08.ipynb` (Apr 24) adds a bidirectional variant, `BiMambaBlock`, plus `GlobalSpatialTransformer` and `MultiPeriodFusion`.

Then a roughly three-week gap, after which `GW_Mamba_PEMS03/04/07.ipynb` (May 14) and `GW_Mamba_ALL4_PEMS.ipynb` (May 16) appear — generalization testing across all four PEMS datasets, still in the working folder rather than a dedicated one.

---

## Phase 6 — Final validation
**May 13–25, 2026 · 4 notebooks · folder: `gw mamba/`**

A dedicated folder appears, and the architecture is frozen. `gw-mamba-pems08_baseline.ipynb` (May 13) carries the comment *"Architecture IDENTICAL to PEMS08 baseline"* into the dataset-specific notebooks that follow — confirmation, in the code itself, that no further architecture changes happen from here on; only re-training and re-evaluation per dataset. `GW_Mamba_PEMS04.ipynb` and `gw-mamba-pem3.ipynb` (both May 23), then `gw-mamba-pem07.ipynb` (May 25 — the last-dated notebook in the entire corpus) close out the project.

Full results in [`06_results_across_datasets.md`](06_results_across_datasets.md).

---

## Summary table

| Phase | Dates | Notebooks | Headline transition |
|---|---|---|---|
| 0 — Toy dataset | Feb 14–22 | 18 | LSTM/GRU/CNN + attention hybrids, first GNN attempt |
| 1 — PEMS08 baselines | Mar 3–5 | 8 | LSTM → TCN → per-sensor norm → first graph model |
| 2 — MD-GRTN reproduction | Mar 13–Apr 11 | 66 | Denoise ablation, GAT swap, closest test MAE 15.164 vs. 13.114 target |
| 3 — Graph WaveNet | Apr 4–16 | 21 | 3-support diffusion conv; first metric win (MAPE) vs. baseline |
| 4 — Wide search | Apr 6–21 | 30 | STID/TI-GNet/FUSION/physics-informed; Mamba born inside PI_GWM |
| 5 — GW-Mamba emergence | Apr 21–May 16 | 7 | Dedicated Mamba+GWNet line, then bidirectional variant, then 4-dataset testing |
| 6 — Final validation | May 13–25 | 4 | Architecture frozen; final numbers per dataset |

**154 notebooks total.**
