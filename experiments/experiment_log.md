# Experiment Log — Milestones

The full 154-notebook catalog is in [`docs/appendix_notebook_catalog.md`](../docs/appendix_notebook_catalog.md). This table is the narrower "spine" of the project: the notebooks that mark an actual architectural or methodological turning point, each with metrics verified directly from saved notebook output (not estimated). "Baseline" is MD-GRTN's published PEMS08 numbers (MAE 13.114 / RMSE 22.623 / MAPE 8.471%) unless noted.

| # | Notebook | Phase | Architecture | Major change | Metric (MAE) | vs. baseline | Decision |
|---|---|---|---|---|---|---|---|
| 1 | `1_traffic flow prediction.ipynb` | 0 | LSTM | First notebook; establishes data harness | 17.15–18.44 (toy data) | n/a (different dataset) | Kept harness, dropped architecture |
| 2 | `3_traffic_flow_pred_llm.ipynb` | 0 | Transformer (`TimeSeriesLLM`) | First attention-based model | 14.02 (toy data) | n/a | Attention kept as a recurring component |
| 3 | `6_traffic_flow_gnn.ipynb` | 0 | GNN | First graph-based attempt | 2.17–2.62 (toy data, different scale) | n/a | Confirmed spatial modeling is worth pursuing |
| 4 | `23_..._v4_TCN (1).ipynb` | 1 | TCN | First dilated causal conv on PEMS08 | 16.59 | +3.48 | Dilated conv absorbed into WaveBlock later |
| 5 | `24_..._v5_persensor (1).ipynb` | 1 | LSTM + per-sensor norm | Per-sensor normalization introduced | 18.51 (mean per-sensor) | — | Kept permanently in preprocessing |
| 6 | `27_..._GCN_Transformer (1).ipynb` | 1 | GCN+Transformer | First graph convolution on PEMS08 | 18.86 | +5.75 | Pivot point into graph-based modeling |
| 7 | `mdgrtn_phase1.ipynb` | 2 | MD-GRTN (full) | First MD-GRTN reproduction | 23.63 (early epoch) | — | Baseline architecture established |
| 8 | `mdgrtn_nodiff.ipynb` | 2 | MD-GRTN, denoiser removed | Ablation: is `BackNet` needed? | not captured | — | Pursued in parallel; outcome inconclusive from saved output |
| 9 | `mdgrtn-v11 actual.ipynb` | 2 | MD-GRTN + GAT | Best documented MD-GRTN-line result | 15.164 (test) | +2.050 | Closest this line got; superseded by GWNet |
| 10 | `mdgrtn_v14_14_9_current_best.ipynb` | 2 | MD-GRTN + GAT | Marked checkpoint; full training curve | 16.417 (val, ep.15) | — | Reference checkpoint for the line |
| 11 | `gwnet-v18.ipynb` | 3 | Graph WaveNet | First GWNet notebook | 14.923 (test) | +1.809 | **First metric win (MAPE) vs. baseline** |
| 12 | `gwnet_v21_strong_full.ipynb` | 3 | Graph WaveNet | Config stabilizes: `n_supports=3`, composite loss, EMA, OneCycleLR | not captured | — | This config becomes the standing GWNet reference |
| 13 | `gatstcnet.ipynb` → `_oom_fixed.ipynb` | 4 | GAT+ST+Conv | Cross-period attention → 17GB OOM → fixed | not captured | — | Line continued 2 more iterations, then dropped |
| 14 | `fusion_traffic.ipynb` | 4 | FUSION (GWNet×STAEformer×MD-GRTN) | Explicit 3-way architecture combination | no saved output | — | **Not iterated further; no documented result** |
| 15 | `PI_GWM_PEMS08.ipynb` | 4 | Physics-informed + curriculum loss | Derivative feature + `CurriculumLoss`; **first `SelectiveSSM`** | not captured | — | Both ideas carried into GW-Mamba |
| 16 | `GW_Mamba_PEMS08.ipynb` | 5 | GW-Mamba (v1) | Dedicated Mamba refinement block added | not captured | — | Architecture direction set |
| 17 | `GW_Mamba_v3_PEMS08.ipynb` | 5 | GW-Mamba (bidirectional) | `BiMambaBlock` tried | not captured | — | Not the final design (unidirectional shipped) |
| 18 | `gw-mamba-pems08_baseline.ipynb` | 6 | **GW-Mamba (final)** | Architecture frozen | **13.494** (test) | **+0.380** | Final PEMS08 result — RMSE & MAPE beaten, MAE narrowly missed |
| 19 | `GW_Mamba_PEMS04.ipynb` (gw mamba/) | 6 | GW-Mamba (final) | Re-trained, same architecture | **18.528** | −0.062 (vs. GWNet 18.59) | ✅ Full sweep beaten |
| 20 | `gw-mamba-pem3.ipynb` | 6 | GW-Mamba (final) | Re-trained, same architecture | **14.880** (test) | +1.880 (vs. GWNet 13.0) | ❌ Val→test generalization gap |
| 21 | `gw-mamba-pem07.ipynb` | 6 | GW-Mamba (final) | Re-trained, same architecture; last-dated notebook | **19.803** | −0.777 (vs. GWNet 20.58) | ✅ Full sweep beaten |


