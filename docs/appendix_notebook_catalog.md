# Appendix — Full Notebook Catalog

All 154 notebooks, grouped by phase and ordered chronologically within each. Each entry is generated from the same automated extraction pass used throughout this report (class definitions, config values, and printed metric/baseline lines pulled directly from notebook code and saved outputs) — not hand-written summaries, so the level of detail per notebook reflects what that notebook actually contains and saved, not an editorial judgment of its importance. Notebooks discussed in depth elsewhere link back to the relevant chapter.


## Phase 0 — Toy Dataset Prototyping (Feb 14–22)

*18 notebooks*

**`1_traffic flow prediction.ipynb`** (2026-02-14 23:29, `other models/model on small dataset/`) — Code references: CNN, GRU, LSTM, Transformer. Config: batch_size=32, epochs=20. Output includes: Step t+3 → MAE: 19.62 | RMSE: 25.46.

**`2_traffic_flow_pred_junc.ipynb`** (2026-02-15 01:45, `other models/model on small dataset/`) — Code references: CNN, GRU, LSTM, Transformer. Config: batch_size=32, epochs=20. Output includes: J4 → MAE: 2.40, RMSE: 3.20.

**`3_traffic_flow_pred_llm.ipynb`** (2026-02-15 01:51, `other models/model on small dataset/`) — Defines `PositionalEncoding, TimeSeriesLLM, TrafficDataset`. Config: lr=0.001, batch_size=32, optimizer=Adam, seq_len=24. Output includes: RMSE: 18.089785970135537. ⚠️ 1 cell(s) show a code error in saved output.

**`4_traffic_flow_llm_junc.ipynb`** (2026-02-15 01:56, `other models/model on small dataset/`) — Defines `PositionalEncoding, TimeSeriesLLM`. Config: lr=0.0003, batch_size=32, epochs=40, optimizer=Adam, seq_len=24. Output includes: RMSE: 3.737294990760219.

**`5_traffic_flow_junc+global.ipynb`** (2026-02-15 02:03, `other models/model on small dataset/`) — Defines `LLM_CNN, LLM_GRU, LLM_LSTM`. Config: lr=0.0003, batch_size=32, optimizer=Adam, seq_len=24. Output includes: RMSE: 12.816789606808536.

**`6_traffic_flow_gnn.ipynb`** (2026-02-15 02:07, `other models/model on small dataset/`) — Defines `GNNLayer, GNNModel`. Config: lr=0.0003, batch_size=32, optimizer=Adam, seq_len=24. Output includes: RMSE: 3.554754570967517.

**`7_traffic _flow_hybrid_inverse_llm.ipynb`** (2026-02-15 02:11, `other models/model on small dataset/`) — Defines `CNN_LLM, GRU_LLM, LSTM_LLM`. Config: lr=0.0003, batch_size=32, optimizer=Adam, seq_len=24. Output includes: RMSE: 3.606564571207151.

**`8_traffic_flow_pems08_cnn+llm.ipynb`** (2026-02-15 02:16, `other models/model on small dataset/`) — Defines `CNNTransformer, TrafficDataset`. Config: lr=0.00012, batch_size=64, epochs=100, optimizer=Adam, seq_len=24, pred_len=1. Output includes: RMSE: 25.380805730438933.

**`9_traffic_flow_pems08_gru+llm.ipynb`** (2026-02-15 02:19, `other models/model on small dataset/`) — Defines `GRUTransformer, TrafficDataset`. Config: lr=0.00012, batch_size=64, epochs=100, optimizer=Adam, seq_len=36, pred_len=1. Output includes: MAPE: 172765.97670037972.

**`10_traffic_flow_pems08_lstm+llm.ipynb`** (2026-02-15 02:21, `other models/model on small dataset/`) — Defines `LSTMTransformer`. Config: lr=0.00012, epochs=100, optimizer=Adam, seq_len=36, pred_len=1. Output includes: MAPE: 158770.17354440744.

**`13_traffic_flow_pems08_cnn+llm_v2.ipynb`** (2026-02-15 17:08, `other models/model on small dataset/`) — Defines `CNNLLM, TrafficDataset`. Config: lr=0.0001, optimizer=Adam, seq_len=48, pred_len=3. Output includes: MAPE: 10.737555153574212.

**`12__traffic_flow_pems08_cnn+llm_v3.ipynb`** (2026-02-17 17:46, `other models/model on small dataset/`) — Defines `CNNLLM, PositionalEncoding, TrafficDataset`. Config: lr=3e-4, epochs=60, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=72, pred_len=12. Output includes: RMSE: 28.033489363240285.

**`14_traffic_flow_cnn+llm_v4.ipynb`** (2026-02-18 02:55, `other models/model on small dataset/`) — Defines `CNN_LLM_PRO, TrafficDataset`. Config: lr=5e-4, batch_size=32, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 339719.7.

**`15_traffic_flow_lstm+llm_v2.ipynb`** (2026-02-20 23:51, `other models/model on small dataset/`) — Defines `LSTM_LLM_PRO, TrafficDataset`. Config: lr=3e-4, batch_size=32, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 10.671251.

**`16_traffic_flow_lstm+llm_v3.ipynb`** (2026-02-21 00:31, `other models/model on small dataset/`) — Defines `LSTM_LLM_RES, TrafficDataset`. Config: lr=2e-4, batch_size=32, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 11.037854.

**`17_traffic_flow_cnn+lstm+llm.ipynb`** (2026-02-21 02:05, `other models/model on small dataset/`) — Defines `CNN_LSTM_LLM, TrafficDataset`. Config: lr=3e-4, batch_size=32, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 11.049533.

**`18_traffic_flow_gnn+llm.ipynb`** (2026-02-22 03:37, `other models/model on small dataset/`) — Defines `GNN_LLM_ADV, GraphConv, TrafficDataset`. Config: lr=3e-4, batch_size=32, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 12.405424.

**`19_traffic_flow_gnn+llm_v3.ipynb`** (2026-02-22 21:36, `other models/model on small dataset/`) — Defines `AdaptiveGraphConv, Adaptive_GNN_LLM, TrafficDataset`. Config: lr=3e-4, batch_size=32, epochs=60, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=96, pred_len=12. Output includes: MAPE: 11.843196.


## Phase 1 — PEMS08 Direct Baselines (Mar 3–5)

*8 notebooks*

**`21_traffic_flow_pems08_lstm_colab_gpu (1).ipynb`** (2026-03-03 11:51, `other models/`) — Defines `LSTMTransformer, TrafficDataset`. Config: lr=0.00015, epochs=120, optimizer=Adam, scheduler=CosineAnnealingLR, seq_len=48, pred_len=1. Output includes: MAPE: 12.0357%  (flow > 1 only).

**`22_traffic_flow_pems08_fixed_lstm+llm_v2 (1).ipynb`** (2026-03-03 12:18, `other models/`) — Defines `LSTMTransformer, TrafficDataset`. Config: lr=1e-6, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=48, pred_len=1. Output includes: MAPE: 11.0704%  (flow > 1 only).

**`21_traffic_flow_pems08_v3_llm+lstm_fix_overfit (1).ipynb`** (2026-03-03 12:35, `other models/`) — Defines `LSTMTransformer, TrafficDataset`. Config: lr=1e-6, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=48, pred_len=1. Output includes: MAPE: 11.3762%  (flow > 1 only).

**`23_traffic_flow_pems08_v4_TCN (1).ipynb`** (2026-03-03 12:45, `other models/`) — Defines `TCN, TemporalBlock, TrafficDataset`. Config: lr=1e-6, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=48, pred_len=1. Output includes: MAPE: 11.4552%  (flow > 1 only).

**`24_traffic_flow_pems08_v5_persensor (1).ipynb`** (2026-03-04 05:45, `other models/`) — Defines `LSTMModel, TrafficDataset`. Config: lr=1e-6, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=1. Output includes: Per-sensor MAE — min: 2.60  max: 81.28  mean: 18.51.

**`25_traffic_flow_pems08_v6_fixed (1).ipynb`** (2026-03-04 05:58, `other models/`) — Defines `LSTMModel, TrafficDataset`. Config: lr=1e-6, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=36, pred_len=3. Output includes: Step 3 (+15 min)  MAE=18.6281.

**`26_traffic_pems08_CNN_LSTM_v2.ipynb`** (2026-03-05 10:50, `other models/`) — Defines `CNNBlock, CNNLSTMModel, TrafficDataset`. Config: lr=1e-6, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=36, pred_len=1. Output includes: MAPE: 15.1419%.

**`27_traffic_pems08_GCN_Transformer (1).ipynb`** (2026-03-05 11:12, `other models/`) — Defines `GCNTransformer, GraphConv, PositionalEncoding, TrafficDataset`. Config: lr=1e-6, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=1. Output includes: MAPE: 11.3363% (paper MD-GRTN:  8.471%).


## Phase 2 — MD-GRTN Reproduction & Search (Mar 13–Apr 11)

*66 notebooks*

**`mdgrtn_phase1.ipynb`** (2026-03-13 23:35, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Output includes: Epoch 040 | Loss=0.0990 | Val MAE=16.740  RMSE=25.450  MAPE=13.51%.

**`mdgrtn_exact_phase 3.ipynb`** (2026-03-14 21:57, `other models/`) — Defines `GCN_GRU_Layer, MAFModule, MDAFModule, MDGRTN, MDModule` (+9 more). Config: lr=1e-3, batch_size=64, optimizer=AdamW, seq_len=12, pred_len=12.

**`mdgrtn_phase2.ipynb`** (2026-03-14 21:58, `other models/`) — Defines `BackNet, ChebGraphConv, GCN_GRU_Layer, MDAFModule, MDGRTN` (+6 more). Config: lr=5e-4, batch_size=16, epochs=100, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12. Result: Ep 011 | Loss=0.1671 | MAE=17.121 RMSE=26.417 MAPE=6599.87% ✓ best.

**`mdgrtn_final.ipynb`** (2026-03-15 21:39, `other models/`) — Defines `GCN_GRU_Layer, MAFModule, MDAFModule, MDGRTN, MDGRTNDataset` (+9 more). Config: lr=1e-3, batch_size=64, optimizer=AdamW, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v2.ipynb`** (2026-03-19 19:20, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3.ipynb`** (2026-03-19 20:41, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_v4.ipynb`** (2026-03-19 22:01, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=150, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_phase2 .o.ipynb`** (2026-03-20 04:56, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_phase2_fixed.ipynb`** (2026-03-20 05:12, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_paper_exact.ipynb`** (2026-03-23 19:43, `other models/`) — Defines `BackNet, GCN_GRU_Layer, MAFModule, MDAFModule, MDGRTN` (+6 more). Config: lr=0.001, batch_size=64, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_nodiff.ipynb`** (2026-03-24 16:37, `other models/`) — Defines `GCN_GRU_Layer, MAFModule, MDGRTN_NoDiff, MGRCModule, MultiGraphFusion` (+3 more). Config: lr=1e-3, batch_size=64, epochs=100, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_nodiff_v2.ipynb`** (2026-03-25 18:49, `other models/`) — Defines `GCN_GRU_Layer, MAFModule, MDGRTN_NoDiff, MGRCModule, MultiGraphFusion` (+3 more). Config: lr=1e-3, batch_size=64, epochs=100, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_nodiff_production.ipynb`** (2026-03-25 19:24, `other models/`) — Defines `DatasetWrap, GraphConstructor, MAF, MGRC, Model` (+1 more).

**`mdgrtn_nodiff_updated.ipynb`** (2026-03-25 20:12, `other models/`) — Defines `GCN_GRU_Layer, MAFModule, MDGRTN_NoDiff, MGRCModule, MultiGraphFusion` (+3 more). Config: lr=1e-3, batch_size=64, epochs=800, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v2 no diff.ipynb`** (2026-03-26 10:00, `other models/`) — Defines `GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN, MGRCModule` (+4 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v2 gat without diff and trans.ipynb`** (2026-03-26 16:48, `other models/`) — Defines `BackNet, GATConv, GAT_GRU_Layer, MDAFModule, MDGRTN` (+4 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v2 gat.ipynb`** (2026-03-26 16:51, `other models/`) — Defines `BackNet, GATConv, GAT_GRU_Layer, MDAFModule, MDGRTN` (+4 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 3seq.ipynb`** (2026-03-26 16:59, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 seq3 updated.ipynb`** (2026-03-27 15:48, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 2seq.ipynb`** (2026-03-27 20:10, `other models/`) — Defines `BackNet, GCN_GRU_Layer, GraphConv, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 2seq gat.ipynb`** (2026-03-27 22:01, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 2seq no gcn ,, with gcn.ipynb`** (2026-03-27 22:12, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v3 a.ipynb`** (2026-03-28 10:04, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=1e-3, batch_size=32, epochs=100, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v4.ipynb`** (2026-03-28 10:52, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=5e-4, batch_size=64, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v4 new.ipynb`** (2026-03-28 11:01, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v5.ipynb`** (2026-03-28 21:37, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+6 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12.

**`mdgrtn_phase1_v6.ipynb`** (2026-03-28 22:27, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v7.ipynb`** (2026-03-28 22:33, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v8.ipynb`** (2026-03-28 22:43, `other models/`) — Defines `BackNet, GAT_GRU_Layer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v9.ipynb`** (2026-03-29 03:58, `other models/`) — Defines `BackNet, GATLayer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v10.ipynb`** (2026-03-29 10:27, `other models/`) — Defines `BackNet, GATLayer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v11.ipynb`** (2026-03-29 10:39, `other models/`) — Defines `BackNet, GATLayer, GraphAttention, MDGRTN, MGRCModule` (+3 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12.

**`mdgrtn_phase1_v12.ipynb`** (2026-03-29 14:54, `other models/`) — Defines `AdjacencyFusion, BackNet, DualDiffusionModule, GATLayer, GraphAttention` (+5 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12.

**`mdgrtn_phase1_v13.ipynb`** (2026-03-30 23:01, `other models/`) — Defines `AdjacencyFusion, BackNet, DualDiffusionModule, GATLayer, GraphAttention` (+5 more). Config: lr=1e-5, batch_size=32, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12.

**`mdgrtn_phase3.ipynb`** (2026-03-30 23:43, `other models/`) — Defines `AdjacencyFusion, BackNet, DilatedCausalConv, DilatedTCNBlock, DilatedTCNStack` (+6 more). Config: lr=1e-5, batch_size=32, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=32, pred_len=12.

**`mdgrtn-v11 actual.ipynb`** (2026-03-30 23:54, `other models/`) — Defines `BackNet, GATLayer, GraphAttention, MDAFModule, MDGRTN` (+7 more). Config: lr=5e-4, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Checkpoint saved: mdgrtn_phase1_ckpt.pt (epoch 6, best_mae=16.570).

**`mdgrtn_phase1_v14.ipynb`** (2026-03-31 10:25, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn_phase1_v15.ipynb`** (2026-03-31 20:37, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+4 more). Config: lr=5e-4, batch_size=64, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=16, pred_len=12.

**`mdgrtn-v14 14.9.ipynb`** (2026-04-01 16:51, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_phase1_v16.ipynb`** (2026-04-01 16:56, `other models/`) — Defines `GATLayer, GraphAttention, InputProjection, MDGRTN, MGRCModule` (+3 more). Config: lr=5e-6, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_phase1_v17.ipynb`** (2026-04-01 21:11, `other models/`) — Defines `GATLayer, GraphAttention, InputProjection, MDGRTN, MGRCModule` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_phase1_v18.ipynb`** (2026-04-01 21:15, `other models/`) — Defines `GATLayer, GraphAttention, InputProjection, MDGRTN, MGRCModule` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_gwn.ipynb`** (2026-04-01 21:53, `other models/`) — Defines `AdaptiveAdjacency, MDGRTN, SinusoidalPE, TemporalTransformerLayer, TimeEmbedding` (+2 more). Config: lr=1e-3, batch_size=64, epochs=150, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_new_v1.ipynb`** (2026-04-02 11:52, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+5 more). Config: lr=5e-4, batch_size=128, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_new_v1_updated.ipynb`** (2026-04-02 12:37, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+5 more). Config: lr=5e-4, batch_size=128, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_phase1_v20.ipynb`** (2026-04-02 20:01, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+5 more). Config: lr=5e-4, batch_size=128, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn_v14_14_9_current_best.ipynb`** (2026-04-03 15:22, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+4 more). Config: lr=5e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=18, pred_len=12. Result: Ep 034 | Loss=0.0257 | MAE=15.703 RMSE=24.886 MAPE=8.83% lr=4.7e-04  ← best ✓.

**`mdgrtn_phase1_v21.ipynb`** (2026-04-03 15:30, `other models/`) — Defines `GATLayer, GraphAttention, InputProjection, MDGRTN, MGRCModule` (+4 more). Config: lr=1e-5, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Ep 034 | Loss=0.0257 | MAE=15.703 RMSE=24.886 MAPE=8.83% lr=4.7e-04  ← best ✓.

**`mdgrtn_v22.ipynb`** (2026-04-03 18:45, `other models/`) — Defines `AdaptiveGraph, GATConv, MDGRTNv15, STFormerBlock, SinusoidalPE` (+4 more). Config: lr=2e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`mdgrtn_v23.ipynb`** (2026-04-03 19:30, `other models/`) — Defines `AdaptiveGraph, GATConv, MDGRTNv15, STFormerBlock, SinusoidalPE` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn_v24.ipynb`** (2026-04-03 20:34, `other models/`) — Defines `AdaptiveGraph, GATConv, MDGRTNv15, STFormerBlock, SinusoidalPE` (+4 more). Config: lr=1e-6, batch_size=64, epochs=200, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgrtn v24.ipynb`** (2026-04-03 22:03, `other models/`) — Defines `AdaptiveGraph, GATConv, MDGRTNv15, STFormerBlock, SinusoidalPE` (+4 more). Config: lr=1e-6, batch_size=64, epochs=200, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Ep 007 | Loss=0.0261 | MAE=20.928 RMSE=31.470 MAPE=12.25% lr=3.0e-04  <- best.

**`mdgrtn_v25.ipynb`** (2026-04-03 23:14, `other models/`) — Defines `AdjacencyFusion, GraphAttention, InputProjection, MDGRTNv15, MGRCModule` (+6 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=16, pred_len=12.

**`mdgrtn_v26.ipynb`** (2026-04-04 10:55, `other models/`) — Defines `GCNLayer, InputDenoiser, MDGRTNv26, MGRCModule, STFormerBlock` (+5 more). Config: lr=5e-4, batch_size=32, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`mdgrtn_v27.ipynb`** (2026-04-04 14:36, `other models/`) — Defines `ConvDenoiser, InputProjection, MDAFModule, MDGRTNv26, MGRCBlock` (+6 more). Config: lr=2e-3, batch_size=48, epochs=300, optimizer=AdamW, scheduler=CosineAnnealingWarmRestarts, seq_len=12, pred_len=12.

**`mdgrtn_v28.ipynb`** (2026-04-04 15:35, `other models/`) — Defines `GCNGRUBlock, GCNLayer, InputProjection, MDGRTNv26, MGRCModule` (+5 more). Config: lr=1e-3, batch_size=32, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`pems08_beat_mdgrtn.ipynb`** (2026-04-04 16:44, `other models/`) — Defines `DualGraphBias, PEMSDataset, STEmbedding, SpatialLayer, TemporalLayer` (+1 more). Config: lr=2e-3, batch_size=64, epochs=150, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mdgrtn-v30.ipynb`** (2026-04-04 18:02, `other models/`) — Defines `AdjacencyFusion, GATLayer, GraphAttention, InputProjection, MDGRTN` (+4 more). Config: lr=3e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn-v31.ipynb`** (2026-04-04 18:18, `other models/`) — Defines `AdaptiveDecoder, DualGraphGAT, GraphAttention, InputProjection, MDGRTN` (+3 more). Config: lr=1e-5, batch_size=32, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgrtn-v32.ipynb`** (2026-04-04 18:58, `other models/`) — Defines `AdaptiveDecoder, DualGraphGAT, GraphAttention, InputProjection, MDGRTN` (+3 more). Config: lr=1e-5, batch_size=32, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`mdgwnet-v21.ipynb`** (2026-04-07 16:12, `other models/`) — Defines `DiffusionGCN, MDGWNet, MultiPeriodFusion, STFormer, SpatialTransformer` (+3 more). Config: lr=1e-5, batch_size=48, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`mdgwnet-v18.ipynb`** (2026-04-07 16:24, `other models/`) — Defines `DiffusionGCN, MDGWNet, MultiPeriodFusion, STFormer, SpatialTransformer` (+3 more). Config: lr=1e-3, batch_size=48, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`mdgwnet-v117.ipynb`** (2026-04-07 18:03, `other models/`) — Defines `DiffusionGCN, GatedPeriodFusion, MPGWNet, TrafficDataset3Stream, WaveBlock`. Config: lr=2e-3, batch_size=48, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`superior_mdgrtn_beater.ipynb`** (2026-04-11 19:28, `other models/`) — Defines `DiffusionGCN, GWNetV26, STFormer, SpatialTransformerLayer, TemporalTransformerLayer` (+2 more). Config: lr=1e-3, batch_size=48, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`superior_mdgrtn_beater_vram_fixed.ipynb`** (2026-04-11 19:30, `other models/`) — Defines `CheckpointedWaveBlock, DiffusionGCN, GWNetV26LowVRAM, TrafficDataset3S`. Config: lr=1e-3, batch_size=32, epochs=250, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`mdgrtn_beater_user_paths.ipynb`** (2026-04-11 19:32, `other models/`) — Code references: MDGRTN. Config: lr=1e-3, batch_size=32, epochs=200, scheduler=OneCycleLR, seq_len=24, pred_len=12.


## Phase 3 — Graph WaveNet Family (Apr 4–16)

*21 notebooks*

**`gwnet-v18.ipynb`** (2026-04-04 21:50, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=5e-5, batch_size=64, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`gwnet-v20.ipynb`** (2026-04-04 22:35, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=5e-5, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`gwnet-v21.ipynb`** (2026-04-05 08:31, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`gwnet-v22.ipynb`** (2026-04-05 11:36, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 014 | Loss=0.0341 | MAE=16.811 RMSE=26.004 MAPE=9.34% lr=3.0e-04  ← best ✓.

**`gwnet-v20-3-o.ipynb`** (2026-04-05 11:45, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=5e-5, batch_size=128, epochs=55, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12. Result: Ep 019 | Loss=0.1577 | MAE=15.043 RMSE=24.576 MAPE=8.65% lr=8.3e-04  ← best ✓.

**`gwnet-v23.ipynb`** (2026-04-06 05:10, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=128, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 019 | Loss=0.1577 | MAE=15.043 RMSE=24.576 MAPE=8.65% lr=8.3e-04  ← best ✓.

**`gwnet_v24.ipynb`** (2026-04-06 17:06, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=2e-3, batch_size=64, epochs=150, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`gwnet-v20-4.o.ipynb`** (2026-04-06 18:35, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=5e-5, batch_size=64, epochs=55, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12. Result: Ep 014 | Loss=0.1594 | MAE=15.251 RMSE=24.597 MAPE=8.70% lr=9.3e-04  ← best ✓.

**`gwnet-v20-5-o.ipynb`** (2026-04-06 21:49, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=5e-5, batch_size=64, epochs=55, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12. Result: Ep 015 | Loss=0.1585 | MAE=15.248 RMSE=24.369 MAPE=9.08% lr=9.1e-04  ← best ✓.

**`gwnet_v26.ipynb`** (2026-04-06 22:16, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=2e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=18, pred_len=12.

**`gwnet-v29.ipynb`** (2026-04-08 08:45, `other models/`) — Defines `DiffusionGCN, GWNetV22, TrafficDataset, WaveBlock`. Config: lr=1e-5, batch_size=64, epochs=150, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12.

**`gwnet_rtx1.ipynb`** (2026-04-08 14:59, `other models/`) — Defines `DiffusionGCN, GWNetV23, PeriodFusion, TrafficDataset3S, WaveBlock`. Config: lr=2e-3, batch_size=48, epochs=250, optimizer=AdamW, scheduler=OneCycleLR, seq_len=16, pred_len=12.

**`gwnet-rtx2.ipynb`** (2026-04-08 16:39, `other models/`) — Defines `DiffusionGCN, GWNetV24, PeriodFusion, TrafficDataset3S, WaveBlock`. Config: lr=5e-6, batch_size=48, epochs=250, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=16, pred_len=12.

**`gwnet-rtx3.ipynb`** (2026-04-09 09:50, `other models/`) — Defines `DiffusionGCN, GWNetV25, STFormer, SpatialTransformerLayer, TemporalTransformerLayer` (+2 more). Config: lr=1e-5, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12.

**`gwnet_v21_strong_full.ipynb`** (2026-04-11 20:06, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=48, epochs=70, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`gwnet_ab1.ipynb`** (2026-04-11 22:39, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=48, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12.

**`gwnet_ab2.ipynb`** (2026-04-12 03:40, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`gwnet_ab3.ipynb`** (2026-04-12 03:51, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=24, pred_len=12.

**`gwnet-vk1.ipynb`** (2026-04-16 16:14, `other models/`) — Defines `DiffusionGCN, GWNet, TrafficDataset, WaveBlock`. Config: lr=1e-5, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`gwnet-vk 2.ipynb`** (2026-04-16 20:14, `other models/`) — Defines `AdaptiveTemporalPool, DiffusionGCN, DropPath, GWNet, TrafficDataset` (+1 more). Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`gwnet-vk3.ipynb`** (2026-04-16 22:38, `other models/`) — Defines `AdaptiveTemporalPool, DiffusionGCN, DropPath, GWNet, TrafficDataset` (+1 more). Config: lr=1e-5, batch_size=64, epochs=200, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.


## Phase 4 — Wide Architecture Search (Apr 6–21)

*30 notebooks*

**`stgtnet.ipynb`** (2026-04-06 05:45, `other models/`) — Defines `DiffusionGCN, STFormerBlock, STGTNet, TrafficDataset`. Config: lr=5e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 019 | Loss=0.1577 | MAE=15.043 RMSE=24.576 MAPE=8.65% lr=8.3e-04  ← best ✓.

**`stwavenet.ipynb`** (2026-04-06 11:29, `other models/`) — Defines `MultiScaleWaveBlock, NodeAdaptiveGCN, STWaveNet, TrafficDataset`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=16, pred_len=12. Result: Ep 019 | Loss=0.1577 | MAE=15.043 RMSE=24.576 MAPE=8.65% lr=8.3e-04  ← best ✓.

**`stid-pems08.ipynb`** (2026-04-08 10:13, `other models/`) — Defines `PEMSDataset, STID, STIDBlock, SpatialMLP, TemporalMLP`. Config: lr=2e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`tignet-pems.ipynb`** (2026-04-08 10:25, `other models/`) — Defines `AdaptiveGCN, GatedPeriodFusion, PEMSDataset, SpatialMLP, TIGNet` (+3 more). Config: lr=2e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`stagnet.ipynb`** (2026-04-08 19:21, `other models/`) — Defines `FeedForward, InputProjection, STAGNet, STBlock, SpatialAttention` (+2 more). Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`MSDTN_GWNNet.ipynb`** (2026-04-11 11:02, `other models/`) — Defines `AdaptiveAdjacency, DilatedTemporalConv, GatedGraphAttention, MSDTN, MSDTNLayer` (+4 more). Config: lr=1e-3, batch_size=64, epochs=200, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12.

**`MSDTN.ipynb`** (2026-04-11 11:08, `other models/`) — Defines `AdaptiveAdjacency, DilatedTemporalConv, GatedGraphAttention, MSDTN, MSDTNLayer` (+4 more). Config: lr=1e-3, batch_size=64, epochs=200, optimizer=Adam, scheduler=ReduceLROnPlateau, seq_len=12.

**`mpstf.ipynb`** (2026-04-11 16:13, `other models/`) — Defines `FFNBlock, MPSTF, STBlock, SpatialAttentionBlock, TemporalAttentionBlock` (+1 more). Config: lr=5e-4, batch_size=48, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`mpstf_fixed.ipynb`** (2026-04-11 16:24, `other models/`) — Defines `FFNBlock, MPSTF, STBlock, SpatialAttentionBlock, TemporalAttentionBlock` (+1 more). Config: lr=5e-4, batch_size=48, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`pasta_v1.ipynb`** (2026-04-12 11:33, `other models/`) — Defines `AdaptiveDiffGCN, PASTANet, PeriodCrossAttn, SpatialAttnBlock, TemporalAttnBlock` (+2 more). Config: lr=5e-4, batch_size=32, epochs=300, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`gwgru_v1.ipynb`** (2026-04-12 17:19, `other models/`) — Defines `DiffusionGCN, GWGRUNet, TrafficDataset3S, WaveBlock`. Config: lr=1e-5, batch_size=32, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`wavest_v1.ipynb`** (2026-04-12 21:18, `other models/`) — Defines `DiffusionGCN, STFormer, SpatialTransformerLayer, TemporalTransformerLayer, TrafficDataset3S` (+2 more). Config: lr=1e-5, batch_size=32, epochs=400, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12.

**`wavegru_v2.ipynb`** (2026-04-13 15:25, `other models/`) — Defines `DiffusionGCN, TrafficDataset3S, WaveBlock, WaveGRUNet`. Config: lr=1e-5, batch_size=32, epochs=400, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`pems08net.ipynb`** (2026-04-13 18:52, `other models/`) — Defines `DiffusionGCN, PEMS08Net, PreNormSTFormer, PreNormSpatialAttn, PreNormTemporalAttn` (+2 more). Config: lr=1e-5, batch_size=32, epochs=400, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`final_model.ipynb`** (2026-04-14 08:10, `other models/`) — Defines `DiffusionGCN, FinalModel, TrafficDataset3S, WaveBlock`. Config: lr=1e-6, batch_size=32, epochs=500, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`pems08netv2.ipynb`** (2026-04-14 15:19, `other models/`) — Defines `DiffusionGCN, PEMS08NetV2, TrafficDataset3S, WaveBlock`. Config: lr=1e-5, batch_size=32, epochs=800, optimizer=AdamW, scheduler=CosineAnnealingWarmRestarts, seq_len=12, pred_len=12.

**`staeformerplus.ipynb`** (2026-04-14 21:14, `other models/`) — Defines `STAEformerPlus, STPreNormLayer, TrafficDataset3S`. Config: lr=1e-3, batch_size=64, epochs=200, optimizer=Adam, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`gwstnet.ipynb`** (2026-04-15 06:29, `other models/`) — Defines `DiffusionGCN, GWSTNet, PreNormSpatialAttn, PreNormTemporalAttn, STBlock` (+2 more). Config: lr=1e-3, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`gatstcnet.ipynb`** (2026-04-18 05:59, `other models/`) — Defines `CrossPeriodAttn, GATSTCNet, GATTCNBlock, GraphAttn, LightDenoise` (+2 more). Config: lr=5e-5, batch_size=64, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`gatstcnet_oom_fixed.ipynb`** (2026-04-18 06:09, `other models/`) — Defines `CrossPeriodAttn, GATSTCNet, GATTCNBlock, GraphAttn, LightDenoise` (+2 more). Config: lr=5e-5, batch_size=32, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`gatstcnet_final.ipynb`** (2026-04-18 10:28, `other models/`) — Defines `CrossPeriodAttn, GATSTCNet, GATTCNBlock, GraphAttn, LightDenoise` (+2 more). Config: lr=5e-5, batch_size=32, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`gatstcnet_v2.ipynb`** (2026-04-19 20:57, `other models/`) — Defines `CrossPeriodAttn, GATSTCNet, GATTCNBlock, GraphAttn, LightDenoise` (+2 more). Config: lr=5e-5, batch_size=32, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`fusion_traffic.ipynb`** (2026-04-19 21:00, `other models/`) — Defines `CrossPeriodAttention, DiffusionGCN, DropPath, FUSIONModel, FusedSTAttention` (+4 more). Config: lr=7e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=24, pred_len=12.

**`fusion_traffic_fixed.ipynb`** (2026-04-19 21:16, `other models/`) — Defines `CrossPeriodAttention, DiffusionGCN, DropPath, FUSIONModel, FusedSTAttention` (+4 more). Config: lr=7e-4, batch_size=64, epochs=200, optimizer=AdamW, scheduler=OneCycleLR, seq_len=12, pred_len=12.

**`mpstaeformer.ipynb`** (2026-04-20 16:47, `other models/`) — Defines `Dataset3S, FFN, MPSTAEformer, STAETable, STBlock` (+2 more). Config: lr=5e-4, batch_size=64, epochs=300, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`mgstformer.ipynb`** (2026-04-20 18:01, `other models/`) — Defines `DatasetMG, FFN, HorizonDecoder, MGSTformer, STAETable` (+3 more). Config: lr=1e-4, batch_size=64, epochs=10, optimizer=AdamW, scheduler=CosineAnnealingLR, seq_len=12, pred_len=12.

**`msta_gwnet_v27.ipynb`** (2026-04-20 21:33, `other models/`) — Defines `DiffusionGCN, MSTAGWNet, PEMSDataset, STAE, WaveBlock`. Config: lr=1e-5, batch_size=64, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`msta_gwnet_v27_debugged.ipynb`** (2026-04-20 21:42, `other models/`) — Defines `DiffusionGCN, MSTAGWNet, PEMSDataset, STAE, WaveBlock`. Config: lr=1e-5, batch_size=64, epochs=300, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Baseline  : MAE=13.114 | RMSE=22.623 | MAPE=8.471%.

**`gwmpc_v28.ipynb`** (2026-04-21 11:18, `other models/`) — Defines `DiffusionGCN, GWMPCNet, PEMSDataset, WaveBlock`. Config: lr=2e-3, batch_size=64, epochs=300, optimizer=AdamW, scheduler=OneCycleLR, seq_len=18, pred_len=12.

**`PI_GWM_PEMS08.ipynb`** (2026-04-21 20:13, `other models/`) — Defines `CurriculumLoss, DiffusionGCN, PEMS08Dataset, PIGWM, PIGWMLayer` (+2 more). Config: lr=3e-3, batch_size=128, epochs=5, optimizer=AdamW, scheduler=CosineAnnealingLR.


## Phase 5 — GW-Mamba Emergence (Apr 21–May 16)

*7 notebooks*

**`GW_Mamba_PEMS08.ipynb`** (2026-04-21 20:59, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMS08Dataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12.

**`GW_Mamba_v3_PEMS08.ipynb`** (2026-04-24 22:44, `other models/`) — Defines `BiMambaBlock, DiffusionGCN, GWMambaV3, GlobalSpatialTransformer, MultiPeriodFusion` (+3 more). Config: lr=5e-5, batch_size=24, epochs=120, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=24, pred_len=12.

**`gw-mamba-pems08_baseline.ipynb`** (2026-05-13 11:45, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMS08Dataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=48, epochs=50, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12. Result: Loaded best checkpoint  (val MAE=14.067).

**`GW_Mamba_PEMS03.ipynb`** (2026-05-14 00:37, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=32, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`GW_Mamba_PEMS04.ipynb`** (2026-05-14 00:37, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=48, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`GW_Mamba_PEMS07.ipynb`** (2026-05-14 00:37, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=16, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12.

**`GW_Mamba_ALL4_PEMS.ipynb`** (2026-05-16 06:15, `other models/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, optimizer=AdamW, scheduler=ReduceLROnPlateau.


## Phase 6 — Final Validation (May 13–25)

*4 notebooks*

**`gw-mamba-pems08_baseline.ipynb`** (2026-05-13 11:45, `gw mamba/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMS08Dataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=48, epochs=50, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=18, pred_len=12. Result: Loaded best checkpoint  (val MAE=14.067).

**`GW_Mamba_PEMS04.ipynb`** (2026-05-23 09:09, `gw mamba/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=48, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: MAPE :   10.821%  (baseline 12.820%  Δ=-1.999)  ✅ BEAT. (A cell shows a manual KeyboardInterrupt — a stopped/re-run cell, not a code error.)

**`gw-mamba-pem3.ipynb`** (2026-05-23 16:22, `gw mamba/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=32, epochs=50, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: Best Val MAE : 13.785  (baseline 13).

**`gw-mamba-pem07.ipynb`** (2026-05-25 05:37, `gw mamba/`) — Defines `DiffusionGCN, GWMamba, MambaRefinementBlock, PEMSDataset, SelectiveSSM` (+1 more). Config: lr=5e-5, batch_size=16, epochs=100, optimizer=AdamW, scheduler=ReduceLROnPlateau, seq_len=12, pred_len=12. Result: MAPE :    7.835%  (baseline 8.210%  Δ=-0.375)  ✅ BEAT. (A cell shows a manual KeyboardInterrupt — a stopped/re-run cell, not a code error.)


---

**Total: 154 notebooks catalogued.**
