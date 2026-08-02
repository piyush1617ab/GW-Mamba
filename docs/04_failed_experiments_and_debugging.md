# 04 — Failed Experiments & Debugging

The original brief for this report insisted failed experiments matter as much as the successful ones. Four are documented here with real evidence: an OOM crash with the exact tensor shape that caused it, a VRAM fix, a structural ablation, and an architecture that was built but — as far as the saved notebooks show — never got a documented result.

---

## The GATSTCNet OOM crash

**What was being tested:** `gatstcnet.ipynb` (Apr 18) combines GAT-style graph attention with cross-period attention (attending across recent/hourly/daily windows simultaneously) — the hypothesis that full attention across both the graph and the time-period dimensions would out-perform Graph WaveNet's fixed/adaptive diffusion supports.

**What broke**, in Piyush's own code comment from the fix notebook:
```
# Old: q=(B, S*N, d), k=(B, 2*S*N, d) → attn=(B,H,4080,8160) = 17GB OOM!
```
Flattening sequence length and node count together into one attention axis (`S*N`) turns the attention matrix quadratic in `S×N` rather than in `S` or `N` alone. With PEMS08's 170 nodes and the multi-period sequence lengths in play, that flattened dimension hits 4080/8160 — an attention matrix large enough to need roughly 17GB on its own, before the rest of the model. This is the clearest, most precisely-diagnosed failure in the entire corpus — the fix notebook's title is literally `gatstcnet_oom_fixed.ipynb`.

**The fix** (verified from code, three changes together):
1. Batch size halved (comment: *"halved: GAT (B\*S,N,N) is the VRAM bottleneck"*).
2. Mixed precision used selectively — `torch.amp.autocast('cuda', enabled=False)` specifically *around* the attention computation (disabling autocast there, likely because fp16 was numerically unstable at that matrix size), while `GradScaler` and autocast stay enabled elsewhere in the model.
3. Explicit `torch.cuda.empty_cache()` calls between steps.

**Outcome:** `gatstcnet_final.ipynb` and `gatstcnet_v2.ipynb` follow (Apr 18–19), but this line doesn't extend past two more iterations — the fundamental cost of flattening node×sequence into one attention axis doesn't go away just because the crash does, and the project's direction (simplify to a stronger backbone, add one targeted refinement) runs counter to adding more full-attention blocks. GAT-style attention itself, more narrowly, was already dropped from the MD-GRTN line for related cost reasons (see the graph-convolution comparison in [`02_architecture_families.md`](02_architecture_families.md)).

---

## The VRAM problem, again — gradient checkpointing

**Where:** `superior_mdgrtn_beater_vram_fixed.ipynb` (Apr 11), a memory-constrained rework of the same day's `superior_mdgrtn_beater.ipynb`.

Different fix this time, for a different bottleneck (not attention — this is a deep WaveBlock stack running out of memory from stored activations): **gradient checkpointing**. Each `WaveBlock` is wrapped so its activations aren't kept in memory during the forward pass; they're recomputed during the backward pass instead, trading compute time for memory:
```python
def make_checkpointed_waveblock(...):
    """Gradient checkpointed WaveBlock - saves ~40% memory"""
    ...
    return checkpoint(core, x, sup, use_reentrant=False)
```
Combined with a batch-size reduction (48 → 32) and a model variant explicitly named `GWNetV26LowVRAM` (~50% of the original parameter count), and per-epoch VRAM logging (`VRAM:{vram:.1f}GB`) added directly to the training loop print statements — this notebook is instrumented specifically to watch the fix work, not just apply it and hope. Both this and the GATSTCNet fix point at the same underlying constraint recurring throughout Phase 2–4: training on Colab/Kaggle-tier GPUs (T4-class, per the `generate_full_ipynb.py` config comment *"Strong-safe defaults (T4 friendly)"*) put a hard ceiling on how large the attention- or depth-heavy variants could get before needing engineering workarounds rather than architecture changes.

---

## The `nodiff` ablation: does MD-GRTN's denoiser earn its keep?

**Where:** `mdgrtn_nodiff.ipynb` and siblings (Mar 24–26), branching off the main MD-GRTN line.

**The hypothesis being tested:** MD-GRTN's `BackNet` (an MLP "denoiser," described in-code as the diffusion model's *backward process*) sits in front of the attention-fusion module, adding real complexity — a full extra submodule per input sequence. The `nodiff` line asks whether that denoising step is actually necessary, or whether the downstream attention fusion and graph-temporal backbone can absorb noisy input directly.

**What changed, precisely** (class-level diff between `mdgrtn_phase1.ipynb` and `mdgrtn_nodiff.ipynb`):
- Removed: `BackNet`, `MDModule` (the denoiser and its wrapper).
- `MDAFModule` → `MAFModule` (attention fusion without the diffusion-denoising stage feeding it).
- Added: `MultiGraphFusion` (a new module, not a straight subtraction — some capacity was redirected rather than simply deleted).

**Outcome:** I found no side-by-side metric comparison (`nodiff` vs. the full version, same epoch count, same everything else) captured in the saved outputs of these specific notebooks. What is verifiable is that the `nodiff` naming convention (`mdgrtn_nodiff`, `mdgrtn_nodiff_v2`, `mdgrtn_nodiff_production`, `mdgrtn_nodiff_updated`) continues for several more iterations through late March — suggesting the simplified line was actively developed rather than dismissed outright — but the mainline MD-GRTN notebooks (with `BackNet`/`MDModule` intact) continue in parallel and are what eventually feeds into `mdgrtn_gwn.ipynb` and the cross-pollination with Graph WaveNet. Whether the denoiser specifically helped or hurt isn't answerable from the saved evidence; what's answerable is that both versions were seriously pursued rather than one being an quick, abandoned side branch.

---

## FUSION: built, not documented as evaluated

**Where:** `fusion_traffic.ipynb` → `fusion_traffic_fixed.ipynb` (Apr 19, sixteen minutes apart).

`FUSIONModel` (see [`02_architecture_families.md`](02_architecture_families.md) for the architecture) is a real, complete implementation — three input windows, components drawn from GWNet, STAEformer, and MD-GRTN. But **neither notebook has any saved execution output**: no error cells, but also no printed metrics and no final output captured in either file. This is different from the OOM crash above, where the failure mode is explicit and diagnosed in-code — here, the notebooks simply don't contain evidence of a completed, evaluated run.

Two honest readings are possible, and the notebooks don't resolve which: either FUSION was run with outputs cleared before saving (losing the record), or it was abandoned before a full evaluation completed — plausible given it's the most parameter-heavy model in the corpus, combining three architectures' worth of components on the same T4-class hardware that was already causing VRAM problems for single-architecture models. What's clear from the timeline is that FUSION is not iterated on again after these two notebooks (16 minutes apart, Apr 19), and the project's next major step (Apr 21, two days later) is `PI_GWM_PEMS08` — a much smaller, more targeted addition (one physics feature, one loss schedule) rather than another combination attempt. The direction change is visible even without a documented FUSION failure: after this point, every new idea in the corpus is additive-and-small, not combinative-and-large.

---

## The six-week MD-GRTN plateau

Not a single crash or a single ablation, but a pattern worth naming directly: **Phase 2 (66 notebooks, Mar 13 – Apr 11) never produces a notebook, among those I opened in depth, that beats the 13.114 MAE target it was built to beat.** The closest documented result is `mdgrtn-v11 actual.ipynb`'s test MAE of 15.164 (Mar 30) — real progress from where the line started, but still 2.05 above target after more than two weeks of phase-numbered iteration, GAT swaps, sequence-length sweeps, and adjacency-fusion variants.

The project's actual path past this plateau wasn't "iterate MD-GRTN harder" — it was lateral: `gwnet-v18.ipynb` (Apr 4, while MD-GRTN iteration was still ongoing) reaches a *better* MAE (14.923) with a structurally simpler model, and is the first notebook in the corpus to beat the baseline on any metric at all. The lesson embedded in the file dates themselves: the six weeks on MD-GRTN weren't wasted (the baseline numbers, the data harness, and the GAT-vs-GCN comparison all carry forward), but the eventual win came from switching backbones, not from further tuning the original one.
