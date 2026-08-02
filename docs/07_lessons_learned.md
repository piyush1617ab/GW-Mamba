# 07 — Lessons Learned

Six takeaways, each tied back to a specific, verifiable moment in the notebook history rather than stated as a general principle first.

---

**1. A simpler backbone plus one targeted addition beat a "combine everything" attempt.**
FUSION explicitly merged three architecture families (GWNet × STAEformer × MD-GRTN) and has no documented completed evaluation in the notebooks — the most ambitious model in the corpus is also the one with the least evidence it worked. Two days later, the project's actual next step was `PI_GWM_PEMS08`: one new input feature, one loss schedule change. The architecture that shipped, `GWMamba`, follows the same pattern — Graph WaveNet's backbone kept intact, one refinement layer added at the end, nothing else changed structurally. Scope discipline, not architectural ambition, is what the successful path actually looked like.

**2. Chasing a strong baseline by directly iterating on its architecture has diminishing returns — switching backbones can beat continued tuning.**
Six weeks and 66 notebooks went into reproducing and improving MD-GRTN; the closest documented result still missed the target by 2.05 MAE. The eventual close-the-gap moment came from a structurally different, simpler architecture (Graph WaveNet) reaching a *better* number within days of first being tried. The 66 notebooks weren't wasted — the baseline numbers, the data harness, and the GCN-vs-GAT comparison all carried forward — but the lesson holds regardless: if direct iteration on a specific architecture is flattening out, the next experiment worth running might be a different backbone entirely, not another version of the same one.

**3. Build the ablation *and* save its output.**
The TING ablation (`tignet-pems.ipynb`) is the most carefully planned experiment in the entire corpus — six variants scoped out, with a comment explicitly noting it's *"required for a publishable paper"* — and it has no saved metrics. Good experimental design without a preserved result is functionally the same as not running the experiment, for anyone (including a future version of yourself) trying to reconstruct what was learned. FUSION has the same gap. Where results *were* saved (the OOM fix, the MD-GRTN test-set numbers, the four final PEMS runs), this report could say something precise; where they weren't, it could only say what was *planned*.

**4. On constrained hardware, memory engineering is a real, recurring skill — not a one-off fix.**
Two separate, structurally different OOM/VRAM problems show up in this project: an attention matrix that scales quadratically with `sequence_length × num_nodes` (fixed with reduced batch size + selective mixed precision), and a deep WaveBlock stack whose stored activations exceed available memory (fixed with gradient checkpointing, trading ~40% memory for extra compute). Both fixes are standard techniques, applied correctly and specifically to the actual bottleneck in each case — worth having concrete, rather than generic, answers ready for "tell me about a memory-constrained training problem" in an interview.

**5. Metrics disagree with each other, and that disagreement is informative, not just noise.**
The final PEMS08 result — MAE narrowly missed, RMSE and MAPE both beaten — isn't a wash to round off to "roughly matched the baseline." RMSE penalizes large errors more than MAE does; beating it while missing MAE points at a specific, describable difference in error distribution (fewer big misses, slightly higher typical error) rather than a uniformly-worse-or-better model. Knowing which metric actually matters for a given downstream use (a traffic system that cares about avoiding severe misprediction cares more about RMSE than a system that just wants low average error) turns "did it work?" from a yes/no into a more useful answer.

**6. Validation performance is not test performance, and the PEMS03 gap is worth being upfront about.**
Val MAE 13.785 vs. test MAE 14.880 on PEMS03 is the one result in this project that doesn't match the pattern of the other three datasets. It's tempting to leave a result like this out of a portfolio writeup; it's more useful to keep it in, precisely because "I have one result that shows a generalization gap, and here's what I think it means and don't yet know for certain" is a stronger interview answer than four datasets that all conveniently beat their baselines with no exceptions.

---

**A process note, not a research one:** `generate_full_ipynb.py` — a script that programmatically constructs full notebooks (title, seed-setting cell with deterministic `cudnn` flags, config, model, training loop) rather than hand-editing cells one at a time — is a small piece of infrastructure that doesn't show up in any metric table but is exactly the kind of tooling that makes running 154 structured experiments over three and a half months tractable in the first place.
