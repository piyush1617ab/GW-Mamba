# GW-Mamba Research Report

Complete documentation of the research journey behind GW-Mamba: 154 experimental iterations across nine architecture families, reconstructed from notebook code, outputs, and filenames.

This is a companion to [`paper/GW-Mamba.pdf`](../paper/GW-Mamba.pdf) (manuscript under review) — the paper reports the final method and results; this report documents how the project arrived there, including the experiments and ideas that didn't make it into the paper.

## Chapters

1. [Research Timeline](01_research_timeline.md) — phase-by-phase chronology with dates and rationale for each transition
2. [Architecture Families](02_architecture_families.md) — every architecture family explored: why, what worked, why kept or dropped
3. [Key Innovations](03_key_innovations.md) — per-sensor normalization, physics-informed feature, three-support diffusion convolution, curriculum loss, Mamba refinement
4. [Failed Experiments & Debugging](04_failed_experiments_and_debugging.md) — OOM crashes, ablations, and ideas that didn't pan out
5. [Final Architecture](05_final_architecture.md) — full technical walkthrough of GW-Mamba, code-grounded
6. [Results Across Datasets](06_results_across_datasets.md) — final numbers on all four PEMS datasets
7. [Lessons Learned](07_lessons_learned.md) — takeaways that generalize beyond this project
8. [Notebook Catalog (Appendix)](appendix_notebook_catalog.md) — all 154 notebooks, one entry each

## Note on results

The baseline comparison numbers in this report were extracted directly from values printed inside the working notebooks during experimentation. The published-track numbers in [`paper/GW-Mamba.pdf`](../paper/GW-Mamba.pdf) and [`../experiments/benchmark_results.md`](../experiments/benchmark_results.md) are the ones properly cited from the literature and should be treated as authoritative for citation purposes; a small number of notebook-era baseline figures in this report (chiefly the PEMS04/PEMS07 Graph WaveNet reference values) differ from the paper's cited figures for that reason.
