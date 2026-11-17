# RagMTL (MAGRA)

MAGRA is an experimental, CPU-capable retrieval and grounded-answer pipeline, with a reusable benchmark and development-only evaluation controls. It uses **SQLite as the canonical document store** and **`minimax/minimax-m3.1-flash`** through OpenRouter. The preferred local setting is `ai-model-key-increased-paid`, with an authorized ceiling of **$100 per UTC month** and one shared ledger preserving all previous spending. Confirmed API credit, budget or quota exhaustion stops the whole goal until the user restores access. No improvement over the strongest feasible baseline is claimed yet.

## Repository boundaries

RagMTL owns the MAGRA retrieval, evidence aggregation, benchmarking, and continuity packages. [BackToMTL](https://github.com/kusicheng/backtomtl) owns the translation backend and image processing. [OpenMTL](https://github.com/kusicheng/openmtl) owns the frontend and user-facing application.

These are separate repositories. Keep implementation in its owning repository; share functionality through explicit imports, supported APIs, or a deliberately abstracted shared component.
