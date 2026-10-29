# RAG generation audit

- Phase: `dev`.
- Rows completed: 336.
- Unique DEV questions selected: 56.
- Compatible pilot rows reused: 48.
- Retrieval candidates: `5`; generator contexts: `1`.
- Test was not evaluated or used for selection.
- Model loading is excluded from per-question latency.
- Retrieval latency is reused from frozen Phase 1 rankings; the generator receives only ranking position 1.
- Error analysis labels are diagnostic candidates requiring context inspection, not proof of causality.

## Metrics

| Model | Method | EM (%) | F1 (%) | Answerable F1 (%) | Unanswerable F1 (%) | Retrieval mean (s) | Generation mean (s) | Total mean (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B_FT_zero | bm25 | 66.07 | 71.79 | 67.09 | 100.00 | 0.0685 | 0.2597 | 0.3292 |
| 1.5B_FT_zero | hybrid | 69.64 | 74.47 | 72.29 | 87.50 | 0.3389 | 0.2880 | 0.6277 |
| 1.5B_FT_zero | semantic | 71.43 | 76.95 | 73.10 | 100.00 | 0.2703 | 0.2711 | 0.5425 |
| 7B_FT_zero | bm25 | 57.14 | 67.55 | 66.31 | 75.00 | 0.0685 | 0.7606 | 0.8302 |
| 7B_FT_zero | hybrid | 62.50 | 72.11 | 71.63 | 75.00 | 0.3389 | 1.0540 | 1.3938 |
| 7B_FT_zero | semantic | 60.71 | 70.64 | 67.83 | 87.50 | 0.2703 | 0.6691 | 0.9412 |
