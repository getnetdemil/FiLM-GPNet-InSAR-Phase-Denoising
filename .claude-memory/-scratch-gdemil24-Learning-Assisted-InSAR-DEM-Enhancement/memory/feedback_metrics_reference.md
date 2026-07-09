---
name: Authoritative Metrics Table
description: The single source of truth for all FiLMUNet vs Goldstein metric values across all three AOIs. Always use these exact numbers when discussing results.
type: feedback
---

Always refer to this table when discussing metric results. Do NOT use values from memory notes, eval logs, or earlier conversation summaries — these numbers supersede all previous references.

**Why:** User confirmed these as the final authoritative values for the contest submission (2026-04-07).
**How to apply:** Any time a metric value is cited, cross-check against this table first.

## Authoritative Results Table

| Metric | Gold AOI000 | Gold AOI024 | Gold AOI008 | Film AOI000 (Δ) | Film AOI024 (Δ) | Film AOI008 (Δ) |
|--------|------------|------------|------------|-----------------|-----------------|-----------------|
| Closure (rad) ↓ | 1.018 | 0.536 | 0.769 | **0.915** (−10%) | **0.468** (−6%) | 0.771 (+0%) |
| Unwrap ↑ | 0.256 | 0.531 | 0.256 | **0.258** (+0.2p) | **0.608** (+7p) | 0.248 (−0.2p) |
| NMAD (m) ↓ | 40.13 | 18.32 | 40.13 | **39.44** (−2%) | **12.64** (−31%) | **39.40** (−2%) |
| Temp. R. (rad) ↓ | 1.158 | 1.069 | 1.486 | **0.367** (−68%) | **0.361** (−66%) | **1.450** (−2%) |

Bold = FiLMUNet is better than Goldstein for that cell.

## AOI Notes

- **AOI000** (Hawaii): cropped SLC, primary training+eval, 224 pairs
- **AOI024** (W. Australia): full-scale SLC, separate training + fine-tune, ~4 selected pairs
- **AOI008** (Los Angeles): zero-shot evaluation (AOI024 checkpoint, no retraining)

## Key headline numbers (for paper writing)
- M5 Hawaii: −68% (1.158 → 0.367) ← headline result
- M5 AOI024: −66% (1.069 → 0.361) ← confirms generalization
- M5 AOI008: −2% (1.486 → 1.450) ← zero-shot, modest but positive
- M4 AOI024: −31% (18.32 → 12.64) ← strongest DEM improvement
- M1 AOI024: −6% (0.536 → 0.468) ← improved (NOT worse — updated value)
- M1 AOI008: +0% (0.769 → 0.771) ← effectively unchanged (zero-shot, expected)
