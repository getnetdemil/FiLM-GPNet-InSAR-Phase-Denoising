# Ablation Experiments Plan — Table II (Loss-Component Ablation)

**Target paper**: `JSTARS_Journal_Paper/JSTARS_Journal_Paper.tex` (FiLM-GPNet section, joint JSTARS submission)
**Scope**: Fill the six TBD rows in Table II via controlled leave-one-out training runs on **AOI_000 (Hawaii)**.
**Authors on this workstream**: Getnet Demil (Oulu), Muhammad Farhan Humayun (Turku).
**Status**: implementation in flight — sweep #2 running variant 2 of 5 remaining (see §15 Execution Log).

---

## 1. Objective

Replace the six `TBD` entries in Table II (`JSTARS_Journal_Paper.tex` L458–470) with real numbers obtained from matched-recipe training runs on Hawaii, so the paper can claim individual loss-component contributions rather than only the two-row comparison carried over from the IGARSS paper.

The final Table II should have this shape:

| # | Row | Closure ↓ | Unwrap ↑ | Height NMAD ↓ | Temporal ↓ |
|---|-----|-----------|----------|---------------|------------|
| 1 | Published P-S (IGARSS) | 1.021 | — | — | 0.653 |
| 2 | Published P-S + closure + temporal (IGARSS) | 0.915 | — | — | 0.367 |
| 3 | Full objective (reproduced, L_tmp now active) | *run A* | *run A* | *run A* | *run A* |
| 4 | Full − L_unc | *run B* | *run B* | *run B* | *run B* |
| 5 | Full − L_cls | *run C* | *run C* | *run C* | *run C* |
| 6 | Full − L_tmp | *run D* | *run D* | *run D* | *run D* |
| 7 | Full − L_grd | *run E* | *run E* | *run E* | *run E* |
| 8 | No FiLM conditioning (`--zero_film`) | *run F* | *run F* | *run F* | *run F* |

---

## 2. Critical audit finding that reshapes the design

The temporal loss **L_tmp is declared with weight 0.2 in `configs/train/contest.yaml` but is dead code in the published training.**

Mechanism of the bug:

- `src/losses/physics_losses.py:240–277` implements `temporal_consistency_loss(phi_hat, A, coh_weights)` correctly.
- `InSARLoss.forward` at `physics_losses.py:258–263` only activates L_tmp when **both** `phi_stack` and `sbas_A` are non-None on the `PhysicsLossInputs` container.
- The original `experiments/enhanced/train_film_unet.py` constructs `PhysicsLossInputs` at **L519–524** (main batch) and **L565–574** (triplet batch) **without ever populating `phi_stack` or `sbas_A`**. So the guard at L260 always returns 0. The fork `train_film_unet_ltmp_active_20261113.py` fixes this.

Evidence:

- Published checkpoint's `experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_training_summary.json` records `"temporal": 0.0` in `final_val_metrics`, confirming L_tmp fired zero gradient for the entire 50-epoch training run.
- The `tri_*` loss keys in `logs/AOI024_train_film_unet_20260405_1433.log` confirm the triplet loop's temporal breakdown is also zero.

**Implication**: the published recipe's *effective* loss stack is **`{L_n2n, L_unc, L_cls (via triplet loader), L_grd}`** — four active losses, not five. A naïve `--loss_temporal 0` ablation on the published code would therefore produce identical numbers to the full baseline (null ablation). This plan **fixes L_tmp first**, then runs the ablations against the new L_tmp-active baseline.

---

## 3. Design decisions (all made, locked in)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Handle dead L_tmp | **Fix L_tmp first**, retrain full baseline with active L_tmp, then run LOO | Honest ablation; keeps Table II at 8 rows; disclosable in paper |
| Epochs per run | **20 epochs** | Matches existing `scripts/run_ablations.sh` convention; adequate for ablation; half the published 50-epoch budget |
| Orchestration | **New script** `scripts/run_jstars_ablations_20261113.sh` | Clean slate, tailored to Table II rows; does not shoehorn into legacy V1–V5 ablation infra |
| Compute | **1 GPU serial over nights/weekend** | ~5 days elapsed total for 6 runs |
| Reproducibility | Same random seed, same tile stride (128), same optimizer config across all 6 runs | Only the loss / conditioning differs |

---

## 4. Workstream A — Make L_tmp actually fire

### A.1 Build an epoch registry at training setup

In `main()` of `train_film_unet_ltmp_active_20261113.py`, after `pair_dirs = discover_pair_dirs(pairs_dir)` and before dataset construction:

- Scan all `pair_dir.name` for the training manifest. Dir names follow `<id_ref>__<id_sec>` (verified: samples from `data/processed/AOI000_pairs_full_image/` match this convention — e.g. `CAPELLA_C13_SP_SLC_HH_20250328113256_20250328113306__CAPELLA_C13_SP_SLC_HH_20250412061636_20250412061646`).
- Take the sorted unique set of all `id_ref` and `id_sec` strings. Map to integer indices `{acquisition_id: int}`.
- Persist to `<out_dir>/<run_tag>_epoch_registry.json` after `out_dir` is created.

Pseudocode:
```python
all_ids = set()
for pd in pair_dirs:
    parts = pd.name.split("__")
    if len(parts) == 2:
        all_ids.update(parts)
epoch_map = {aid: idx for idx, aid in enumerate(sorted(all_ids))}
```

### A.2 Thread epoch indices through the dataset

`InSARTileDataset.__init__` (L116–150) gains an `epoch_map: Optional[Dict[str, int]] = None` parameter. Store on `self.epoch_map`.

`InSARTileDataset.__getitem__` (L258–287) parses `pair_dir.name.split("__")` and emits two extra tensors in the sample dict:

```python
parts = pair_dir.name.split("__")
e_i = self.epoch_map.get(parts[0], -1) if len(parts) == 2 else -1
e_j = self.epoch_map.get(parts[1], -1) if len(parts) == 2 else -1
# add to return dict:
"epoch_i": torch.tensor(e_i, dtype=torch.long),
"epoch_j": torch.tensor(e_j, dtype=torch.long),
```

`-1` is the sentinel for "no mapping available" — the training loop skips L_tmp for a batch that has any `-1`.

### A.3 Inject phi_stack + sbas_A into the training loop

In `run_epoch` (L491–585), inside the main batch loop, after `denoised, log_var = model(x, meta)` (L517), build:

- `phi_hat`: `atan2(denoised[:, 1], denoised[:, 0])` → shape `(B, H, W)`
- `phi_stack`: `phi_hat.flatten(1)` → shape `(B, H*W)`
- `coh_stack`: `x[:, 2].flatten(1)` → shape `(B, H*W)` (coherence is channel 2 of the 3-channel input when `in_channels == 3`)
- `sbas_A`: `(B, E)` sparse-ish design matrix built from `batch["epoch_i"]`, `batch["epoch_j"]`:
    - `unique = torch.cat([e_i, e_j]).unique()`
    - Local mapping: `pos = {ep.item(): k for k, ep in enumerate(unique)}`
    - For each pair `k` in batch: `A[k, pos(e_i[k])] = -1`, `A[k, pos(e_j[k])] = +1`

Only build these if `--loss_temporal > 0` *and* no batch entry has `epoch_i = -1`. Pass into `PhysicsLossInputs(..., phi_stack=phi_stack, sbas_A=A, coh_stack=coh_stack)`.

**The triplet loop (L538–583) is NOT touched** — it keeps its L_cls focus. A 3-pair SBAS matrix is trivially tiny and uninformative for L_tmp.

### A.4 Disclose the stochastic surrogate in the paper

A random mini-batch of 16 pairs typically touches ~27 unique epochs (per AOI_000's 221 acquisitions × 6,149 pairs → density ~27 pairs/epoch). The per-batch SBAS fit is therefore a **local stochastic SBAS surrogate**, not the full-stack SBAS that Table I reports. Disclose explicitly:

- Add one sentence in `\subsubsection{Pseudo-Supervised Training Objective}` of the tex (around L289–302) noting the per-batch surrogate.
- Mention in the Table II caption that the −L_tmp row ablates against the L_tmp-active baseline (not against the published 50-epoch recipe).

### A.5 Smoke test before any 20-epoch run

Minimal command:
```bash
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet_ltmp_active_20261113.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --epochs 2 --loss_n2n 0 --loss_unc 0 --loss_closure 0 \
    --loss_grad 0 --loss_temporal 1.0 \
    --run_name smoke_test_tmp 2>&1 | tee logs/smoke_test_tmp.log
```

Expectations:
- `train_temporal` logs non-zero, non-NaN (plausible rad² range, typically O(0.1–10)).
- No `lstsq` errors or `UserWarning: torch.linalg.lstsq`.
- Gradients flow through L_tmp (model params update between epoch 1 and 2).

---

## 5. Workstream B — The six training runs

All six runs share the same base configuration:
- Data: `configs/data/capella_aoi_selection.yaml` (Hawaii subset, 70/15/15 temporal split, 256×256 tiles, stride 128)
- Model: `configs/model/film_unet.yaml` (features `[32,64,128,256]`, embed_dim 64, metadata_dim 7)
- Train: `configs/train/contest.yaml` with `--epochs 20` override (half of the published 50)
- Optimizer: AdamW 1e-4, weight_decay 1e-5, cosine annealing, 2-epoch warmup, batch 16, grad_clip 1.0
- Random seed: 42 (default in `contest.yaml`)
- Triplet manifest: `data/manifests/full_index_triplets_full_image.parquet` (where applicable)
- Output dir: `experiments/enhanced/checkpoints/film_unet/<run_tag>/`

Run matrix — each row shows only the flags that differ from the full baseline:

| Tag | Variant | Flags that differ | Approx. runtime |
|---|---|---|---|
| `jstars_full_tmp_active` | Full objective, L_tmp active | (baseline, no diff) | ~23 h |
| `jstars_abl_no_unc` | Full − L_unc | `--loss_unc 0` | ~23 h |
| `jstars_abl_no_cls` | Full − L_cls | **drop** `--triplets_manifest` | ~12 h |
| `jstars_abl_no_tmp` | Full − L_tmp | `--loss_temporal 0` | ~23 h |
| `jstars_abl_no_grd` | Full − L_grd | `--loss_grad 0` | ~23 h |
| `jstars_abl_zero_film` | No FiLM conditioning | `--zero_film` | ~23 h |

**Launch order**:
1. `jstars_full_tmp_active` — this is the new anchor; every other row is reported *relative to this*, not the published IGARSS numbers.
2. `jstars_abl_no_cls` — fastest (~12 h, no triplet loader), good overnight sanity run.
3. The four 23-h LOO runs serially: `jstars_abl_no_unc`, `jstars_abl_no_grd`, `jstars_abl_no_tmp`, `jstars_abl_zero_film`.

Total wall-clock: ~5 GPU-days on one GPU, serial.

---

## 6. Workstream C — Orchestration script

New file: `scripts/run_jstars_ablations_20261113.sh` (~80 lines, bash).

Responsibilities:
1. Set `CUDA_VISIBLE_DEVICES=0` explicitly; refuse to start if `nvidia-smi --query-gpu=utilization.gpu` shows > 50% usage.
2. Define `BASE_FLAGS` once — the published recipe.
3. For each run tag, define only the diff flags.
4. Sequentially:
   - Record start time to `logs/jstars_ablation_log.csv`.
   - Launch `train_film_unet_ltmp_active_20261113.py` with proper env (`LD_LIBRARY_PATH`, `PYTHONPATH`, `conda run --prefix`).
   - On successful training, run the two-step eval from `REPRODUCIBILITY.md:189–215`:
     - `eval/compute_metrics.py --force_inference --skip_snaphu_metrics` (does inference → writes `ifg_film_unet.tif`)
     - `scripts/unwrap_snaphu.py --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif`
     - `eval/compute_metrics.py --snaphu_only` (M2, M3, M4)
   - Append one row to `logs/jstars_ablation_log.csv` with: `tag, start, end, duration_h, closure_m1, unwrap_m2, nmad_m4, temporal_m5, checkpoint_path`.
5. Emit a summary table at the end (`awk` / `column -t` the CSV) so Table II values are copy-paste-ready for the tex.
6. CLI flags:
   - `--dry-run` — echo the commands without executing (mandatory first run for review).
   - `--only <tag>` — re-run a single variant (e.g., for a crash recovery).
   - `--skip-eval` — train only, no post-eval (useful if eval breaks mid-sweep).

---

## 7. Workstream D — Paper edits after numbers land

Inside `JSTARS_Journal_Paper/JSTARS_Journal_Paper.tex`:

1. Replace all 24 `TBD` entries in Table II (L458–470) with real numbers from `logs/jstars_ablation_log.csv`.
2. Strip `\textcolor{red}{…}` wrappers from L414–435 (ablation-intent paragraphs) and L549–551 (Discussion).
3. Add one sentence in `\subsubsection{Pseudo-Supervised Training Objective}` (around L289–302) noting that L_tmp uses a *per-batch stochastic SBAS surrogate* (Workstream A.4 disclosure).
4. Update the Table II caption: drop "Published and planned … TBD indicates measurements to be obtained …"; replace with a tight caption naming the final matrix.
5. **Reproducibility-check text**: if `jstars_full_tmp_active` matches the already-published 0.915 / 0.367 within ±2%, add a one-line statement to that effect. If it diverges materially (because L_tmp is now active and shifts the optimum), document the delta honestly — do not fudge the numbers to match.
6. If any ablation row shows removing a loss is *net-beneficial* (ablation outperforms full baseline), add a short interpretation in the Discussion.

---

## 8. Critical files

| Path | Status | Workstream |
|------|--------|------------|
| `src/losses/physics_losses.py` | read-only reference | audit (done) |
| `experiments/enhanced/train_film_unet.py` | **preserved verbatim** — original, do not touch | — |
| `experiments/enhanced/train_film_unet_ltmp_active_20261113.py` | **new file** (fork of original with L_tmp wired in) | A.1, A.2, A.3 |
| `scripts/run_jstars_ablations_20261113.sh` | new file | C |
| `JSTARS_Journal_Paper/JSTARS_Journal_Paper.tex` | modify after numbers land | D |
| `logs/jstars_ablation_log.csv` | new, written by C | C |
| `<out_dir>/<run_tag>_epoch_registry.json` | new, written per run | A.1 |
| `<out_dir>/<run_tag>_training_summary.json` | already written by existing code | B |
| `configs/train/contest.yaml` | no change | — |

---

## 9. Verification

1. **Smoke test** (A.5): `train_temporal` logs plausible non-zero values; no NaNs; no lstsq errors.
2. **Anchor-run reproducibility**: `jstars_full_tmp_active` numbers within ±5% of published Table I FiLM-GPNet AOI_000 row (20 vs 50 epochs + L_tmp newly active will shift the optimum; ±5% is an acceptable band).
3. **Ablation direction sanity**: each `−L_x` row is either worse or comparable to the full baseline. A materially-better LOO is a bug signal — investigate before publishing.
4. **End-to-end dry-run**: `bash scripts/run_jstars_ablations.sh --dry-run` prints six matching command lines; manual eyeball check before the real sweep.
5. **Tex compile post-backfill**: `pdflatex JSTARS_Journal_Paper.tex && bibtex && pdflatex && pdflatex` — no undefined references, all figures resolve, bibliography clean. `grep -n '\\\\textcolor{red}' JSTARS_Journal_Paper.tex` returns zero matches related to Table II.

---

## 10. Risks and fallbacks

| Risk | Likelihood | Fallback |
|------|-----------|----------|
| Mini-batch SBAS is too noisy; L_tmp training diverges | Medium | Lower `--loss_temporal` from 0.2 to 0.05, or accumulate predictions over K mini-batches before computing the SBAS fit |
| Adding `epoch_i`/`epoch_j` to the dataloader breaks the triplet loader | Low | Smoke test exercises both loaders; `TripletTileDataset` is untouched by the plan |
| 5-day serial queue is interrupted (job killer, reboot) | Medium | Orchestration script persists state in `jstars_ablation_log.csv`; `--only <tag>` lets us resume per-variant |
| "−L_cls" (drop triplets) run converges to a different optimum because the triplet loader also provides extra `L_n2n + L_unc + L_grd` contributions | **Known, disclose in paper** | Our "−L_cls" is honestly "no triplet batch", which matches what deployment-without-triplets would look like |
| Per-pair coherence isn't representative for the SBAS weighting | Low | `coh_weights` is passed but averaged per pair inside `temporal_consistency_loss`; already implemented |
| Pair-dir names don't follow `<id_ref>__<id_sec>` on some AOIs | Low | Sentinel `-1` in epoch_i/epoch_j skips L_tmp for that batch; log a count of skipped batches per epoch |

---

## 11. Timeline

Assuming kickoff **2026-11-13**:

| Phase | Work | Elapsed | Cumulative |
|-------|------|---------|------------|
| Code changes (A.1–A.3) | ~2 h | — | Day 1 |
| Smoke test (A.5) | ~30 min | — | Day 1 |
| Write orchestration (C) | ~2 h | — | Day 1 |
| Run 1 (`jstars_full_tmp_active`) | 23 h | 1 day | Day 2 |
| Run 2 (`jstars_abl_no_cls`) | 12 h | 0.5 day | Day 2–3 |
| Run 3 (`jstars_abl_no_unc`) | 23 h | 1 day | Day 3–4 |
| Run 4 (`jstars_abl_no_grd`) | 23 h | 1 day | Day 4–5 |
| Run 5 (`jstars_abl_no_tmp`) | 23 h | 1 day | Day 5–6 |
| Run 6 (`jstars_abl_zero_film`) | 23 h | 1 day | Day 6–7 |
| Eval collation + tex backfill (D) | ~2 h | — | Day 7 |

**Total: ~1 week** on a single dedicated GPU.

---

## 12. Reproducibility checklist (for co-authors reviewing results)

After the sweep completes, the artefacts a reviewer / co-author can inspect:

- [ ] `experiments/enhanced/checkpoints/film_unet/jstars_*/` — one directory per variant with `_final.pt`, `_training_summary.json`, `_epoch_registry.json`.
- [ ] `logs/jstars_*.log` — one training log per variant.
- [ ] `logs/jstars_ablation_log.csv` — tabular summary with all four metrics per variant.
- [ ] `experiments/enhanced/outputs/jstars_*/metrics_comparison.csv` — per-pair M1–M5 CSVs feeding the aggregate Table II numbers.
- [ ] `data/processed/AOI000_pairs_full_image/*/ifg_film_unet.tif` and `unw_phase_film_unet.tif` — per-run intermediate artefacts (will overwrite between runs unless we add a run-tag suffix; decision below).

**Open design question** — do we want per-variant `ifg_film_unet.tif` to be kept separately (e.g., `ifg_film_unet_jstars_abl_no_unc.tif`)? Pros: reviewers can see per-variant qualitative outputs. Cons: 6× disk usage on `data/processed/AOI000_pairs_full_image/`. Defer decision until sweep is running.

---

## 13. What this plan does NOT include

- Edits to the paper's **Introduction, Related Work, Conclusion, Abstract, or Keywords** — those live in the shared joint-paper sections and are being written across teams.
- The **non-Table-II cleanups** (typos, caption fixes, bib dedup, ring-artifact investigation in DEM-Comp.png) — tracked separately; can be addressed on a sibling feature branch.
- Ablations on **AOI_024 or AOI_008** — Hawaii only, per agreed scope.
- A **bar chart / visualization** for Table II — can be added if the numbers tell an interesting story after the fact.
- **Fixing the bib duplicates** (`Zebker1992`/`zebker1992decorrelation`, `Hanssen2001`/`hanssen2001radar`) — sibling branch.

---

## 14. Change log

| Date | Author | Change |
|------|--------|--------|
| 2026-11-13 | Getnet Demil | Initial plan after audit of training loop and loss implementations |
| 2026-11-13 | Getnet Demil | Fork `train_film_unet_ltmp_active_20261113.py` built; `scripts/run_jstars_ablations_20261113.sh` and `configs/data/capella_aoi000_ltmp_ablation_20261113.yaml` added; pushed to GitHub branch `JSTARS-ablation` (commit `b6c45eb`). |
| 2026-11-14 | Getnet Demil | Data-path correction: config points at `data/processed/pairs/` (224 legacy Hawaii pairs, 2-band Re/Im `ifg_goldstein.tif`) instead of `AOI000_pairs_full_image/` (only 4 completed pairs). Fork accepts both filename conventions. |
| 2026-11-14 | Getnet Demil | Epoch registry resolves `id_ref`/`id_sec` from each pair's `coreg_meta.json` (not from `pair_dir.name.split("__")`) because 100 of 224 legacy dir names are truncated at 80 chars. |
| 2026-11-14 | Getnet Demil | Monkey-patched `temporal_consistency_loss` with a ridge-regularized version inside the fork (λ=1e-3). Original lstsq produced NaN on CUDA for under-determined mini-batch SBAS; ridge eliminates it. Published `physics_losses.py` untouched. |
| 2026-11-14 | Getnet Demil | Orchestrator reordered (fastest `jstars_abl_no_cls` first for autonomous-window ergonomics) and hardened to continue-on-error so one bad variant does not abort the sweep. |
| 2026-11-14 | Getnet Demil | Fixed `pd` shadow bug in `main()` (loop variable `for pd in pair_dirs` was shadowing `import pandas as pd`, breaking `pd.read_parquet(args.triplets_manifest)` 50 lines later). Loop var renamed to `pdir`. |

---

## 15. Execution log

Live, append-only record of actual runs. Metrics come from `<run_tag>_training_summary.json`; eval (M1–M5) is deferred until the full sweep completes.

| # | Tag | Status | Start → End (UTC) | Duration | Final train_tmp | Final train_total | Notes |
|---|-----|--------|-------------------|----------|-----------------|-------------------|-------|
| 1 | `jstars_abl_no_cls_20261114_0128` | ✅ complete | 23:28 Nov 13 → 05:38 Nov 14 | 6.16 h | 0.0738 | −0.506 | No triplet loader. `_n_tmp_fired: 9369/9369` on every epoch. Clean. |
| 2 | `jstars_full_tmp_active_20261114_0740` | 🟢 in flight | 05:40 Nov 14 → ~02:30 Nov 15 (est) | ~21 h (est) | 0.0752 @ epoch 4 | −0.437 @ epoch 4 | Anchor row. With triplets (`hawaii_triplets_strict.parquet`, 24,171 triplets → 51,150 triplet-tiles). |
| 3 | `jstars_abl_no_unc` | ⏳ queued | pending | — | — | — | `--loss_unc 0`, triplets on |
| 4 | `jstars_abl_no_grd` | ⏳ queued | pending | — | — | — | `--loss_grad 0`, triplets on |
| 5 | `jstars_abl_no_tmp` | ⏳ queued | pending | — | — | — | `--loss_temporal 0`, triplets on |
| 6 | `jstars_abl_zero_film` | ⏳ queued | pending | — | — | — | `--zero_film`, triplets on |

Sweep master logs:
- `logs/jstars_sweep_master_20261113_2328.log` — sweep #1 (variant 1 complete + variant 2 crash evidence)
- `logs/jstars_sweep_master_20261114_0540.log` — sweep #2 (relaunch, currently in variant 2)

Per-variant CSVs:
- `logs/jstars_ablation_log_20261113_2328.csv` — variant 1 row
- `logs/jstars_ablation_log_20261114_0540.csv` — variants 2–6 (will populate as each completes)

Smoke tests (all on 2026-11-13/14, logs under `logs/smoke_test_*`):
- v1 / v2: `train_tmp=nan` — root cause was lstsq instability on CUDA under-determined mini-batch SBAS
- v3: `train_tmp=nan` even with production weights — confirmed not a weight issue, it's the solver
- v4: `train_tmp=0.074` stable over 2 epochs after installing ridge-regularized replacement; cleared to launch sweep
