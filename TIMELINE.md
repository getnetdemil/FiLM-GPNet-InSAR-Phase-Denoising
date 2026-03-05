# 6-Week Timeline: IEEE GRSS DFC 2026
## Start: February 19, 2026 | Deadline: April 6, 2026

---

## Overview

```
Week 1  │ Feb 19–25 │ Data access + exploration
Week 2  │ Feb 26–Mar 4  │ Preprocessing pipeline
Week 3  │ Mar 5–11  │ Architecture adaptation + initial training
Week 4  │ Mar 12–18 │ Full-scale training + ablations
Week 5  │ Mar 19–25 │ Analysis + storytelling visuals
Week 6  │ Mar 26–Apr 6  │ Paper writing + final submission
         │ Apr 6    │ *** SUBMISSION DEADLINE ***
```

---

## Week 1: Data Access & Exploration (Feb 19–25)

### Goals
- [ ] Set up Capella STAC API access
- [ ] Download representative temporal stacks (≥3 regions)
- [ ] Understand X-band SAR characteristics vs Sentinel-1 C-band
- [ ] Identify best target regions for change detection

### Tasks

**Day 1-2 (Feb 19-20):**
- [ ] Install dependencies: `pystac-client`, `boto3`, `rasterio`, `shapely`
- [ ] Run `data/explore_stac.py` to query the STAC catalog
- [ ] Map available temporal stacks by region and density

**Day 3-4 (Feb 21-22):**
- [ ] Run `data/select_stacks.py` to identify top stacks by:
  - Temporal depth (many acquisitions)
  - Scene diversity (urban, agricultural, coastal)
  - Geometry consistency (similar incidence angles)
- [ ] Download 2-3 pilot stacks via `data/download_capella_data.py`

**Day 5-7 (Feb 23-25):**
- [ ] Visual inspection of GEO images in QGIS / Python
- [ ] Compare X-band appearance vs Sentinel-1 (resolution, texture)
- [ ] Run `notebooks/01_data_exploration.ipynb`
- [ ] Document key observations in `PROGRESS.md`

### Deliverable
- 2-3 temporal stacks downloaded (~50-100 image pairs)
- Data exploration notebook completed
- Key scenes and regions identified

---

## Week 2: Preprocessing Pipeline (Feb 26 – Mar 4)

### Goals
- [ ] Adapt S1 preprocessing to Capella X-band SAR
- [ ] Build temporal pair dataset
- [ ] Implement PyTorch Dataset class with augmentation
- [ ] Verify data loading pipeline end-to-end

### Tasks

**Day 1-3 (Feb 26–28):**
- [ ] Implement `preprocessing/capella_preprocessor.py`
  - Lee speckle filter (adapted for X-band)
  - dB conversion: `20 * log10(amplitude)` for X-band
  - Normalization: calibrate to X-band dynamic range
  - Co-registration quality check between temporal pairs
- [ ] Test on downloaded pilot data

**Day 4-5 (Mar 1-2):**
- [ ] Implement `preprocessing/temporal_pair_builder.py`
  - Parse STAC metadata for temporal ordering
  - Build all valid pairs per stack (gap filter: 7–180 days)
  - Compute amplitude-difference pseudo-labels
  - Save pair manifests as JSON

**Day 6-7 (Mar 3-4):**
- [ ] Implement `preprocessing/change_dataset.py`
  - PyTorch Dataset loading temporal pairs
  - Patch extraction (256×256 with 128-pixel stride)
  - SAR-specific augmentation pipeline
  - Train/val/test split
- [ ] Verify: `python -c "from preprocessing.change_dataset import ChangeDetectionDataset; ..."`
- [ ] Update `PROGRESS.md`

### Deliverable
- Full preprocessing pipeline working
- ~10,000 temporal patch pairs ready for training
- Dataset statistics documented

---

## Week 3: Architecture Adaptation + Initial Training (Mar 5–11)

### Goals
- [ ] Implement `TemporalCrossFuse` model
- [ ] Implement training losses (BCE + Dice + contrastive + reconstruction)
- [ ] Verify on LEVIR-CD (labeled benchmark) to check architecture validity
- [ ] First Capella training run

### Tasks

**Day 1-2 (Mar 5-6):**
- [ ] Implement `models/siamese_encoder.py` (shared-weight CNN backbone)
- [ ] Implement `models/temporal_attention.py` (bidirectional cross-attention, reuse from CrossFuse)
- [ ] Implement `models/change_decoder.py` (U-Net style decoder with skip connections)
- [ ] Integrate into `models/temporal_crossfuse.py`
- [ ] Unit test: `python -c "from models.temporal_crossfuse import TemporalCrossFuse; ..."`

**Day 3-4 (Mar 7-8):**
- [ ] Implement `training/losses.py`
  - `BCEWithLogitsLoss` for pseudo-labels
  - `DiceLoss` for spatial coverage
  - `ContrastiveLoss` for self-supervised signal
  - `ReconstructionLoss` for physical consistency
  - `CombinedChangeLoss` with learnable weights
- [ ] Implement `training/metrics.py`
  - F1, IoU, Precision, Recall (for LEVIR-CD validation)
  - Temporal consistency score (Capella)
  - Change area distribution statistics

**Day 5-6 (Mar 9-10):**
- [ ] Quick validation on LEVIR-CD dataset
  - Download LEVIR-CD: https://justchenhao.github.io/LEVIR/
  - 5-epoch quick test: `python training/train.py --config training/config.yaml --quick_test`
  - Expected: F1 > 0.65 within 5 epochs (proves architecture works)
- [ ] First Capella training run (small scale, 2 stacks, 10 epochs)

**Day 7 (Mar 11):**
- [ ] Analyze first results, debug any issues
- [ ] Update `PROGRESS.md` with architecture status

### Deliverable
- TemporalCrossFuse model implemented and verified
- LEVIR-CD quick test F1 > 0.65
- First Capella change maps generated

---

## Week 4: Full-Scale Training + Ablations (Mar 12–18)

### Goals
- [ ] Full training on all downloaded Capella data
- [ ] Run ablation study (4 variants)
- [ ] Generate quantitative results table
- [ ] Download additional Capella stacks if needed

### Tasks

**Day 1-2 (Mar 12-13):**
- [ ] Download additional Capella stacks (target: 5+ diverse regions)
- [ ] Run full preprocessing on all data
- [ ] Launch full training: `bash scripts/full_train.sh`
  - 50 epochs, all data, mixed precision
  - Monitor: loss curves, sample change maps per 10 epochs

**Day 3-4 (Mar 14-15):**
- [ ] Run ablation study via `experiments/ablation.py`:
  - Variant A: Difference baseline (no attention, just |T2-T1|)
  - Variant B: Concatenation fusion (early fusion, no attention)
  - Variant C: Unidirectional attention only (T1→T2)
  - Variant D: Full TemporalCrossFuse (bidirectional + transformer)
- [ ] Save all results to `experiments/results/`

**Day 5-7 (Mar 16-18):**
- [ ] Full evaluation on test split
- [ ] Generate comparison tables (LEVIR-CD + temporal consistency)
- [ ] Identify 3-4 best case studies for paper figures
- [ ] Update `PROGRESS.md`

### Deliverable
- Fully trained model with best checkpoint saved
- Ablation table showing contribution of each component
- 3+ compelling case studies identified

---

## Week 5: Analysis + Temporal Storytelling Visuals (Mar 19–25)

### Goals
- [ ] Generate all paper figures
- [ ] Create temporal animations (GIFs)
- [ ] Thematic classification of change types
- [ ] Prepare all visual storytelling materials

### Tasks

**Day 1-3 (Mar 19-21):**
- [ ] Run `evaluation/visualize.py` on best case studies:
  - Figure 1: Architecture diagram
  - Figure 2: 3×3 grid (T1 | T2 | Change map) for 3 scene types
  - Figure 3: Ablation comparison
  - Figure 4: Temporal stack animation (if time permits)
- [ ] Create before/after composites for urban, agricultural, infrastructure

**Day 4-5 (Mar 22-23):**
- [ ] Run `evaluation/temporal_story.py`
  - Multi-temporal change accumulation visualization
  - Change trend plots (% area changed over time)
  - Scene-type narrative summaries

**Day 6-7 (Mar 24-25):**
- [ ] Polish all figures to publication quality
  - 300 DPI minimum
  - Consistent colormap (perceptually uniform)
  - Clear labels and scalebars
- [ ] Prepare supplemental visualizations (high-resolution versions)
- [ ] Update `PROGRESS.md`

### Deliverable
- All paper figures ready (publication quality)
- Temporal animations generated
- GitHub repo cleaned and documented

---

## Week 6: Paper Writing + Submission (Mar 26 – Apr 6)

### Goals
- [ ] Write complete 4-page IEEE paper
- [ ] Clean and document code
- [ ] Submit via Google Form before April 6

### Tasks

**Day 1-3 (Mar 26-28):**
- [ ] Write `paper/main.tex`:
  - Section 1: Introduction
  - Section 2: Methodology (architecture + training)
  - Section 3: Experiments (results + ablations)
  - Section 4: Conclusion
- [ ] Insert all figures with proper captions

**Day 4-5 (Mar 29-30):**
- [ ] Technical review of paper
- [ ] Code cleanup for GitHub release:
  - Remove dead code
  - Add docstrings and comments
  - Create `requirements.txt`
  - Add inference demo script
  - Write GitHub README

**Day 6-7 (Apr 1-2):**
- [ ] Internal proofreading pass
- [ ] Create GitHub repository (public)
- [ ] Prepare submission document (initial concise version)
- [ ] Final check against contest rules

**Buffer Days (Apr 3-6):**
- [ ] Final revisions
- [ ] Submit via Google Form: https://forms.gle/gnmu1fKJAqNcxbK98
- [ ] Email backup: dfc26@googlegroups.com

### Deliverable
- Complete paper PDF
- Public GitHub repository
- Submitted to contest

---

## Critical Path (Must Complete)

```
Week 1: Data downloaded ──▶
Week 2: Pipeline working ──▶
Week 3: Model trains (LEVIR-CD ≥ F1 0.65) ──▶
Week 4: Full training complete ──▶
Week 5: Figures ready ──▶
Week 6: SUBMITTED (Apr 6)
```

If any step is delayed by >2 days, escalate immediately and simplify scope.

---

## Minimum Viable Submission (Fallback)

If time runs short, the minimum credible submission consists of:
1. Simple amplitude-difference change detection with despeckling
2. CNN-based change detector (no transformer — faster to train)
3. 2 case studies with before/after visualization
4. GitHub repo with preprocessing + inference code
5. 4-page paper describing the approach and visual results

This is still a valid and potentially competitive submission given the open-ended nature
of the contest.
