"""
Build an isolated evaluation tree for one JSTARS ablation variant.

Creates  data/processed/JSTARS_eval_<variant>_<YYYYMMDD_HHMM>/<pair_name>/
holding READ-ONLY symlinks to the IGARSS pair inputs in data/processed/pairs/.
Files the eval pipeline writes (ifg_film_unet.tif, log_var.tif,
unw_phase_film_unet.tif) are never symlinked, so they are created as new files
inside the JSTARS tree and the IGARSS outputs are never touched.

Symlink names are mapped to what eval/compute_metrics expects:
    ifg_goldstein_complex_real_imag.tif -> pairs/<p>/ifg_goldstein.tif
    ifg_raw_complex_real_imag.tif       -> pairs/<p>/ifg_raw.tif
    coherence.tif, coreg_meta.json, unw_phase.tif (Goldstein baseline unwrap)

Prints the created tree path on the last stdout line.

Usage:
    python scripts/build_jstars_eval_tree_20261119.py --variant jstars_full_tmp_active
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINKS = {
    "ifg_goldstein_complex_real_imag.tif": ("ifg_goldstein_complex_real_imag.tif", "ifg_goldstein.tif"),
    "ifg_raw_complex_real_imag.tif":       ("ifg_raw_complex_real_imag.tif", "ifg_raw.tif"),
    "coherence.tif":                       ("coherence.tif",),
    "coreg_meta.json":                     ("coreg_meta.json",),
    "unw_phase.tif":                       ("unw_phase.tif",),
}
WRITTEN_BY_EVAL = ("ifg_film_unet.tif", "log_var.tif", "unw_phase_film_unet.tif")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--src_pairs_dir", default="data/processed/pairs")
    p.add_argument("--dst_root", default="data/processed")
    p.add_argument("--ts", default=None, help="Timestamp suffix (default: now, YYYYMMDD_HHMM)")
    args = p.parse_args()

    src = (ROOT / args.src_pairs_dir).resolve()
    ts = args.ts or datetime.utcnow().strftime("%Y%m%d_%H%M")
    dst = (ROOT / args.dst_root / f"JSTARS_eval_{args.variant}_{ts}").resolve()
    if dst.exists():
        sys.exit(f"Refusing to reuse existing tree: {dst}")
    dst.mkdir(parents=True)

    n_pairs, n_missing = 0, 0
    for pair in sorted(d for d in src.iterdir() if d.is_dir()):
        out = dst / pair.name
        out.mkdir()
        for link_name, candidates in LINKS.items():
            target = next((pair / c for c in candidates if (pair / c).exists()), None)
            if target is None:
                n_missing += 1
                print(f"WARN missing {link_name} in {pair.name}", file=sys.stderr)
                continue
            os.symlink(target, out / link_name)
        for name in WRITTEN_BY_EVAL:
            assert not (out / name).exists() and not (out / name).is_symlink()
        n_pairs += 1

    print(f"Built {n_pairs} pair dirs ({n_missing} missing inputs) in {dst}", file=sys.stderr)
    print(dst)


if __name__ == "__main__":
    main()
