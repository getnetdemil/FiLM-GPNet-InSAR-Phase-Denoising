#!/usr/bin/env bash
# One-screen status for the JSTARS eval sweep. Pipe through `watch` to refresh.
#   watch -c -n 30 bash scripts/status_jstars_eval_20261119.sh
# -c keeps ANSI colors, -n 30 refreshes every 30 s.
set -u
cd /scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
TS="${EVAL_TS:-20261119_0926}"
BOLD=$'\e[1m'; DIM=$'\e[2m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; BLUE=$'\e[36m'; RST=$'\e[0m'

VARIANTS=(
    jstars_full_tmp_active
    jstars_abl_no_unc
    jstars_abl_no_cls
    jstars_abl_no_tmp
    jstars_abl_no_grd
    jstars_abl_zero_film
)

echo "${BOLD}JSTARS eval status  — $(date -u +%FT%TZ)   EVAL_TS=$TS${RST}"
echo

# --- GPU -------------------------------------------------------------------
printf "${BOLD}GPU${RST}  "
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '{printf "cuda%s %s%% %sMiB  ", $1, $2, $3}'
echo

# --- Stage A (inference + M1 + M5) -----------------------------------------
echo "${BOLD}Stage A — inference + M1 closure + M5 temporal${RST}"
printf "  %-24s %s %s %s\n" "variant" "infer" "closure(G→M)" "temporal(G→M)"
for t in "${VARIANTS[@]}"; do
    L="logs/JSTARS_eval_infer_${t}_${TS}.log"
    if [[ ! -f "$L" ]]; then
        printf "  %-24s %s\n" "$t" "${DIM}pending${RST}"
        continue
    fi
    n=$(grep -c "Inference: " "$L" 2>/dev/null || echo 0)
    done_=$(grep -c "Inference complete" "$L" 2>/dev/null || echo 0)
    m1g=$(grep "Triplet Closure Error" "$L" 2>/dev/null | tail -1 | awk '{print $5"→"$6}')
    m5g=$(grep "Temporal Residual"    "$L" 2>/dev/null | tail -1 | awk '{print $4"→"$5}')
    if [[ "$done_" -gt 0 && -n "$m1g" ]]; then
        tag="${GREEN}done${RST}"
    elif [[ "$done_" -gt 0 ]]; then
        tag="${YELLOW}metrics${RST}"
    else
        tag="${BLUE}${n}/224${RST}"
    fi
    printf "  %-24s %-18s %-14s %-14s\n" "$t" "$tag" "${m1g:-—}" "${m5g:-—}"
done
echo

# --- Stage B (SNAPHU unwrap on SLURM) --------------------------------------
echo "${BOLD}Stage B — SNAPHU on SLURM${RST}"
printf "  %-24s %s %s %s\n" "variant" "unwrapped" "SLURM" "partition"
for t in "${VARIANTS[@]}"; do
    TREE="data/processed/JSTARS_eval_${t}_${TS}"
    if [[ ! -d "$TREE" ]]; then
        printf "  %-24s %s\n" "$t" "${DIM}no tree yet${RST}"
        continue
    fi
    nunw=$(find "$TREE" -mindepth 2 -maxdepth 2 -name unw_phase_film_unet.tif -type f 2>/dev/null | wc -l)
    ntot=$(find "$TREE" -mindepth 2 -maxdepth 2 -name ifg_film_unet.tif -type f 2>/dev/null | wc -l)
    [[ "$ntot" -eq 0 ]] && ntot=224
    if [[ "$nunw" -eq "$ntot" && "$ntot" -gt 0 ]]; then pct="${GREEN}${nunw}/${ntot}${RST}"
    elif [[ "$nunw" -eq 0 ]]; then pct="${DIM}0/${ntot}${RST}"
    else pct="${BLUE}${nunw}/${ntot}${RST}"
    fi
    # match the canonical job name OR any "..._itee_..." resubmit of the same variant
    # Collect parts with dedupe via sort|uniq from the shell, not awk
    sq=$(squeue -u "$USER" -h -o "%j|%t|%r|%P" \
        | awk -F'|' -v tag="JSTARS_snaphu_${t}_" 'index($1, tag)==1 {s[$2]++; part[$4]=1} END {
            out=""; for (k in s) out=out k ":" s[k] " ";
            if (out=="") out="—";
            p=""; for (k in part) p=(p?p","k:k); if (p=="") p="—";
            print out "|" p
        }')
    state="${sq%|*}"; part_csv="${sq#*|}"
    # Dedupe partition CSV in shell (portable, no gawk needed)
    if [[ "$part_csv" != "—" ]]; then
        part_csv=$(echo "$part_csv" | tr ',' '\n' | sort -u | paste -sd,)
    fi
    sq="${state}|${part_csv}"
    state="${sq%|*}"; part="${sq#*|}"
    printf "  %-24s %-14s %-24s %s\n" "$t" "$pct" "$state" "$part"
done
echo

# --- Watchers --------------------------------------------------------------
echo "${BOLD}Watchers${RST}"
_watcher() {
    local label="$1" pattern="$2" logglob="$3" needed_msg="$4"
    local p=$(pgrep -af "$pattern" | grep -v pgrep | head -1)
    if [[ -n "$p" ]]; then
        printf "  %-18s ${GREEN}alive${RST}  pid=%s\n" "$label" "$(echo "$p" | awk '{print $1}')"
        local wlog=$(ls -t $logglob 2>/dev/null | head -1)
        [[ -n "$wlog" ]] && tail -1 "$wlog" | sed 's/^/     /'
    else
        printf "  %-18s ${DIM}done${RST}  (%s)\n" "$label" "$needed_msg"
        local wlog=$(ls -t $logglob 2>/dev/null | head -1)
        [[ -n "$wlog" ]] && tail -1 "$wlog" | sed 's/^/     /'
    fi
}
_watcher "snaphu-submit"  "watch_submit_jstars_snaphu"  "logs/JSTARS_snaphu_watcher_*.log"    "all 5 non-anchor arrays already queued; nothing to resubmit"
_watcher "stage-C (M2/M4)" "watch_stage_c_jstars"       "logs/JSTARS_stage_c_watcher_*.log"   "fires M2/M4 as each variant hits 224/224 unwrapped"
echo

# --- SLURM queue summary ---------------------------------------------------
echo "${BOLD}SLURM queue${RST}  (our jobs only)"
squeue -u "$USER" -h -o "%.18i %.10P %.2t %.10M %.42j %R" | grep -i jstars \
  | awk '{
       # compress array task runs to just counts
       if (match($5, /^JSTARS/)) {
         tag=$5; state=$3; part=$2;
         if (match($1, /\[/)) { kind="PD" } else { kind="R" }
         k=tag"|"part"|"state
         c[k]++
       }
     } END {
       for (k in c) {
         split(k, a, "|"); printf "  %-10s %-2s %-45s x%d\n", a[2], a[3], a[1], c[k]
       }
     }' | sort
R=$(squeue -u "$USER" -h -t R | wc -l)
PD=$(squeue -u "$USER" -h -t PD | wc -l)
printf "  total: %sR${RST} %sPD${RST}\n" "${GREEN}${R}" "${YELLOW}${PD}"
