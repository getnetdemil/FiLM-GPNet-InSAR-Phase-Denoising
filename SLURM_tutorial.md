# SLURM in practice — a guide written while doing it

A walkthrough of how I parallelised 224 × 5 = 1 120 SNAPHU unwraps on the
lehmus cluster for the JSTARS Table II ablation (eval stage B). Every command
is real, every pitfall was hit, every workaround is explained. If you follow
this end-to-end you will understand how to run embarrassingly-parallel work
on a shared SLURM cluster where someone else is always using it.

The whole thing is organised around questions I actually asked while
running the sweep, in the order I asked them. Read top-to-bottom the first
time; use it as a cheat sheet after.

- [1. What SLURM is doing for you](#1-what-slurm-is-doing-for-you)
- [2. Looking at the cluster before you touch it](#2-looking-at-the-cluster-before-you-touch-it)
- [3. Writing a one-pair-per-task SNAPHU batch script](#3-writing-a-one-pair-per-task-snaphu-batch-script)
- [4. One-off vs array jobs; the `%N` throttle](#4-one-off-vs-array-jobs-the-n-throttle)
- [5. Why your jobs aren't starting: reading `Reason=`](#5-why-your-jobs-arent-starting-reading-reason)
- [6. CPU is rarely the bottleneck; memory is](#6-cpu-is-rarely-the-bottleneck-memory-is)
- [7. Node features and constraints — the sapphirerapids story](#7-node-features-and-constraints--the-sapphirerapids-story)
- [8. QOS limits that look like a cluster problem](#8-qos-limits-that-look-like-a-cluster-problem)
- [9. Multi-partition submission and backfill — the big win](#9-multi-partition-submission-and-backfill--the-big-win)
- [10. Updating in-flight jobs with `scontrol update`](#10-updating-in-flight-jobs-with-scontrol-update)
- [11. Nice vs Priority — what users actually control](#11-nice-vs-priority--what-users-actually-control)
- [12. Resubmitting only undone work (idempotent progress)](#12-resubmitting-only-undone-work-idempotent-progress)
- [13. Detached watchers that survive SSH drops](#13-detached-watchers-that-survive-ssh-drops)
- [14. Live monitoring: a one-screen dashboard](#14-live-monitoring-a-one-screen-dashboard)
- [15. Safety guards when outputs mustn't be overwritten](#15-safety-guards-when-outputs-mustnt-be-overwritten)
- [16. The full flow I followed, start to finish](#16-the-full-flow-i-followed-start-to-finish)
- [17. Copy-paste cheat sheet](#17-copy-paste-cheat-sheet)

---

## 1. What SLURM is doing for you

SLURM (Simple Linux Utility for Resource Management) is a batch scheduler
on a shared cluster. It has three concepts you need in muscle memory:

- **Node** — one physical computer. On lehmus, there are many: `lehmus-cn1`
  up to `lehmus-cn15`, plus `-cn2501..2509` and `-testcn1..2`. Each has a
  fixed number of CPU cores and a fixed amount of RAM.
- **Partition** — a *named* pool of nodes. On lehmus: `normal`, `itee`,
  `interactive`, `nmr`, `nmr-small`, `nmr-large`, `debug`. A node can be in
  more than one partition (e.g. cn10–15 are in both `normal` and `itee`).
  Each partition has its own time limits and often its own QOS.
- **Job** — your unit of work. You ask SLURM for *N* CPUs, *X* MB of RAM,
  for *T* time, on a given partition, and it hands you a batch job ID. You
  can also ask for an **array job**: one submission that fans out into N
  numbered tasks sharing the same script, differentiated by
  `$SLURM_ARRAY_TASK_ID`.

SLURM then plays Tetris between your job's requested resources and what
the nodes have free. If it finds a fit, your job starts. If not, your job
waits with a `Reason=` explanation.

The three things SLURM does that nobody tells you about:

1. **Backfill scheduling** — small, short jobs can "sneak in" to gaps that
   larger queued jobs can't yet use. This is the single biggest lever for
   embarrassingly-parallel small jobs on a busy cluster. See §9.
2. **Partitions overlap on nodes** — a node in `normal` AND `itee` means
   jobs from both partitions compete for the same CPUs/RAM on that node.
3. **QOS is a layer separate from partition** — a partition can carry a
   QOS that limits concurrent jobs per user, and that limit bites you even
   when the nodes have plenty of room. See §8.

---

## 2. Looking at the cluster before you touch it

Before you submit anything, learn the lay of the land. The commands below
are what I ran in order on 2026-11-19 when I realised the SNAPHU step
would dominate wall time.

### 2.1 Which partitions exist, how big are they, how busy

```bash
sinfo
# PARTITION   AVAIL  TIMELIMIT   NODES(A/I/O/T) NODELIST
# interactive    up 7-00:00:00          4/1/0/5 lehmus-cn[2-3,7-9]
# normal*        up 14-00:00:0        15/0/0/15 lehmus-cn[1-6,10-15,2507-2509]
# itee           up 14-00:00:0          6/0/0/6 lehmus-cn[10-11,13-15,12]
```

The `A/I/O/T` column is Allocated / Idle / Other / Total. `itee` is
6 nodes, 0 idle right now. `*` on `normal` means it is the default
partition when no `--partition` is given.

For per-node detail including CPU count, RAM, and *feature* tags:

```bash
sinfo -p itee -N -o "%N %c %m %f %T"
# NODELIST     CPUS  MEMORY  AVAIL_FEATURES  STATE
# lehmus-cn10  160   380000  cpu_skylake     mixed-
# lehmus-cn11  160   380000  cpu_skylake     mixed-
# lehmus-cn12  160   380000  cpu_skylake     mixed*
# ...
```

The number after MEMORY is total RAM in MB (380000 = 380 GB). The STATE
suffixes matter:

- `mixed` — some CPUs allocated, some free (good).
- `allocated` — fully in use (bad).
- `idle` — completely free (great).
- `+PLANNED`, `+NOT_RESPONDING`, `+MAINT` — SLURM has reserved the node
  for someone else soon, the node has gone unreachable, or it's in a
  maintenance window. Treat `MAINT`/`NOT_RESPONDING` as unusable.

### 2.2 What resources each node has free *right now*

The counts in `sinfo` lie slightly: a node can be "mixed" but have 99% of
its RAM reserved by one of its running jobs. To see the actual free RAM:

```bash
scontrol show node lehmus-cn10 | grep -E "CPUAlloc|RealMemory|AllocMem|State|Partitions"
#    State=MIXED+PLANNED
#    CPUAlloc=98 CPUEfctv=160 CPUTot=160 CPULoad=22.53
#    RealMemory=380000 AllocMem=379872 FreeMem=304056
#    Partitions=normal,itee
```

Two gotchas:

- `AllocMem` is what SLURM has handed out. `FreeMem` is what the OS
  reports, which is physical RAM not yet dirtied — if a job allocated it
  but hasn't touched it, OS sees it as free. **Trust AllocMem when
  scheduling.**
- `Partitions=normal,itee` means jobs on either partition compete for
  this node's RAM. If a `normal` job holds 300 GB on cn10, an `itee`
  submit asking for 100 GB will PEND.

To snapshot free RAM across a set of nodes:

```bash
for n in lehmus-cn1 lehmus-cn2 lehmus-cn3 lehmus-cn4 lehmus-cn5 lehmus-cn6; do
  scontrol show node $n | awk -v n=$n '
    /RealMemory/{for(i=1;i<=NF;i++){split($i,v,"=");
                 if(v[1]=="RealMemory")rm=v[2]; if(v[1]=="AllocMem")am=v[2]}}
    END{printf "%s  free=%.1fGB / total=%.1fGB\n", n, (rm-am)/1024, rm/1024}'
done
# lehmus-cn1  free=158.2GB / total=752.0GB
# lehmus-cn2  free=282.1GB / total=498.0GB
# lehmus-cn3  free=242.0GB / total=498.0GB
```

### 2.3 Your own account, QOS, limits

```bash
sacctmgr -nP show assoc user=$USER format=account,partition,qos,maxjobs,maxsubmitjobs
# oy||normal||
```

The pipe-separated columns are the account, partition, QOS, and per-user
limits. **Empty values mean "no limit" at that layer.** Here my account is
`oy`, my default QOS is `normal`, and I have no per-association limits.

Per-QOS limits are separate:

```bash
sacctmgr -nP show qos format=name,maxjobspu,maxsubmitjobspu
# normal||
# interactive|1|
```

Read: the `interactive` QOS limits me to **1 concurrent job**. (This bit
me later; see §8.)

### 2.4 Who else is using what

```bash
# count jobs per user on itee right now
squeue -t R -p itee -h -o "%u" | sort | uniq -c | sort -rn
```

If one user is holding most of the RAM, you'll see them at the top. In
our run `mikkokok` held 2.1 TB on itee, which is why itee's cn10–15 felt
full even when the squeue showed nobody running.

### 2.5 The "where will SLURM put my job?" dry-run

```bash
sbatch --test-only -p itee -c 8 --mem=4800 --time=2:00:00 \
       --constraint=cpu_skylake --wrap=":"
# sbatch: Job 2713244 to start at 2026-11-19T19:59:01 a using 8 processors
#         on nodes lehmus-cn14 in partition itee
```

`--test-only` never actually submits; it only prints when and where SLURM
*estimates* the job would start. Estimates are **pessimistic** — the real
start time is often sooner. Use this to compare partitions:

```bash
sbatch --test-only -p normal -c 8 --mem=4800 --time=2:00:00 \
       --constraint=cpu_milan --wrap=":"
# sbatch: Job 2713012 to start at 2026-12-03T17:59:55  <- 2 weeks out

sbatch --test-only -p itee,normal -c 8 --mem=4800 --time=2:00:00 \
       --constraint="cpu_milan|cpu_skylake" --wrap=":"
# sbatch: Job 2713244 to start at 2026-11-19T19:59:01  <- 3 hours out
```

The multi-partition form (`-p itee,normal`) lets SLURM pick whichever
partition gives you a slot sooner — one of the biggest wins in this sweep.

---

## 3. Writing a one-pair-per-task SNAPHU batch script

Below is the SNAPHU array template, stripped to the bare essentials, with
every `#SBATCH` line explained. The real file is
[`scripts/slurm_jstars_snaphu_itee_20261119.sbatch`](scripts/slurm_jstars_snaphu_itee_20261119.sbatch).

```bash
#!/bin/bash
#SBATCH --account=oy                        # charge usage to the "oy" account
#SBATCH --partition=itee                    # queue to submit into
#SBATCH --constraint=cpu_skylake            # only on Skylake nodes (feature tag)
#SBATCH --cpus-per-task=8                   # 8 CPUs per task (SNAPHU threads)
#SBATCH --mem=4800                          # 4800 MB per task (measured peak)
#SBATCH --time=02:00:00                     # kill after 2 h (helps backfill)
#SBATCH --output=logs/slurm/JSTARS_snaphu_%x_%A_%a.log
#   ^ %x = job name,  %A = main job ID,  %a = array task index

set -euo pipefail
cd /scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement

# Environment the Python needs (conda env + GLIBCXX workaround + repo path)
ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement

# Read inputs from two env vars set at submit time; use pairlist to pick a pair
: "${TREE:?TREE not set}" ; : "${PAIRLIST:?PAIRLIST not set}"
PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIRLIST")   # one pair per line
[[ -n "$PAIR" ]] || { echo "no pair at line ${SLURM_ARRAY_TASK_ID}"; exit 1; }

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} pair=$PAIR"

"${ENV_PREFIX}/bin/python" -u scripts/unwrap_snaphu_jstars_20261119.py \
    --pairs_dir "$TREE" --pair_name "$PAIR" \
    --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif \
    --nproc "${SLURM_CPUS_PER_TASK}"
```

Key design choices to copy next time:

1. **One pair per task** instead of "all pairs in a loop inside one job".
   Reasons: (a) one failure doesn't kill 223 good ones; (b) SLURM can
   schedule 224 small jobs across many nodes, but only 1 big job on 1
   node; (c) resubmitting the few that fail is trivial (see §12).
2. **`--time` is tight, not generous.** SLURM's backfill scheduler can
   only slot you in if it knows you'll be out before the next big job
   starts. A 2-h time limit backfills into almost any gap; 24 h doesn't.
3. **`--mem` is the peak observed RSS plus a small margin.** We measured
   4.5 GB with `sacct ... MaxRSS`; we request 4800 MB. If you request
   16 GB "to be safe", you need a node with 16 GB free per task, which is
   often not available.
4. **Environment is set from scratch inside the script.** SLURM does not
   always pass your shell env into the batch job — `conda`, `LD_LIBRARY_PATH`,
   `PYTHONPATH` all need to be set here.
5. **`PAIR` is picked from a per-submit `$PAIRLIST` file.** This keeps the
   script generic and makes "resubmit only the undone pairs" trivially a
   different pairlist (§12).

### Submit it

```bash
# 1. build the pairlist
PL=logs/JSTARS_snaphu_pairs_$(date -u +%Y%m%d_%H%M).txt
find data/processed/JSTARS_eval_... -mindepth 2 -maxdepth 2 \
     -name ifg_film_unet.tif -printf '%h\n' | xargs -n1 basename | sort > "$PL"
N=$(wc -l < "$PL")

# 2. submit as an array
sbatch --array="1-${N}%48" \
       --export=ALL,TREE=data/processed/JSTARS_eval_...,PAIRLIST="$PL" \
       --job-name="JSTARS_snaphu_variant" \
       scripts/slurm_jstars_snaphu_itee_20261119.sbatch
# Submitted batch job 2713016
```

Three important things:

- `--array=1-N` runs N tasks with `SLURM_ARRAY_TASK_ID` from 1 to N.
- **`%48` is the "how many to run concurrently" throttle.** `--array=1-224%48`
  means "224 tasks total, up to 48 at once". This is the single most useful
  array flag. See §4.
- `--export=ALL,KEY=VAL` passes your current env plus any extra KEY=VAL
  pairs into the job. Cleaner than hard-coding paths into the script.

---

## 4. One-off vs array jobs; the `%N` throttle

- `sbatch script.sbatch` — one job, one task.
- `sbatch --array=1-10 script.sbatch` — ten tasks indexed 1..10, all
  started as soon as there's room (could be all at once).
- `sbatch --array=1-224%48 script.sbatch` — 224 tasks, but at most 48
  running concurrently. The rest PEND with `Reason=JobArrayTaskLimit`.

Pick `%N` based on:

- **How many concurrent slots you *could* get.** If memory limits you to
  ~15 on one partition and you also expect ~5 elsewhere, `%20` is a
  realistic ceiling. Setting `%224` doesn't help — SLURM will just mark
  the extras as `Priority`/`Resources` pending.
- **Politeness to other users.** A `%N` cap below the cluster's capacity
  leaves slack for others, which is good citizenship on a shared machine
  and often buys you faster backfill of your own next batch (SLURM's
  "fair share" remembers who hogged).

To **change the throttle on a live array** without resubmitting:

```bash
scontrol update ArrayTaskThrottle=5 JobId=2713017
```

This drops the concurrency immediately. Already-running tasks keep going;
no new ones start past the limit. Equivalent to resubmitting with the
new `%N` but without losing progress.

---

## 5. Why your jobs aren't starting: reading `Reason=`

`squeue` prints a Reason code for every pending task. Memorise these:

| Reason | What it means | Fix |
|---|---|---|
| `Resources` | SLURM couldn't find a node that satisfies your request *right now*. | Wait, loosen `--mem`/`--cpus-per-task`, try another partition. |
| `Priority` | A higher-priority job is ahead of yours in the queue. | Wait, or `scontrol update Nice=+N JobId=...` on your other jobs that are hogging. |
| `QOSMaxJobsPerUserLimit` | A QOS cap (not your job's size) is blocking you. | §8: resubmit elsewhere. |
| `JobArrayTaskLimit` | **Your own** `%N` throttle is the gate. | Harmless — the array is behaving as submitted. |
| `AssocMaxJobsLimit` | Your account has a hard ceiling. | `sacctmgr show assoc user=$USER`, then resubmit under different params. |
| `None` | Running. | ✓ |
| `BeginTime` | You set `--begin=...`. | Correct. |
| `Dependency` | You set `--dependency=afterok:...` and the parent hasn't finished. | Wait. |

To see Reason for one specific pending task:

```bash
scontrol show job 2713016_9 | grep -E "JobState|Reason|Priority"
#    JobState=PENDING Reason=QOSMaxJobsPerUserLimit Dependency=(null)
#    Priority=14005 Nice=0 Account=oy QOS=normal
```

Full detail for debugging:

```bash
scontrol show job -dd 2713016_9
```

If the Reason changes between `Priority` and `Resources` between refreshes,
the scheduler is actively trying to place your job but another job won
the race; wait another backfill cycle.

---

## 6. CPU is rarely the bottleneck; memory is

This is the lesson from today I'll never forget.

```
sinfo -p itee -o "%N %C"
# NODELIST                         CPUS(A/I/O/T)
# lehmus-cn[10-11,13-15,12]        178/142/0/960
```

142 idle CPUs! You'd think we'd start 142/8 = 17 more of our tasks. But:

```bash
scontrol show node lehmus-cn10 | grep -E "RealMemory|AllocMem"
# RealMemory=380000 AllocMem=379872
```

380 GB total, 379 GB already allocated to other jobs. Even though the CPUs
are "free" in SLURM's accounting, every one of them belongs to a job that
holds RAM. Slot-less. **On a shared cluster, memory beats CPU almost
every time as the real scheduling constraint.**

How to react:

1. **Measure your real peak memory first.** Run one task with your
   generous estimate, let it finish, then:
   ```bash
   sacct -j <JOB_ID> --format=JobID%22,MaxRSS,MaxVMSize,Elapsed,State -X -P \
     | head
   ```
   `MaxRSS` is the actual peak in KB. For SNAPHU on 4 k×4 k pairs we
   measured `MaxRSS=4436220K` = 4.3 GB.
2. **Request peak + ~10% margin, not 3× margin.** We dropped from
   `--mem=16G` to `--mem=4800` and immediately fit more of our jobs into
   the same nodes.
3. **On already-queued jobs, bump the request down without resubmit:**
   ```bash
   scontrol update MinMemoryNode=4800 JobId=2713016
   ```
   Note the quirks: **units are MB, not GB** (`4800`, not `4800M` or
   `4.8G`), and SLURM rejects the update for *currently-running* array
   tasks (`"Job is no longer pending execution"` — that message is per
   already-started task; the pending ones DO get updated).

### What memory does your job really use?

The fastest sanity check is remote `ps`:

```bash
ssh lehmus-cn10 "pgrep -af snaphu | head -3 | xargs -I{} ps -o rss= -p {}"
# 744160    # KB
# 696424
# 690812
```

RSS here is 680–740 MB. `sacct MaxRSS` is higher because it catches
transient peaks `ps` can miss.

---

## 7. Node features and constraints — the sapphirerapids story

SLURM lets admins tag nodes with text "features" like `cpu_skylake`,
`cpu_milan`, `disk_nvme`, `ib_hdr`. You restrict your job to a subset of
nodes with `--constraint=...`:

```bash
#SBATCH --constraint=cpu_skylake              # only this feature
#SBATCH --constraint=cpu_milan|cpu_skylake    # either one
#SBATCH --constraint=cpu_skylake&disk_nvme    # both at once
```

I hit this on day one. My first SLURM array had one task die within one
second on `lehmus-cn2509`, with no log, exit code 0:53. Investigation:

```bash
scontrol show node lehmus-cn2509 | grep -E "AvailableFeatures|State|OS"
#    AvailableFeatures=sapphirerapids
#    OS=Linux 4.18.0-...
```

The node is Sapphire Rapids. Apparently our conda env's `python` or one of
its C deps isn't compatible with the AVX-512 variant on those CPUs, or the
node's `/scratch` mount is different, or any of a dozen things. The fix
without debugging that deeply: **exclude the problem feature**.

```bash
#SBATCH --constraint=cpu_milan|cpu_skylake
```

That picked only the known-good Milan and Skylake nodes. Lesson: when
submitting anything that *must* run across many nodes, pick the smallest
feature set that covers the nodes you've actually tested.

---

## 8. QOS limits that look like a cluster problem

Some partitions carry a QOS that caps your concurrent jobs regardless of
resources. For the `interactive` partition on lehmus:

```bash
sacctmgr -nP show qos name=interactive format=name,maxjobspu,maxsubmitjobspu
# interactive|1|
```

`maxjobspu=1` means **one running job per user on that partition**. Any
array you submit to `interactive` will show ONE task running and the rest
pending with `Reason=QOSMaxJobsPerUserLimit`. The partition can be fully
empty and you still can't run more.

The confusing part: *your job's QOS* might be `normal` (because that's
your default from `sacctmgr show assoc`), yet the *partition* enforces
its own `interactive` QOS layer on top. Both layers' limits apply.

Workaround: submit to a different partition. For us that meant moving
off `interactive` and onto `itee`/`normal` with multi-partition (§9).

---

## 9. Multi-partition submission and backfill — the big win

Our single best scheduling move was changing `--partition=itee` to
`--partition=itee,normal`:

```bash
#SBATCH --partition=itee,normal
```

What this does: SLURM considers *either* partition when scheduling. It
picks whichever gives you a slot first. Together with:

```bash
#SBATCH --time=02:00:00   # short enough to backfill
#SBATCH --mem=4800        # small enough to backfill
```

...SLURM's backfill scheduler can slot our 8-CPU/4.8 GB/2-h tasks into
gaps that big queued jobs are reserving but can't yet use.

Concretely: on `normal`, the Milan nodes (cn1–6) had 150–280 GB free
each, but the queue was dominated by huge pending jobs with weeks-long
waits. Our test-only estimate on `normal` alone was Dec 3 (two weeks
out). With multi-partition it was 3 hours out — and real submits started
*immediately*.

### Rules of thumb for backfill

- **Short wall time** is the biggest enabler. If you can bisect a long
  job into array tasks of 1–3 hours, do it.
- **Small memory** helps symmetrically.
- **Low CPU count per task** (like 8 instead of 64) opens many more
  possible gaps.
- Don't waste `--time` or `--mem`. Backfill is probabilistic — SLURM
  decides your job fits only if it's genuinely small.

### When a partition rejects you, it rejects you cluster-wide for that job

If the multi-partition submit is `itee,normal` and you *also* have a QOS
that caps you on `interactive`, SLURM uses each partition's rules. It
won't go through `interactive` just because you didn't list it. List the
partitions where you want to be considered, exclude the ones with bad
QOS layers.

---

## 10. Updating in-flight jobs with `scontrol update`

You don't have to cancel-and-resubmit to change parameters. For *pending*
array tasks, `scontrol update` does it in place.

```bash
# Change the %N throttle
scontrol update ArrayTaskThrottle=5 JobId=2713017

# Change the per-task memory request  (units: MB, not GB!)
scontrol update MinMemoryNode=4800 JobId=2713017

# Lower this job's priority (Nice goes UP to deprioritize; see §11)
scontrol update Nice=+200 JobId=2713245

# Change TimeLimit (format: D-HH:MM:SS or HH:MM:SS)
scontrol update TimeLimit=03:00:00 JobId=2713017
```

Errors to expect:

- `error: Invalid ... value: 6G` — use plain MB (`6144`), not `6G`.
- `2713016_1-8: Job is no longer pending execution` — already-running
  tasks can't be updated; only pending ones. Partial success on an array.
- `Access/permission denied` — you tried something only admins can do
  (e.g., `Nice=-100`).

### Observing an update actually did something

```bash
# Confirm throttle now says 5
squeue -u $USER -o "%.18i %.42j %.5t" | grep -F '%' | head
# 2713017_[8-224%5]  JSTARS_snaphu_jstars_abl_no_tmp_20261119_0  PD
#                                       ↑ array range + throttle
```

---

## 11. Nice vs Priority — what users actually control

Every job has two priority-ish knobs:

- **Priority** — SLURM's computed number that decides the order of
  consideration in the scheduler. You can't set it directly.
- **Nice** — an integer you add to Priority. **Positive Nice = lower
  effective priority** (be nice to others). Users *can* increase their
  own Nice; **users cannot decrease it** (that would require admin rights
  — SLURM returns `Access/permission denied`).

```bash
scontrol update Nice=+200 JobId=2713245   # ✓ deprioritize MY job
scontrol update Nice=-200 JobId=2713246   # ✗ Access/permission denied
```

So the only user-level knob is to **increase the Nice of your own
leading jobs** so SLURM's backfill prefers your other jobs. That's how I
balanced the sweep: I bumped `no_cls`'s Nice to `+200` so `no_tmp`,
`no_grd`, `zero_film`, `no_unc` would be preferred when memory freed up.

Nice does not affect already-running tasks (they keep running). It only
changes the order new placements are considered.

---

## 12. Resubmitting only undone work (idempotent progress)

When tasks fail/time-out or you cancel-and-resubmit, you want to run
only the pairs that haven't finished. This is trivial because we use a
pairlist file: just rebuild the pairlist from "pairs with input but no
output":

```bash
TREE=data/processed/JSTARS_eval_jstars_abl_no_unc_20261119_0926
PL=logs/JSTARS_snaphu_pairs_jstars_abl_no_unc_undone_$(date -u +%Y%m%d_%H%M).txt

find "$TREE" -mindepth 2 -maxdepth 2 -name ifg_film_unet.tif \
     -type f -printf '%h\n' | sort -u > /tmp/with_ifg.$$
find "$TREE" -mindepth 2 -maxdepth 2 -name unw_phase_film_unet.tif \
     -type f -printf '%h\n' | sort -u > /tmp/with_unw.$$
comm -23 /tmp/with_ifg.$$ /tmp/with_unw.$$ | xargs -n1 basename > "$PL"
rm /tmp/with_ifg.$$ /tmp/with_unw.$$

N=$(wc -l < "$PL")
sbatch --array="1-${N}%12" --export=ALL,TREE="$TREE",PAIRLIST="$PL" \
       --job-name="JSTARS_snaphu_jstars_abl_no_unc_resubmit" \
       scripts/slurm_jstars_snaphu_multi_20261119.sbatch
```

Belt-and-braces: put a `skip if output exists` check inside the sbatch
script so that even if two SLURM tasks race on the same pair (which can
happen if you submit the same pair to two partitions), the second one
no-ops:

```bash
if [[ -f "${TREE}/${PAIR}/unw_phase_film_unet.tif" ]]; then
    echo "already unwrapped: $PAIR"; exit 0
fi
```

Combining `comm -23` for pair-list differences with the skip-guard gives
you a workflow you can resubmit as many times as you like without
duplicating work.

---

## 13. Detached watchers that survive SSH drops

SLURM is the batch system, but you often need a *supervisor* process
that submits the next job when the previous one finishes, or runs
post-processing when a tree reaches 224/224 outputs. Don't park this
inside your interactive shell — if your SSH drops, the supervisor dies
and the pipeline stalls overnight.

Pattern that works on any Linux without systemd:

```bash
WLOG=logs/JSTARS_snaphu_watcher_$(date -u +%Y%m%d_%H%M).log
nohup setsid bash scripts/watch_submit_jstars_snaphu_fast_20261119.sh \
    20261119_0926 jstars_abl_no_cls jstars_abl_no_tmp \
    > "$WLOG" 2>&1 < /dev/null & disown
```

Why each piece:

- `nohup` — ignore SIGHUP (sent when the parent shell exits).
- `setsid` — start in a new session; detaches from the terminal entirely.
- `< /dev/null` — no interactive stdin; prevents `pread(0)` blocking.
- `> file 2>&1` — collect output.
- `&` — background.
- `disown` — remove from the shell's job table so Ctrl-D on the shell
  doesn't try to clean it up.

The combo of `nohup` AND `setsid` is deliberate belt-and-braces; either
alone usually works, both together is bulletproof on lehmus.

Finding your watchers later:

```bash
pgrep -af "watch_submit_jstars_snaphu|watch_stage_c_jstars" | grep -v pgrep
# 2306571 bash scripts/watch_stage_c_jstars_20261119.sh 20261119_0926 ...
```

Killing cleanly:

```bash
kill 2306571   # single PID only — multi-arg kill can match wrong things
```

Example watcher body (abbreviated):

```bash
#!/usr/bin/env bash
set -uo pipefail                        # NOT -e: a failing variant should not abort
cd /path/to/repo

TS="$1"; shift
for TAG in "$@"; do
    LOG="logs/JSTARS_eval_infer_${TAG}_${TS}.log"
    echo "$(date -u +%FT%TZ) waiting for inference of $TAG"
    until [[ -f "$LOG" ]] && grep -q "Inference complete" "$LOG"; do
        sleep 60
    done
    JOB=$(bash scripts/submit_jstars_snaphu_fast_20261119.sh "$TAG" "$TS")
    echo "$(date -u +%FT%TZ) submitted $TAG -> job $JOB"
done
```

---

## 14. Live monitoring: a one-screen dashboard

`squeue` gives one line per task. For 24 running tasks and 300 pending
across 6 variants you need something aggregate. See
[`scripts/status_jstars_eval_20261119.sh`](scripts/status_jstars_eval_20261119.sh).
Run it inside `watch`:

```bash
watch -c -n 30 bash scripts/status_jstars_eval_20261119.sh
```

Patterns worth copying:

### 14.1 Count running tasks per *variant*

```bash
squeue -u $USER -h -t R -o "%.18i %.10P %.42j" \
  | awk '{split($3,a,"_202"); print $2, a[1]}' \
  | sort | uniq -c
#      1 interactiv JSTARS_snaphu_jstars_abl_no_unc
#      4 itee       JSTARS_snaphu_jstars_abl_no_cls
#      5 itee       JSTARS_snaphu_jstars_abl_no_grd
#      5 itee       JSTARS_snaphu_jstars_abl_no_tmp
#      5 itee       JSTARS_snaphu_jstars_abl_zero_film
```

(The `_202` split is a cheap trick: our run tags end with `_YYYYMMDD_HHMM`
and nothing else contains "_202", so splitting there peels off the
timestamp and leaves the variant name.)

### 14.2 Compress an array's state into one short string

```bash
squeue -u $USER -h -o "%j|%t|%r|%P" | awk -F'|' -v tag="JSTARS_snaphu_variant_" '
  index($1, tag)==1 {s[$2]++; part[$4]=1} END {
    out=""; for (k in s) out=out k ":" s[k] " "
    p=""; for (k in part) p=(p?p","k:k)
    print out "|" p
  }'
# R:5 PD:1 |itee,normal
```

The `index($1,tag)==1` matches both the original `JSTARS_snaphu_variant_`
and any sibling like `JSTARS_snaphu_variant_multi_...`, so the dashboard
collapses multi-partition resubmits of the same variant into one row.

### 14.3 File-based progress is more trustworthy than SLURM counts

For our pipeline the *real* progress signal is "how many
`unw_phase_film_unet.tif` exist on disk", which never lies even if SLURM
double-counts or misreports:

```bash
TREE=data/processed/JSTARS_eval_jstars_abl_no_tmp_20261119_0926
n=$(find "$TREE" -mindepth 2 -maxdepth 2 \
       -name unw_phase_film_unet.tif -type f | wc -l)
echo "progress: $n/224"
```

### 14.4 Historical per-task runtime from `sacct`

Once a few tasks have finished, measure actual runtime so you can project
ETA:

```bash
sacct -S 2026-11-19T00:00 -u $USER -X -o JobName%42,State,Elapsed -n \
  | grep -i snaphu | awk '
    {state=$2; split($3,t,":"); sec=t[1]*3600+t[2]*60+t[3]
     split($1, a, "_202"); tag=a[1]
     if (state=="COMPLETED") {c[tag]++; sum[tag]+=sec}}
    END {for (k in c) printf "%-40s %d done  avg=%.1f min\n", k, c[k], sum[k]/c[k]/60}'
# JSTARS_snaphu_jstars_abl_no_cls   23 done  avg=30.4 min
# JSTARS_snaphu_jstars_abl_no_grd   30 done  avg=17.7 min
# JSTARS_snaphu_jstars_abl_no_tmp   30 done  avg=17.3 min
```

The variance across variants here is a real finding, not just scheduling
noise: `no_cls` pairs produce lower-coherence phase, which makes SNAPHU
work harder. That told me I had to balance throttles by giving `no_cls`
*less* concurrency (because each of its tasks is expensive), not more.

---

## 15. Safety guards when outputs mustn't be overwritten

The IGARSS 2026 outputs under `data/processed/pairs/` are the basis of
published Table I — if we overwrote them, we'd lose reproducibility.
Three mechanisms, layered:

1. **New directories named `JSTARS_<id>_<timestamp>/` for everything.**
   Enforced by a build-tree script that symlinks read-only inputs:
   ```python
   LINKS = {"ifg_goldstein_complex_real_imag.tif":
                ("ifg_goldstein_complex_real_imag.tif", "ifg_goldstein.tif"),
            "coherence.tif": ("coherence.tif",),
            ...}
   WRITTEN_BY_EVAL = ("ifg_film_unet.tif", "log_var.tif", "unw_phase_film_unet.tif")
   # we never symlink the WRITTEN_BY_EVAL names, so they land as new files
   ```

2. **Guards inside the forked eval/snaphu scripts that refuse to run
   outside a `JSTARS_*` tree, and refuse to write through a symlink.**
   ```python
   if not Path(args.pairs_dir).resolve().name.startswith("JSTARS_"):
       sys.exit("Refusing to run on non-JSTARS pairs_dir ...")
   for name in ("ifg_film_unet.tif", "log_var.tif"):
       if (pair_dir / name).is_symlink():
           raise RuntimeError(f"Refusing to write through symlink: {pair_dir/name}")
   ```

3. **A snapshot of the IGARSS output files before you start.** If
   anything changed, `diff -q` catches it:
   ```bash
   SNAP=logs/JSTARS_igarss_pairs_snapshot_$(date -u +%Y%m%d_%H%M).txt
   find data/processed/pairs -maxdepth 2 \
        \( -name 'ifg_film_unet.tif' -o -name 'log_var.tif' \
           -o -name 'unw_phase_film_unet.tif' -o -name 'unw_phase.tif' \) \
        -printf '%T@ %s %p\n' | sort -k3 > $SNAP
   # ... run eval ...
   diff -q $SNAP \
     <(find data/processed/pairs -maxdepth 2 \( -name 'ifg_film_unet.tif' ... \) \
       -printf '%T@ %s %p\n' | sort -k3) \
     && echo "IGARSS outputs UNCHANGED"
   ```

Pair these ideas with any shared pipeline whose "default" write path
also holds results you must preserve.

---

## 16. The full flow I followed, start to finish

Documented so you can lift it wholesale into a new project.

```text
step  action                                           outcome
────  ───────────────────────────────────────────────  ──────────────────────────────
 1    sinfo, scontrol show node, sacctmgr show qos     partition/node/QOS map
 2    Build one-pair-per-task sbatch, test on 1 pair   per-task runtime known
 3    sacct --format=MaxRSS on the test task           real memory need known
 4    Submit full array with --mem = peak + margin     avoids over-provisioning
 5    Watch pending Reason= on squeue                  identifies the real block
       └── Resources     → §6 (mem) or widen partition
       └── Priority      → §11 (Nice down competing)
       └── QOSMax...     → §8 (resubmit elsewhere)
 6    Tune %N throttle and Nice to balance progress    equal finish time
 7    Multi-partition when one is slow                 §9 backfill
 8    Watcher + detached shell for auto-next-step      §13 survives SSH drops
 9    Snapshot + guards for critical outputs           §15
10    Dashboard (watch -n 30 status script)            visibility
11    Resubmit undone pairs via comm -23 pairlist      §12 recover from failures
```

The specific sequence I ran today:

```bash
# 1. Learn the cluster
sinfo
sinfo -p itee -N -o "%N %c %m %f %T"
sacctmgr -nP show assoc user=$USER
sacctmgr -nP show qos

# 2. Smoke one pair on the login node
python scripts/unwrap_snaphu_jstars_20261119.py --pair_name <ONE> --nproc 1
# 33 min — too slow, need SLURM

# 3. Smoke one SLURM task
sbatch --array=1 scripts/slurm_jstars_snaphu_20261119.sbatch
# one task died on sapphirerapids node — added --constraint

# 4. Submit anchor variant array (normal partition, 4 CPU)
bash scripts/submit_jstars_snaphu_20261119.sh jstars_full_tmp_active 20261119_0926

# 5. Switch rest to interactive (faster per pair) — then discover QOS cap
bash scripts/submit_jstars_snaphu_fast_20261119.sh jstars_abl_no_unc 20261119_0926

# 6. Move those to itee once QOS cap discovered
bash scripts/submit_jstars_snaphu_itee_20261119.sh jstars_abl_no_cls 20261119_0926

# 7. Discover itee memory block; drop --mem from 16G to 4800 MB
scontrol update MinMemoryNode=4800 JobId=<...>

# 8. Expand to multi-partition for backfill onto normal's Milan nodes
bash scripts/submit_jstars_snaphu_multi_20261119.sh jstars_abl_no_cls 20261119_0926

# 9. Re-balance by throttling the long-pole variant and Nice'ing its priority
scontrol update ArrayTaskThrottle=4 JobId=<no_cls>
scontrol update Nice=+200 JobId=<no_cls>

# 10. As each variant finishes, auto-fire stage C (M2/M4)
nohup setsid bash scripts/watch_stage_c_jstars_20261119.sh ... &

# 11. Monitor with the dashboard
watch -c -n 30 bash scripts/status_jstars_eval_20261119.sh
```

---

## 17. Copy-paste cheat sheet

```bash
# ── Cluster inspection ───────────────────────────────────────────────────
sinfo                                        # partitions + states
sinfo -p PART -N -o "%N %c %m %f %T"         # per-node CPU, mem, features
scontrol show node NODENAME                  # one node in full
scontrol show job JOBID                      # one job in full
scontrol show job -dd JOBID                  # includes Reason, TRES, …
sacctmgr -nP show assoc user=$USER           # my account/qos/limits
sacctmgr -nP show qos                        # cluster-wide QOS limits
squeue -u $USER                              # my queue
squeue -u $USER -t R                         # only running
squeue -u $USER --start                      # pessimistic start estimates
sbatch --test-only -p ... --mem=... -c ... --wrap=":"   # dry run

# ── Submitting ───────────────────────────────────────────────────────────
sbatch -p PART -c N --mem=MB --time=HH:MM:SS \
       --array=1-N%K --export=ALL,KEY=VAL \
       --job-name=NAME script.sbatch

sbatch --parsable ...                        # prints only the JobID

# ── Inflight updates ─────────────────────────────────────────────────────
scontrol update ArrayTaskThrottle=K JobId=J  # change %N
scontrol update MinMemoryNode=MB JobId=J     # change --mem (UNIT: MB)
scontrol update TimeLimit=HH:MM:SS JobId=J   # extend/trim time
scontrol update Nice=+N JobId=J              # deprioritize (users can only raise)

# ── Cancel ───────────────────────────────────────────────────────────────
scancel JOBID                                # whole job or array
scancel JOBID_TASK                           # one array task
scancel -t PD JOBID                          # only pending tasks of an array
scancel -u $USER                             # ALL your jobs, nuclear

# ── Measuring ────────────────────────────────────────────────────────────
sacct -j JOBID --format=JobID%22,State,MaxRSS,MaxVMSize,Elapsed -X -P
sacct -S YYYY-MM-DDTHH:MM -u $USER -o JobName%42,State,Elapsed -n -X

# ── Backgrounding a watcher ──────────────────────────────────────────────
nohup setsid bash watcher.sh ARGS > LOG 2>&1 < /dev/null & disown
pgrep -af watcher.sh                         # find later
kill PID                                     # stop it

# ── Progress-from-disk pattern ───────────────────────────────────────────
find TREE -mindepth 2 -maxdepth 2 -name OUTPUT.tif | wc -l

# ── Resubmit undone pairs (idempotent) ───────────────────────────────────
comm -23 <(find TREE -name INPUT.tif  -printf '%h\n' | sort -u) \
         <(find TREE -name OUTPUT.tif -printf '%h\n' | sort -u) \
  | xargs -n1 basename > PAIRLIST
sbatch --array="1-$(wc -l < PAIRLIST)%K" --export=ALL,PAIRLIST=PAIRLIST,... script
```

---

## Mistakes I made (and how to avoid them)

- Submitted with `--mem=16G` → OOM-slot-less. Measure peak first.
- Set `--cpus-per-task` too high → fewer concurrent slots. Scale with
  peak-thread-utilisation, not with "the node has that many".
- Used `--partition=interactive` without knowing about `maxjobspu=1`.
  Query `sacctmgr show qos` before committing to a partition.
- `set -euo pipefail` in a watcher script → one failing variant killed
  the whole chain. Keep `-u -o pipefail` but drop `-e` in orchestrators.
- Named a loop variable `pd` in Python → shadowed `import pandas as pd`
  fifty lines later. Pick loop variable names that don't match imports.
- Killed watchers by matching too broadly with `pgrep` → killed the
  terminal shell too. Use `pgrep -af` to inspect first, then kill by
  exact PID, never by pattern.
- Guessed starting times from `--test-only`. Always submit *one* real
  task to measure; the estimate is far more pessimistic than reality.
- Over-constrained `--constraint` → only certain nodes qualified. Use
  the **widest safe feature expression**, like `cpu_milan|cpu_skylake`,
  not just `cpu_skylake`.

That's it. If you internalise just §6 (memory, not CPU), §8 (QOS layers),
§9 (multi-partition backfill), §11 (Nice goes up for you), and §13
(`nohup setsid`), you'll beat 90% of SLURM users on a shared cluster.
