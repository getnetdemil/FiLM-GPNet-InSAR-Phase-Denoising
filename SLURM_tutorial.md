# SLURM in practice on the lehmus cluster

This guide has two parts.

- **[Part I — General SLURM workflow on lehmus](#part-i--general-slurm-workflow-on-lehmus)** is the
  minimum you need to know to run *any* batch job here — inspect the
  cluster, write an sbatch, submit, monitor, debug, clean up. If you've
  never used SLURM before, read Part I top to bottom.
- **[Part II — Case study: parallelising 1 120 SNAPHU unwraps for Table II](#part-ii--case-study-parallelising-1-120-snaphu-unwraps-for-table-ii)**
  is a concrete walk-through of applying Part I to the JSTARS ablation
  evaluation, including the mistakes I actually made. Read it when you
  hit a problem that Part I's general principles don't solve on their own.

The practical cheat sheet at the end works as a standalone reference.

---

## Part I — General SLURM workflow on lehmus

### 1. What SLURM actually does

SLURM (Simple Linux Utility for Resource Management) is a batch
scheduler: you describe the compute you want (how many CPUs, how much
RAM, for how long, on what kind of node), SLURM finds a matching slot
somewhere in the cluster, and runs your script there.

Three concepts you need in muscle memory:

- **Node** — one physical computer. On lehmus: `lehmus-cn1`, …,
  `lehmus-cn15`, `lehmus-cn2501..2509`, `lehmus-testcn1..2`. Each has a
  fixed number of CPU cores (64, 96, or 160) and a fixed amount of RAM
  (380–1000 GB).
- **Partition** — a named pool of nodes with its own time limits and
  often its own QOS. On lehmus: `normal`, `itee`, `interactive`,
  `nmr{,-small,-large}`, `debug`. A node can belong to several
  partitions (e.g. `cn10–15` are in both `normal` and `itee`).
- **Job** — your unit of work. You submit a *script* (`sbatch`) with
  resource requests; SLURM assigns a job ID and runs it when a slot is
  found. An **array job** is one submission that fans out into N
  numbered tasks differentiated by `$SLURM_ARRAY_TASK_ID`.

Three non-obvious truths about how SLURM schedules:

1. **Backfill scheduling** — small, short jobs can "sneak in" to gaps
   that larger queued jobs can't yet use. This is the single biggest
   lever for running many small jobs on a busy cluster. See §5.9.
2. **Partitions overlap on nodes** — a node in `normal` AND `itee`
   means jobs from both partitions compete for the same CPUs and RAM
   on that node. Free CPU on one partition doesn't mean free RAM.
3. **QOS is a layer separate from partition** — a partition can carry a
   QOS that caps per-user concurrent jobs, and that cap bites even when
   nodes have plenty of room. See §6.3.

### 2. Lehmus-specific environment setup

Before any lehmus SLURM script runs, three environment variables need to
be set (SLURM does not pass your login shell's env into batch jobs by
default):

```bash
# Conda env with torch, rasterio, snaphu, pandas, pyarrow, …
export ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu

# The repo root
export REPO=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement

# rasterio on lehmus needs the conda env's libstdc++ (GLIBCXX workaround)
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# so Python finds src/
export PYTHONPATH="${REPO}"
```

Inside any sbatch script, set these BEFORE calling your Python. The
cleanest pattern is to declare them explicitly in every sbatch; don't
rely on `#SBATCH --export=ALL` propagating your login env.

### 3. Inspect before you submit

Learn the lay of the land before asking SLURM for anything.

#### 3.1 Which partitions exist and how busy they are

```bash
sinfo
# PARTITION   AVAIL  TIMELIMIT   NODES(A/I/O/T) NODELIST
# interactive    up 7-00:00:00          4/1/0/5 lehmus-cn[2-3,7-9]
# normal*        up 14-00:00:0        15/0/0/15 lehmus-cn[1-6,10-15,2507-2509]
# itee           up 14-00:00:0          6/0/0/6 lehmus-cn[10-15]
```

`A/I/O/T` is Allocated / Idle / Other / Total. `*` on `normal` means
it's the default partition if you don't specify one.

Per-node detail with CPU count, RAM, and *features*:

```bash
sinfo -p itee -N -o "%N %c %m %f %T"
# NODELIST     CPUS  MEMORY  AVAIL_FEATURES  STATE
# lehmus-cn10  160   380000  cpu_skylake     mixed-
# lehmus-cn14  160   380000  cpu_skylake     mixed-
```

Memory is in MB (380000 = 380 GB). State suffixes to recognise:

- `mixed` — partly used, partly free. Good.
- `allocated` — fully in use. Bad.
- `idle` — completely free. Great.
- `+PLANNED`, `+NOT_RESPONDING`, `+MAINT` — reserved for an upcoming
  job, down, or in maintenance. Treat the last two as unusable.

Available features on lehmus: `cpu_milan`, `cpu_skylake`, `sapphirerapids`,
`disk_nvme`, `ib_hdr`. You use these with `--constraint=…` to restrict
which nodes your job can land on (§5.7).

#### 3.2 What a node actually has free *right now*

`sinfo`'s "mixed" lies. A node can be mixed on CPU and still have 99% of
its RAM reserved. Check for real:

```bash
scontrol show node lehmus-cn10 | \
  grep -E "CPUAlloc|RealMemory|AllocMem|State|Partitions"
#    State=MIXED+PLANNED
#    CPUAlloc=98 CPUEfctv=160
#    RealMemory=380000 AllocMem=379872 FreeMem=304056
#    Partitions=normal,itee
```

Trust `AllocMem` for scheduling — `FreeMem` is the OS view of physical
pages not yet dirtied by running jobs, which SLURM doesn't care about.

Scan several nodes at once:

```bash
for n in lehmus-cn1 lehmus-cn2 lehmus-cn3 lehmus-cn4 lehmus-cn5 lehmus-cn6; do
  scontrol show node $n | awk -v n=$n '
    /RealMemory/{for(i=1;i<=NF;i++){split($i,v,"=");
                 if(v[1]=="RealMemory")rm=v[2]; if(v[1]=="AllocMem")am=v[2]}}
    END{printf "%s  free=%.1fGB / total=%.1fGB\n", n, (rm-am)/1024, rm/1024}'
done
# lehmus-cn1  free=158.2GB / total=752.0GB
# lehmus-cn2  free=282.1GB / total=498.0GB
```

#### 3.3 Your own account, QOS, and limits

```bash
sacctmgr -nP show assoc user=$USER format=account,partition,qos,maxjobs,maxsubmitjobs
# oy||normal||
```

Pipe-separated columns are account, partition, QOS, and per-user limits.
**Empty values mean "no limit" at that layer.** Here my account is `oy`,
default QOS `normal`, no association-level limits.

Per-QOS limits (independent of your association):

```bash
sacctmgr -nP show qos format=name,maxjobspu,maxsubmitjobspu
# normal||
# interactive|1|
```

`interactive` QOS caps you at **1 concurrent job** regardless of your
account. This matters when you later see a partition sitting idle but
your array is still PENDING. See §6.3.

#### 3.4 Who else is using the cluster

```bash
# count running tasks per user on a partition
squeue -t R -p itee -h -o "%u" | sort | uniq -c | sort -rn
```

If one other user is holding most of a partition's RAM, you'll see them
at the top. Useful when debugging why "nothing" is running but the
partition's actually full.

#### 3.5 Will SLURM accept my request? When would it start?

```bash
sbatch --test-only -p itee -c 8 --mem=4800 --time=2:00:00 \
       --constraint=cpu_skylake --wrap=":"
# sbatch: Job 2713244 to start at 2026-11-19T19:59:01
#         using 8 processors on nodes lehmus-cn14 in partition itee
```

`--test-only` never submits; it prints the predicted start time and
node. **Estimates are pessimistic** — a real submit often starts sooner.
Use `--test-only` to A/B two partitions:

```bash
sbatch --test-only -p normal       -c 8 --mem=4800 --time=2:00:00 --wrap=":"
sbatch --test-only -p itee,normal  -c 8 --mem=4800 --time=2:00:00 --wrap=":"
```

### 4. The anatomy of an sbatch script

Below is the pattern every lehmus sbatch should follow. The example is
embarrassingly parallel — one task per "pair" from a text file — because
that's the most common case for this project.

```bash
#!/bin/bash
# ── SLURM directives (must come before any executable code) ────────────
#SBATCH --account=oy                  # which SLURM account to charge
#SBATCH --partition=itee              # queue (or comma-separated list)
#SBATCH --constraint=cpu_skylake      # node feature expression (OR with |)
#SBATCH --cpus-per-task=8             # CPU cores for this task
#SBATCH --mem=4800                    # MB of RAM per task (NOT a unit!)
#SBATCH --time=02:00:00               # kill after 2 h (short time → backfills better)
#SBATCH --output=logs/slurm/%x_%A_%a.log
#                       %x = job name, %A = main job ID, %a = array index

# ── Fail-fast + env ────────────────────────────────────────────────────
set -euo pipefail

REPO=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}"
cd "$REPO"

# ── Read inputs via env vars set at submit time ────────────────────────
: "${PAIRLIST:?PAIRLIST not set}"
PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIRLIST")
[[ -n "$PAIR" ]] || { echo "no pair at line ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# ── Idempotency: don't redo finished work ──────────────────────────────
if [[ -f "outputs/${PAIR}.done" ]]; then
    echo "already done: $PAIR"; exit 0
fi

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} pair=$PAIR start=$(date -u +%FT%TZ)"

"${ENV_PREFIX}/bin/python" -u your_script.py \
    --pair_name "$PAIR" \
    --nproc "${SLURM_CPUS_PER_TASK}"

echo "end=$(date -u +%FT%TZ)"
```

Design rules that keep you out of trouble:

1. **One unit of work per task** instead of looping inside the script.
   Reasons: a) one failure doesn't kill 223 good ones; b) SLURM can
   schedule 224 small tasks across many nodes, not one big job on one
   node; c) resubmitting only the failures is trivial if your pairlist
   is just "pairs to do".
2. **Keep `--time` tight, not generous.** Backfill scheduling (§5.9)
   can only slot you in if SLURM is sure you'll be out before the next
   big job starts. 2 h backfills into almost any gap; 24 h doesn't.
3. **`--mem` is peak RSS + ~10%, not "to be safe".** If you request
   16 GB per task you need a node with 16 GB free per task, which is
   often not available even when CPUs are. See §6.1.
4. **Set env inside the script.** `--export=ALL` works for simple
   cases but conda, GLIBCXX, and PYTHONPATH are safer explicit.
5. **Guard idempotency inside the task.** If the output already exists,
   `exit 0`. Lets you resubmit the whole array without duplicating
   finished work.

### 5. Submitting

#### 5.1 A single job

```bash
sbatch script.sbatch
# Submitted batch job 2713244
```

#### 5.2 An array job

```bash
# Build a one-line-per-item file first (the "pairlist")
printf 'pair_a\npair_b\npair_c\n' > logs/pairlist.txt

sbatch --array="1-3" \
       --export=ALL,PAIRLIST=logs/pairlist.txt \
       --job-name="my_job" \
       script.sbatch
```

Each task gets a unique `$SLURM_ARRAY_TASK_ID` from 1 to N. Your script
reads line N from the pairlist.

#### 5.3 Capping concurrency with `%N`

```bash
sbatch --array="1-224%48" ...
```

`%48` means "run at most 48 tasks at once"; the rest pend with
`Reason=JobArrayTaskLimit`. Choose `%N` based on:

- How many slots you can plausibly get (don't ask for 200 if the
  partition only has room for 40).
- Politeness to other users on a shared cluster.

You can change `%N` on a live array without resubmitting:

```bash
scontrol update ArrayTaskThrottle=5 JobId=2713017
```

#### 5.4 Multi-partition submission

```bash
#SBATCH --partition=itee,normal
```

SLURM picks whichever partition gives you a slot first. Pair this with
short `--time` and tight `--mem` to let backfill scheduling (§5.9)
place you onto partitions that are "booked solid" by big queued jobs.

#### 5.5 Capturing the job ID

```bash
JOB=$(sbatch --parsable ...)
echo "submitted $JOB"
```

`--parsable` prints only the number, suitable for scripting.

#### 5.6 `--export=ALL,KEY=VAL` for configuration

Avoid hard-coding paths into the sbatch script. Pass them from the
submit helper:

```bash
sbatch --export=ALL,PAIRLIST=logs/pl.txt,OUT=out/run1 ... script.sbatch
```

Inside the script:

```bash
: "${PAIRLIST:?PAIRLIST not set}"
: "${OUT:?OUT not set}"
```

The `${VAR:?msg}` syntax exits immediately if `VAR` is unset — useful for
catching submission mistakes early.

#### 5.7 Node constraints (features)

```bash
#SBATCH --constraint=cpu_milan|cpu_skylake   # either of these
#SBATCH --constraint=cpu_skylake&disk_nvme   # both at once
```

Use the **widest safe feature expression**. If your code works on Milan
and Skylake but not on Sapphire Rapids (because of a specific CPU
instruction or network/OS mismatch — this is real; see §5.7 story in
Part II), use the OR form, not a single feature.

#### 5.8 GPU allocation

If your job needs GPUs:

```bash
#SBATCH --gres=gpu:1                 # any 1 GPU on this node
#SBATCH --gres=gpu:v100:1            # specifically a V100
#SBATCH --gres=gpu:a30:1             # specifically an A30
```

Inside the script, `CUDA_VISIBLE_DEVICES` is set automatically to the
GPUs SLURM allocated — don't override it.

#### 5.9 Backfill — the small/short trick

The scheduler's backfill pass fits small, short jobs into gaps that
large pending jobs reserve but can't yet use. To benefit from it:

- Keep `--time` short (hours, not days).
- Keep `--mem` tight.
- Keep `--cpus-per-task` modest.
- List multiple partitions (`-p itee,normal`) so backfill can place you
  wherever a gap opens.

### 6. Why isn't my job running? Reading `Reason=`

`squeue` prints a Reason for every pending task. Memorise these:

| Reason | What it means | Fix |
|---|---|---|
| `Resources` | No node satisfies your request right now | Wait, loosen `--mem`/`--cpus-per-task`, add partitions |
| `Priority` | A higher-priority job is ahead of yours | Wait, or `scontrol update Nice=+N` on your own leading jobs |
| `QOSMaxJobsPerUserLimit` | A QOS cap, not your request, is blocking | Resubmit to a different partition (§6.3) |
| `JobArrayTaskLimit` | Your own `%N` throttle is the gate | Harmless — behaving as submitted |
| `AssocMaxJobsLimit` | Account-level hard cap | `sacctmgr show assoc`, change params |
| `None` | Running | ✓ |
| `BeginTime`, `Dependency` | You set `--begin` or `--dependency=afterok:…` | Wait for the condition |

Inspect one task:

```bash
scontrol show job 2713016_9 | grep -E "JobState|Reason|Priority|QOS|Partition"
#    JobState=PENDING Reason=QOSMaxJobsPerUserLimit Dependency=(null)
#    Priority=14005 Nice=0 Account=oy QOS=normal
#    Partition=itee AllocNode:Sid=lehmus-cn5.oulu.fi:...
```

Full dump including features and TRES:

```bash
scontrol show job -dd 2713016_9
```

#### 6.1 "Resources" — memory is usually the real bottleneck

On a shared cluster, **memory beats CPU almost every time** as the
actual scheduling constraint.

If `sinfo` shows free CPUs:

```bash
sinfo -p itee -o "%N %C"
# NODELIST                         CPUS(A/I/O/T)
# lehmus-cn[10-11,13-15,12]        178/142/0/960
```

...check whether those "free" CPUs actually have RAM behind them:

```bash
scontrol show node lehmus-cn10 | grep -E "RealMemory|AllocMem"
# RealMemory=380000 AllocMem=379872
```

380 GB total, 379 GB allocated. The 142 "idle" CPUs are useless because
every one of them belongs to a job that holds the RAM.

Measure your real peak memory from a completed task:

```bash
sacct -j <JOB_ID> -X -P -o JobID%22,MaxRSS,MaxVMSize,Elapsed,State | head
# 2712802_1.batch|4436220K|5147084K|00:11:57|COMPLETED
```

Request peak + ~10%, not 3×.

Shrink `--mem` on *pending* array tasks without resubmitting:

```bash
scontrol update MinMemoryNode=4800 JobId=2713016
#                              ^^^^ UNITS ARE MB, not "4800M" or "4.8G"
```

Currently-running tasks can't be shrunk; only the pending ones. That's
usually fine.

#### 6.2 "Priority" — the queue has prior jobs

Backfill can still place you ahead of larger priority jobs if you fit
into their gaps. If it doesn't, your options are:

- Wait.
- Submit to another partition.
- Increase the Nice of your own leading jobs so your lagging ones pick
  up next (§7.2).

#### 6.3 "QOSMaxJobsPerUserLimit" — the hidden cap

Some partitions carry a QOS that caps your concurrent jobs regardless
of resources. For the `interactive` QOS:

```bash
sacctmgr -nP show qos name=interactive format=name,maxjobspu
# interactive|1
```

`maxjobspu=1` means one running job per user on that partition. The
partition can be fully empty and you still can't run more. The only
workaround is to resubmit to a different partition — there's no
user-side way to bypass QOS.

### 7. Changing live jobs with `scontrol update`

You rarely need to cancel-and-resubmit. For pending array tasks:

```bash
scontrol update ArrayTaskThrottle=5  JobId=2713017   # new %N limit
scontrol update MinMemoryNode=4800   JobId=2713017   # new --mem (MB)
scontrol update TimeLimit=03:00:00   JobId=2713017   # new --time
scontrol update Nice=+200            JobId=2713017   # deprioritize (see §7.2)
```

Typical errors:

- `Invalid MinMemoryNode value: 6G` — use `6144` (MB), not `6G`.
- `2713016_1-8: Job is no longer pending execution` — already-running
  tasks can't be updated; pending ones still get the change.
- `Access/permission denied` — you asked for something only admins can
  do (e.g., `Nice=-100`).

Verify an update took:

```bash
squeue -u $USER -o "%.18i %.42j %.5t" | grep '%'
# 2713017_[8-224%5] JSTARS_snaphu_variant_X  PD
#                           ↑ throttle is now 5
```

#### 7.1 Cancelling

```bash
scancel JOBID            # whole job or whole array
scancel JOBID_TASK       # one array task (e.g., 2713017_3)
scancel -t PD JOBID      # only pending tasks; leave running ones alone
scancel -u $USER         # NUCLEAR — every job of yours
```

#### 7.2 Nice vs Priority — what users can actually control

Every job has:

- **Priority** — SLURM-computed scheduling order. You can't set it.
- **Nice** — an integer you add to Priority. **Positive Nice = lower
  effective priority**. Users can only *raise* their own Nice; lowering
  it requires admin rights.

```bash
scontrol update Nice=+200 JobId=2713245   # ✓ deprioritize MY leading job
scontrol update Nice=-200 JobId=2713246   # ✗ Access/permission denied
```

So the only user-side knob for balance is to **increase the Nice of
your own hog jobs** so the backfill scheduler prefers your other jobs.
Nice doesn't affect already-running tasks.

### 8. Monitoring

#### 8.1 Live state

```bash
squeue -u $USER                                   # everything
squeue -u $USER -t R                              # only running
squeue -u $USER -o "%.18i %.10P %.2t %.10M %.42j %R"   # custom columns
squeue -u $USER --start                           # pessimistic start estimates
```

#### 8.2 Count running tasks per variant or job name

```bash
squeue -u $USER -h -t R -o "%.18i %.10P %.42j" \
  | awk '{split($3,a,"_202"); print $2, a[1]}' \
  | sort | uniq -c
```

The `_202` split is a cheap trick: run tags end with `_YYYYMMDD_HHMM`
and nothing else contains "_202" (good until 2030), so splitting there
peels the timestamp off the variant name.

#### 8.3 Progress-from-disk is more trustworthy than SLURM counts

For pipelines where each task writes one output file, count files
directly. It never lies:

```bash
find outputs/ -mindepth 2 -maxdepth 2 -name 'result.tif' | wc -l
```

#### 8.4 Historical per-task runtime from `sacct`

Once some tasks have finished, project ETA from real numbers:

```bash
sacct -S "2026-11-19T00:00" -u $USER -X -o JobName%42,State,Elapsed -n \
  | awk '{split($3,t,":"); sec=t[1]*3600+t[2]*60+t[3]
          if ($2=="COMPLETED"){c[$1]++; sum[$1]+=sec}}
         END {for (k in c) printf "%-40s  done=%d  avg=%.1f min\n",
                                   k, c[k], sum[k]/c[k]/60}'
```

### 9. Detached supervisors that survive SSH drops

SLURM is the batch system, but you often want a supervisor process that
submits the next array when the previous one finishes, or runs
post-processing when a tree reaches 224/224 outputs. Don't park this
inside your interactive shell — if your SSH drops, the supervisor dies
and the pipeline stalls overnight.

The portable incantation:

```bash
WLOG=logs/watcher_$(date -u +%Y%m%d_%H%M).log
nohup setsid bash scripts/your_watcher.sh ARGS \
    > "$WLOG" 2>&1 < /dev/null & disown
```

Why each piece:

- `nohup` — ignore SIGHUP (sent when the parent shell exits).
- `setsid` — new session; detaches from the terminal entirely.
- `< /dev/null` — no stdin; prevents blocking on reads.
- `> file 2>&1` — collect output.
- `&` — background.
- `disown` — remove from the shell's job table.

`nohup` AND `setsid` together is belt-and-braces; either alone usually
works, both together is bulletproof.

Find and kill later:

```bash
pgrep -af your_watcher.sh | grep -v pgrep
# 2306571 bash scripts/your_watcher.sh ARGS
kill 2306571
```

### 10. Resubmitting only undone work

The idempotent pattern. Build a pairlist of inputs with no output:

```bash
comm -23 \
  <(find TREE -name INPUT.tif  -printf '%h\n' | sort -u) \
  <(find TREE -name OUTPUT.tif -printf '%h\n' | sort -u) \
  | xargs -n1 basename > logs/undone.txt

N=$(wc -l < logs/undone.txt)
sbatch --array="1-${N}%12" \
       --export=ALL,PAIRLIST=logs/undone.txt \
       script.sbatch
```

Belt-and-braces: put a `skip if output exists` check inside the sbatch
(§4) so that if two array runs ever race on the same pair, the second
one no-ops.

### 11. Resource-use hygiene

- Log output goes to `logs/slurm/` — create that dir once; SLURM won't.
- Use timestamped filenames (`run_YYYYMMDD_HHMM.log`) so repeated runs
  don't overwrite each other.
- `--time` ≈ actual runtime + a modest safety margin; hugely-oversized
  time limits keep your jobs out of backfill.
- `--mem` ≈ measured peak RSS + 10–15%.
- Don't park long-running interactive shells if you can help it — they
  hold a slot (CPU + GPU + RAM) for the full `--time` even when idle.
- Clean up finished array outputs periodically (`sacct -S`, old logs)
  so the control plane stays snappy.

### 12. Top-of-head troubleshooting checklist

1. Did my sbatch actually submit? → `squeue -u $USER | grep JOBID`
2. Why is it PENDING? → `scontrol show job JOBID | grep Reason`
3. Does any node satisfy my request? → `sinfo -p PART -N -o "%N %c %m %f %T"`
4. Is my QOS capping me? → `sacctmgr show qos`
5. Am I over-requesting RAM? → `sacct -j OLD_JOB -X -o MaxRSS -n`
6. Did the task start but die fast? → `cat logs/slurm/*_${JOBID}_*.log`
7. Did my supervisor die? → `pgrep -af watcher_name`
8. Did an OOM-killer hit? → `dmesg | tail` on the node (via `ssh` if allowed)

---

## Part II — Case study: parallelising 1 120 SNAPHU unwraps for Table II

This is how I applied Part I to the JSTARS ablation evaluation on
2026-11-19. 6 variants × 224 Capella Hawaii pairs = 1 344 SNAPHU unwraps.
The anchor was already done; the remaining 1 120 are what Part II is about.

Every command here is real, every mistake was mine, every workaround
is what I'd do again next time.

### P1. The problem

For each of 6 ablation variants we needed to unwrap `ifg_film_unet.tif`
→ `unw_phase_film_unet.tif` with SNAPHU on all 224 Hawaii pairs. SNAPHU
is single-threaded per tile and CPU-bound. On the login node (`cn5`)
one pair took 33 minutes with `--nproc=1`. Straight line: 33 min × 224
pairs × 6 variants ≈ 15 days. Not acceptable.

Two hard constraints:

1. **IGARSS 2026 outputs in `data/processed/pairs/` must not be
   touched.** Published Table I depends on them.
2. **I'm one of many users on lehmus.** Can't just grab all nodes.

### P2. Inspecting what we actually have

Walking through §3 of Part I on the real cluster:

```bash
sinfo
# normal*        up 14-00:00:0        15/0/0/15 lehmus-cn[1-6,10-15,2507-2509]
# itee           up 14-00:00:0          6/0/0/6 lehmus-cn[10-15]
# interactive    up 7-00:00:00          4/1/0/5 lehmus-cn[2-3,7-9]
```

`itee` is 6 skylake nodes (cn10–15). `normal` overlaps with those *plus*
cn1–6 (Milan). `interactive` is a different set.

```bash
sacctmgr -nP show assoc user=$USER
# oy||normal||
```

My default QOS is `normal`, no account limits.

```bash
sacctmgr -nP show qos
# normal||
# interactive|1|
```

The `interactive` QOS has `maxjobspu=1`. Important later.

### P3. Measure one task before you scale

Smoke test, 1 pair, on cn5 login node:

```bash
export ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
"$ENV_PREFIX/bin/python" -u scripts/unwrap_snaphu_jstars_20261119.py \
    --pair_name <ONE_PAIR> --nproc 1
# Elapsed wall clock time:  0:33:08
```

33 minutes. Try `--nproc=4` (test SLURM array, 1 task, 4 CPUs):

```bash
sbatch --array=1 -p normal -c 4 --mem=16G --time=1:00:00 \
       --export=ALL,TREE=...,PAIRLIST=... \
       scripts/slurm_jstars_snaphu_20261119.sbatch
```

Inside the task it logs:

```
Elapsed wall clock time:  0:10:56
```

Also measure the RSS:

```bash
sacct -j 2712744_2 -X -o MaxRSS -P
# 4436220K  =  4.3 GB peak
```

Two findings:

- **4.3 GB peak RSS**, not 16. We're over-provisioning memory 3.7×.
- **`--nproc=4` gives ~3× speedup over 1**. Bumping further helps a bit
  less (SNAPHU parallelises at the tile level, not inside tiles).

### P4. First mistake: sapphirerapids

First real submit:

```bash
sbatch --array=1-2 -p normal -c 4 --mem=16G --time=2:00:00 ...
# ... 2712744_[1-2]
```

Task 1 died in 1 second on `lehmus-cn2509`, no log:

```bash
sacct -j 2712744_1 -o JobID,State,ExitCode,NodeList -n
# 2712744_1  FAILED  0:53  lehmus-cn2509
scontrol show node lehmus-cn2509 | grep Features
# AvailableFeatures=sapphirerapids
```

The newer Sapphire Rapids nodes (cn2507–2509) don't play nicely with our
conda env (AVX-512 variant, possibly a library mismatch — didn't dig in).
Workaround: constrain away:

```bash
#SBATCH --constraint=cpu_milan|cpu_skylake
```

That's rule §5.7 of Part I in practice: use the widest safe feature
expression.

### P5. The forks (preserving IGARSS outputs)

Constraint 1: IGARSS outputs must not be touched. Solution: fork the
eval and unwrap scripts and layer guards:

- `eval/compute_metrics_jstars_20261119.py` — adds `--zero_film`,
  **refuses** `--pairs_dir` unless its name starts with `JSTARS_`, and
  refuses to write through a symlink.
- `scripts/unwrap_snaphu_jstars_20261119.py` — adds `--nproc`,
  `--pair_name`, same JSTARS and symlink guards.

Each variant gets its own tree:
`data/processed/JSTARS_eval_<variant>_<EVAL_TS>/<pair>/`. Inputs are
read-only symlinks back to `data/processed/pairs/`. Written files
(`ifg_film_unet.tif`, `log_var.tif`, `unw_phase_film_unet.tif`) land as
new files in the JSTARS_* tree.

Pre-run snapshot:

```bash
find data/processed/pairs -maxdepth 2 \
     \( -name 'ifg_film_unet.tif' -o -name 'log_var.tif' \
        -o -name 'unw_phase_film_unet.tif' -o -name 'unw_phase.tif' \) \
     -printf '%T@ %s %p\n' | sort -k3 > logs/JSTARS_igarss_pairs_snapshot_$(date -u +%Y%m%d_%H%M).txt
```

If anything got touched later, `diff -q` catches it.

### P6. Submit helpers + one sbatch per path

I ended up with four sbatch templates, one per SLURM path I explored:

- `slurm_jstars_snaphu_20261119.sbatch` — `normal`, 4 CPU/task.
- `slurm_jstars_snaphu_fast_20261119.sbatch` — `interactive`, 8 CPU/task.
- `slurm_jstars_snaphu_itee_20261119.sbatch` — `itee`, 8 CPU/task.
- `slurm_jstars_snaphu_multi_20261119.sbatch` — `itee,normal`, 8 CPU/task.

Each pairs with its submit helper (`submit_jstars_snaphu_*.sh`) that
builds the pairlist and calls `sbatch`.

### P7. Switching partitions as bottlenecks appeared

#### Move 1 — `normal` is clogged

First real submit of the anchor variant went to `normal`. Projected
start: Dec 3 (two weeks out).

#### Move 2 — try `interactive`, hit the QOS cap

```bash
bash scripts/submit_jstars_snaphu_fast_20261119.sh jstars_abl_no_unc TS
# array 2712802 submitted
squeue -u $USER
# 2712802_1   R
# 2712802_[2-224%24] PD  QOSMaxJobsPerUserLimit
```

Only 1 running. Checked `sacctmgr show qos`: `interactive|1|`. QOS cap.
Can't bypass.

#### Move 3 — switch the remaining variants to `itee`

Cancel pending `interactive` arrays (not the one running task), build
undone pairlists, resubmit to `itee`:

```bash
for j in 2712803 2712902 2712992 2713001; do scancel $j; done
for tag in jstars_abl_no_cls jstars_abl_no_tmp jstars_abl_no_grd jstars_abl_zero_film; do
  bash scripts/submit_jstars_snaphu_itee_20261119.sh "$tag" "$EVAL_TS"
done
```

8 tasks started instantly on cn10/cn14 (test estimate had said 22 h —
very pessimistic).

#### Move 4 — hit itee's memory block

We only scaled to 8 concurrent on itee. Checked:

```bash
scontrol show node lehmus-cn10 | grep -E "CPUAlloc|AllocMem"
# CPUAlloc=98 CPUEfctv=160
# AllocMem=379872 of 380000
```

142 CPUs idle, 5 GB RAM free. **Memory, not CPU, was the block.**
Someone else was holding 2.1 TB on cn10–15:

```bash
squeue -t R -p itee -h -o "%u %m" | sort | uniq -c | sort -rn | head
```

Fix: shrink `--mem` from 16 GB to 4800 MB (measured peak was 4.3 GB).

On already-pending arrays, in-place:

```bash
for j in 2713016 2713017 2713018 2713019; do
  scontrol update MinMemoryNode=4800 JobId=$j
done
```

For all future submits, patched the sbatch files.

Still only ~14 concurrent after that. The RAM on cn10–15 was shared
with `normal`-partition users and they weren't leaving.

#### Move 5 — the big win, multi-partition into Milan nodes

Checked what's free elsewhere. Milan nodes cn1–6 (in `normal` partition)
had 150–280 GB free each. 1.3 TB total free. Why? Because the big
pending `normal` jobs wait for enough contiguous resources, but our
small 8-CPU/4.8-GB/2-h tasks can **backfill** into the gaps.

New sbatch template with multi-partition:

```bash
#SBATCH --partition=itee,normal
#SBATCH --constraint=cpu_milan|cpu_skylake
#SBATCH --cpus-per-task=8
#SBATCH --mem=4800
#SBATCH --time=02:00:00
```

Cancel pending itee tasks only (keep running ones), build undone
pairlists, resubmit:

```bash
for j in 2713016 2713017 2713018 2713019 2713060; do
  scancel -t PD $j
done
for tag in jstars_abl_no_cls jstars_abl_no_tmp jstars_abl_no_grd jstars_abl_zero_film jstars_abl_no_unc; do
  bash scripts/submit_jstars_snaphu_multi_20261119.sh "$tag" "$EVAL_TS"
done
```

**One minute later, 7 of our tasks were running on cn1 (Milan).** The
`sbatch --test-only` estimate for `normal` had said Dec 3; backfill
placed us immediately.

### P8. Balancing variants with throttle + Nice

After moving to multi-partition, running counts drifted:

```
no_cls      13
no_tmp       5
no_grd       5
zero_film    3
no_unc       1
```

`no_cls` was getting most slots because its arrays were submitted
earliest, and its SNAPHU runs take 30 min each (low-coherence output
from a model trained without the closure loss) — hogs kept running.
Rebalance:

```bash
# Throttle down the leader
scontrol update ArrayTaskThrottle=4  JobId=2713016   # no_cls orig
scontrol update ArrayTaskThrottle=12 JobId=2713245   # no_cls multi

# Deprioritize it for future placements
scontrol update Nice=+200 JobId=2713245
scontrol update Nice=+200 JobId=2713016
```

Within half an hour the running mix was:

```
no_cls       7   (down from 13)
no_tmp       7
no_grd       7
zero_film    7
no_unc       6
```

Equal finish projected.

### P9. Watchers + dashboard

Two supervisors run detached:

- **SNAPHU submitter** (`watch_submit_jstars_snaphu_fast_20261119.sh`):
  waits for each variant's inference to complete, then submits its
  SNAPHU array. One-shot; exits after submitting the last variant.
- **Stage-C runner** (`watch_stage_c_jstars_20261119.sh`): polls every
  2 min for each variant hitting 224/224 `unw_phase_film_unet.tif`,
  then runs the 30-second M2/M4 computation. One-shot per variant.

Both launched with the §9-Part-I incantation:

```bash
nohup setsid bash scripts/watch_stage_c_jstars_20261119.sh ARGS \
    > logs/JSTARS_stage_c_watcher_$(date -u +%Y%m%d_%H%M).log 2>&1 < /dev/null & disown
```

Live monitoring via `scripts/status_jstars_eval_20261119.sh`:

```bash
watch -c -n 30 bash scripts/status_jstars_eval_20261119.sh
```

Shows GPU state, per-variant Stage-A/B/C progress (file counts, not
SLURM numbers), running vs pending SLURM task counts per variant with
partition CSV, both watchers' liveness and their last log lines, and
total R/PD counts.

### P10. Resubmitting the few that fail

Any SNAPHU task that fails (timeout, node death, random SIGKILL) just
leaves the pair without a `unw_phase_film_unet.tif`. The idempotent
pattern recovers them trivially (§10 Part I):

```bash
TREE=data/processed/JSTARS_eval_jstars_abl_no_unc_20261119_0926
PL=logs/JSTARS_snaphu_pairs_no_unc_undone_$(date -u +%Y%m%d_%H%M).txt
comm -23 \
  <(find "$TREE" -name ifg_film_unet.tif -printf '%h\n' | sort -u) \
  <(find "$TREE" -name unw_phase_film_unet.tif -printf '%h\n' | sort -u) \
  | xargs -n1 basename > "$PL"
N=$(wc -l < "$PL")
sbatch --array="1-${N}%12" --export=ALL,TREE="$TREE",PAIRLIST="$PL" \
       --job-name="JSTARS_snaphu_jstars_abl_no_unc_resubmit" \
       scripts/slurm_jstars_snaphu_multi_20261119.sbatch
```

No state needed — the tree itself is the source of truth.

### P11. Summary: what moved the needle

In order of impact for this workload:

1. **Multi-partition `itee,normal` + short `--time`** → backfill got us
   onto Milan nodes with hundreds of GB free. Worth ~3× throughput.
2. **Shrinking `--mem` from 16 GB to 4.8 GB** (measured peak) →
   doubled concurrent-task density on memory-pressured nodes.
3. **Excluding sapphirerapids with `--constraint`** → eliminated the
   1-second task deaths with no logs.
4. **Nice + throttle rebalance** → made all 5 variants finish within
   hours of each other instead of `no_cls` being a day late.
5. **File-based progress in the dashboard** → the one signal that never
   lied even when SLURM's own task counts were confusing.

### P12. Mistakes I made

- Submitted with `--mem=16G` → OOM-slot-less. **Measure first.**
- Wrote `--mem=6G` to `scontrol update` → "Invalid value". **MB only.**
- Used `set -euo pipefail` in a watcher → one failing variant killed the
  whole chain. **In supervisors, drop `-e`.**
- Named a Python loop variable `pd` → shadowed `import pandas as pd` 50
  lines down. **Loop vars must not collide with imports.**
- Killed watchers by matching too broadly with `pgrep` → also killed my
  own terminal shell. **`pgrep -af` to inspect, then kill by exact PID.**
- Trusted `sbatch --test-only` estimates ("Dec 3") as reality. **Real
  submits can start weeks earlier because of backfill.**
- Over-constrained features (`cpu_skylake` only) → missed Milan nodes
  with huge free RAM. **Use OR expressions for the safe superset.**

---

## Cheat sheet

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
sbatch --test-only -p ... --mem=... -c ... --wrap=":"   # dry-run

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
sbatch --array="1-$(wc -l < PAIRLIST)%K" \
       --export=ALL,PAIRLIST=PAIRLIST,... script
```

That's it. The two concepts that take the longest to internalise are:

- **Memory is the real scheduling constraint**, not CPU, on any shared
  cluster.
- **Backfill rewards small + short**, which means you win by *asking
  for less*, not more.

Everything else in SLURM is either a knob to tune those two levers, or
a diagnostic for when they fail.
