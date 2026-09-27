# Running this project on NCI Gadi

Gadi runs **PBS Pro**. Work is submitted as batch jobs, never run on login nodes.
This directory holds a ready-to-submit pipeline for the multi-seed geometry study.

Version strings and queue limits below change over time — the commands in
"Orient yourself" print the current truth, so prefer them over anything asserted here.

---

## 1. Access

You need three things: an NCI account, membership of a **project** with a compute
allocation (the `-P` code, e.g. `ab12`), and SSH.

```bash
ssh <nci-username>@gadi.nci.org.au
```

NCI requires multi-factor authentication for interactive access. Enrol at
<https://my.nci.org.au> before your first login; you will be prompted for a TOTP
code from your authenticator app alongside your password. If you use SSH keys,
register the public key through `my.nci.org.au` rather than appending it to
`~/.ssh/authorized_keys` by hand.

Convenience — add to your **local** `~/.ssh/config`:

```
Host gadi
    HostName gadi.nci.org.au
    User <nci-username>
    ServerAliveInterval 60
```

Then `ssh gadi`.

## 2. Orient yourself (run these first)

```bash
id                              # which project groups you belong to
nci_account -P <project>        # service units: granted, used, remaining
lquota                          # storage quota on /home, /scratch, /g/data
module avail python3            # which python builds exist right now
qstat -Qf normal | head -40     # queue limits, incl. max walltime by job size
```

## 3. Where to put the code

| Path | Use | Caution |
|---|---|---|
| `/home/<grp>/<user>` | dotfiles only | ~10 GB; **not** auto-mounted in jobs |
| `/scratch/<project>/<user>` | active runs, checkpoints | **purged** — files untouched for ~100 days are deleted |
| `/g/data/<project>/<user>` | anything you must keep | only if your project has an allocation |

Put the repo on `/scratch`, and copy anything you need to keep to `/g/data` or
off-cluster when a run finishes.

```bash
cd /scratch/<project>/$USER
git clone <your-repo-url> qwarm-gnn-rl
cd qwarm-gnn-rl
```

**The single most common Gadi mistake:** a job that cannot see its own files.
Every filesystem a job touches must be declared:

```
#PBS -l storage=scratch/<project>+gdata/<project>
```

Omit it and the job starts, fails to find the repo, and dies immediately.

## 4. Build the environment (login node — compute nodes have no internet)

```bash
bash hpc/setup_env.sh
```

Creates `.venv-gadi/` with torch (CPU), torch_geometric, qiskit-aer and the
`qwarm` package. Override the python module if the default is gone:

```bash
PYTHON_MODULE=python3/3.12.1 bash hpc/setup_env.sh
```

## 5. Submit

```bash
mkdir -p logs

# Stage 1 -- train checkpoints across seeds and scales (the expensive part)
qsub -P <project> -l storage=scratch/<project> hpc/train_array.pbs

# Stage 2 -- the geometry study, chained to run only if stage 1 succeeds
qsub -P <project> -l storage=scratch/<project> \
     -W depend=afterokarray:<stage1-jobid>[] hpc/geometry_study.pbs
```

Monitor:

```bash
qstat -u $USER            # queued / running
qstat -f <jobid>          # full detail, incl. why a job is held
qdel <jobid>              # cancel (append [] for a whole array)
tail -f logs/qwarm-train.o<jobid>.<idx>
```

## 6. Sizing

`hpc/train_array.pbs` defaults to 5 seeds × 3 scales = 15 tasks, `ncpus=8`,
24 h walltime, on the `normal` queue.

**Use the CPU queue, not a GPU queue.** The wall-clock driver here is the
CPU-bound simulated-QAOA oracle — your manifest records 36,866 s for one 25×25
warm cell under the full pool against 93–155 s for a cold cell — and the graphs
are far too small for a V100 to pay for itself. Parallelism across cells is the
win, and that is exactly what a job array buys.

Cost scales as `ncpus × walltime × queue charge rate`; check the rate with
`nci_account` and estimate before submitting 15 × 24 h × 8 cores.

Two levers if the array is too slow or too expensive:

- `ORACLE_POOL=classical_only` (the default) skips QAOA entirely. The geometry
  results do not depend on demonstration provenance — your own source-invariance
  ablation established that — so only use the full pool where the
  demonstration-source claim itself needs it.
- Drop `100x100` from `SCALES` and re-run it separately with a longer walltime.

Both stages are resumable: `--skip-existing` leaves finished checkpoints alone,
so a timed-out array task can simply be resubmitted.

## 7. Getting results back

```bash
# from your local machine
rsync -avz gadi:/scratch/<project>/$USER/qwarm-gnn-rl/runs/ ./runs/
```

Do bulk data movement through the `copyq` queue rather than a login node when
it is large.
