@echo off
REM ---------------------------------------------------------------------------
REM Replicate the seed-42 goal-relative result across the remaining four seeds.
REM
REM Seed 42 established, at n=145 held-out queries:
REM     warm baseline -> warm +goal-relative :  48 -> 71 reach, McNemar p=0.0022
REM     cold baseline -> cold +goal-relative :  68 -> 70 reach, p=0.89 (nothing)
REM so the effect is warm-specific. This grid decides whether it replicates.
REM
REM Both configs are run per seed because the comparison that matters is the
REM PAIRED baseline -> goal-relative change within each seed, on identical
REM queries. Baseline uses no tag suffix, matching how seed 42 was run.
REM
REM Run from anywhere:  scripts\run_seed_grid.bat
REM Roughly 5-7 hours total. Safe to interrupt: every (seed, config) writes its
REM own JSON, and re-running skips nothing, so restart from the seed you lost.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

set SEEDS=1337 2024 7 314159
set SCALE=25x25
set NEVAL=150

echo ==========================================================
echo  Seed grid: %SEEDS%   scale=%SCALE%   held-out=%NEVAL%
echo  Started %DATE% %TIME%
echo ==========================================================

for %%S in (%SEEDS%) do (
    echo.
    echo ---------- seed %%S : GOAL-RELATIVE ----------
    python scripts\run_fleet_mode.py --seed %%S --scale %SCALE% --goal-relative --tag-suffix _gr
    python scripts\reeval_fleet.py  --seed %%S --scale %SCALE% --tag-suffix _gr --n-eval-queries %NEVAL%

    echo.
    echo ---------- seed %%S : BASELINE ----------
    python scripts\run_fleet_mode.py --seed %%S --scale %SCALE%
    python scripts\reeval_fleet.py  --seed %%S --scale %SCALE% --n-eval-queries %NEVAL%
)

echo.
echo ==========================================================
echo  Finished %DATE% %TIME%
echo  Now run:  python scripts\aggregate_seed_grid.py
echo ==========================================================
endlocal
