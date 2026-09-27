@echo off
REM ---------------------------------------------------------------------------
REM 50x50 goal-coverage replication  (thesis section 5.3, follow-on experiment)
REM
REM Tests whether the goal-coverage boundary established at 25x25 holds one
REM scale up: does the warm-start advantage still hold on DEMONSTRATED goals
REM and still vanish on HELD-OUT goals when the graph is 4x larger?
REM
REM WHY --episodes 80 (and not the 20 in the 50x50 preset):
REM   Multi-query training divides a fixed budget across the training goals, so
REM   what governs whether the comparison is measurable is EPISODES PER GOAL.
REM     25x25 single-query :  5 it x 100 ep            =  500 ep/goal
REM     25x25 multi-query  :  8 it x  25 ep            =  200 ep/goal  (40%)  -> resolved cleanly
REM     50x50 single-query : 10 it x 200 ep            = 2000 ep/goal
REM     50x50 multi-query  : 10 it x  80 ep            =  800 ep/goal  (40%)  <- this run
REM   The preset's 20 episodes would give only 200 ep/goal = 10% of the
REM   single-query budget, far thinner than the ratio that made 25x25 readable,
REM   and both arms would likely collapse to near-zero reach.
REM
REM COST (measured on this machine, RTX 5080, two calibration points):
REM   0.80 s/rollout warm, ~0.45 s/rollout cold; 16,000 rollouts per arm.
REM   ~3.6 h warm + ~1.9 h cold = ~5.4 h per seed; ~16 h for three seeds.
REM
REM Held-out evaluation is 150 queries, per the evaluation-power finding: at 20
REM queries the same agents reported the OPPOSITE ordering at 25x25.
REM
REM Run:  scripts\run_50x50_goalcoverage.bat
REM Safe to interrupt: each seed writes its own JSON. Restart by trimming SEEDS.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

set SEEDS=42 1337 2024
set SCALE=50x50
set ITERS=10
set EPISODES=80
set NTRAIN=20
set NEVAL=150
set TAG=_mq

echo ==========================================================
echo  50x50 goal-coverage replication
echo  seeds=%SEEDS%  %ITERS% it x %EPISODES% ep x %NTRAIN% queries
echo  held-out=%NEVAL%   est. ~5.4 h per seed
echo  started %DATE% %TIME%
echo ==========================================================

for %%S in (%SEEDS%) do (
    echo.
    echo ---------- seed %%S : training both arms ----------
    python scripts\run_fleet_mode.py --seed %%S --scale %SCALE% ^
        --n-train-queries %NTRAIN% --n-eval-queries 20 ^
        --iterations %ITERS% --episodes %EPISODES% ^
        --tag-suffix %TAG%

    echo ---------- seed %%S : re-evaluating at %NEVAL% held-out ----------
    python scripts\reeval_fleet.py --seed %%S --scale %SCALE% ^
        --tag-suffix %TAG% --n-train-queries %NTRAIN% --n-eval-queries %NEVAL%
)

echo.
echo ==========================================================
echo  finished %DATE% %TIME%
echo  Now run:
echo    python scripts\aggregate_seed_grid.py --scale %SCALE% --variant %TAG%
echo  ...then run scripts\analyse_50x50_goalcoverage.py for the paired warm-vs-cold analysis.
echo ==========================================================
endlocal
