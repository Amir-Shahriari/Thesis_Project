@echo off
REM Launch the 50x50 A/B driver detached from the launching terminal session.
REM
REM Why this exists: the driver was first run as a child of the agent session
REM and a Windows Update restart (2026-08-12 07:37, TrustedInstaller "Upgrade
REM (Planned)") killed the masked arm four minutes in. A .bat wrapper is used
REM rather than passing a command string to bash -c through Start-Process,
REM because the nested quoting silently fails and the process exits without
REM writing a log.
cd /d C:\Users\amirh\Desktop\qwarm-gnn-rl
"C:\Program Files\Git\bin\bash.exe" scripts/run_50x50_ab_gpu.sh > runs\ab_gpu_driver.log 2>&1
