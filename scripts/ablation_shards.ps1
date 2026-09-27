# Pause or resume the persistence-ablation shard processes (Windows).
# Suspending freezes a process in place: it keeps its GPU memory and state and
# continues exactly where it stopped on resume, so results are unaffected
# (runs are seeded and deterministic). Only the recorded train_s wall-clock of
# an arm that spans a pause is inflated; it is not used in any analysis.
#
# Usage (PowerShell, from the repo root):
#   .\scripts\ablation_shards.ps1 status
#   .\scripts\ablation_shards.ps1 resume          # resume every shard
#   .\scripts\ablation_shards.ps1 suspend 2       # pause 2 shards (about half the GPU)
param([string]$Action = "status", [int]$Count = 0)

Add-Type -Name NtProc -Namespace Win32 -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
'@ -ErrorAction SilentlyContinue

# Worker = the conda python child that actually holds the GPU context.
$workers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*run_persistence_ablation*' -and $_.ExecutablePath -notlike '*\.venv\*' }

function Is-Suspended($id) {
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if (-not $p) { return $false }
    return (($p.Threads | Where-Object { $_.WaitReason -eq 'Suspended' }).Count -eq $p.Threads.Count)
}

switch ($Action) {
    "status" {
        foreach ($w in $workers) { "{0}  {1}" -f $w.ProcessId, $(if (Is-Suspended $w.ProcessId) { "PAUSED" } else { "running" }) }
        if (-not $workers) { "no shard processes found (run finished or not started)" }
    }
    "resume" {
        foreach ($w in $workers) {
            if (Is-Suspended $w.ProcessId) {
                [void][Win32.NtProc]::NtResumeProcess((Get-Process -Id $w.ProcessId).Handle)
                "resumed $($w.ProcessId)"
            }
        }
    }
    "suspend" {
        $running = $workers | Where-Object { -not (Is-Suspended $_.ProcessId) } | Select-Object -First $Count
        foreach ($w in $running) {
            [void][Win32.NtProc]::NtSuspendProcess((Get-Process -Id $w.ProcessId).Handle)
            "paused $($w.ProcessId)"
        }
    }
}
