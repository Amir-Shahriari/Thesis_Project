# Suspension note

2026-10-18 01:11: at the author's request, all five shard worker processes (PIDs 27080, 27512,
20752, 33260, 33756) were suspended with NtSuspendProcess (scripts/ablation_shards.ps1).
Suspension freezes a process in place; seeds, training trajectories and results are unaffected.
Only the train_s wall-clock of an arm spanning the pause includes the paused time; train_s is
not used in any pre-registered test.

Resumed: all 5 workers resumed 2026-10-18 05:12 at the author's request (no progress lost; suspended processes kept their state).
