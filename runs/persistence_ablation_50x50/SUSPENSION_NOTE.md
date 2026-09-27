# Suspension note

2026-09-26 ~23:10: at the author's request (GPU needed for another application),
two of the five shard worker processes (PIDs 25028 and 19844) were suspended with
NtSuspendProcess and later resumed. Suspension freezes a process in place; seeds,
training trajectories and results are unaffected. The only effect is that the
`train_s` wall-clock of any arm that spanned the pause includes the paused time.
`train_s` is not used in any pre-registered test.

Control: `scripts/ablation_shards.ps1 status|resume|suspend N`.

All five shards were suspended at about 23:20 (game running), then resumed at 2026-09-27 00:34, with priority raised to AboveNormal.
