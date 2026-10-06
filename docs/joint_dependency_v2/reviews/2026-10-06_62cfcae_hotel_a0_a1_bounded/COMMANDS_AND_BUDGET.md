# Commands and budget

Preflight passed. The fixed source commit is
`664ad0410f5cf3b44d3a6c1892c98e37aa936b9e`; the queue deadline is
2026-10-08 21:17:16 Beijing time. The formal launcher is:

```bash
nohup setsid -f /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python -u \
  /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_hotel_a0_a1_5b6e4e4/tools/hotel_a0_a1_queue.py \
  --manifest /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4/formal_launch/QUEUE_MANIFEST.json \
  >/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4/formal_launch/queue.log 2>&1 </dev/null &
```

The manager runs six 40-epoch arms in the preregistered order, then E_joint baseline,
six selected-checkpoint five-seed evaluations and the three A1-3101 interventions.
Per-arm limit is 21,600 s; aggregate training allowance is 36 h; the absolute queue
deadline is 48 h from formal launch. Peak reserved CUDA memory must not exceed
12,884,901,888 bytes.

Monitor:

```bash
tail -n 80 -f /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4/formal_launch/queue.log
```

Safe stop is performed only after matching the recorded manager PID and
`/proc/<pid>/stat` start ticks. Never use `pkill python`.
