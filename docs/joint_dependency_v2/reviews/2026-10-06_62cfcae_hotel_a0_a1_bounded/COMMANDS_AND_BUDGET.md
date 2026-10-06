# Commands and budget

The final absolute command, source commit, manifest path, PID/PGID/SID and Beijing
deadline are written to `LAUNCH_RECEIPT.json` only after preflight. The launcher form is:

```bash
nohup setsid -f /absolute/python -u /absolute/source/tools/hotel_a0_a1_queue.py \
  --manifest /absolute/run/QUEUE_MANIFEST.json \
  >/absolute/run/queue.log 2>&1 </dev/null &
```

The manager runs six 40-epoch arms in the preregistered order, then E_joint baseline,
six selected-checkpoint five-seed evaluations and the three A1-3101 interventions.
Per-arm limit is 21,600 s; aggregate training allowance is 36 h; the absolute queue
deadline is 48 h from formal launch. Peak reserved CUDA memory must not exceed
12,884,901,888 bytes.

Safe stop is performed only after matching the recorded manager PID and
`/proc/<pid>/stat` start ticks. Never use `pkill python`.
