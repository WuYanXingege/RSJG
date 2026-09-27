# JDV2 ETH Official JMM Evaluation

## 1. Purpose and decision

This report audits and executes RSJG Joint Dependency V2 on the official ETH
protocol used by *Joint Metrics Matter: A Better Standard for Trajectory
Forecasting* and *Joint Pedestrian Trajectory Prediction through Posterior
Sampling*.

The displacement-metric protocol is now implemented and validated. The frozen
ETH Stage-B V2-A epoch-5 checkpoint was evaluated on all 253 official ETH
windows with 20 coupled joint samples. The internal scorer agrees with the
authors' original JMM scorer to below 1.0e-8 on every displacement metric.

| Metric (m, K=20, lower is better) | JDV2 Stage-B V2-A |
|---|---:|
| minADE | 0.413709 |
| minFDE | 0.594282 |
| minJADE | 0.480336 |
| minJFDE | 0.763341 |

This is a protocol-valid displacement evaluation, but it is **not** yet a
strict input-fair comparison to Table 1 of *Joint Metrics Matter*: that table
retrained comparison methods without images/semantic maps, whereas the frozen
GDTS/JDV2 candidate model uses the ETH semantic map. The current implementation
also does not report the paper's CRmean/CRJADE collision metrics.

## 2. Meaning of ETH (1.4)

The value 1.4 is not a protocol version or an error metric. It is sequence
density: the mean number of complete agents per 20-frame ETH window. This run
reproduces 364 agent instances / 253 scenes = 1.4387351779.

| Agents in window | Number of windows |
|---:|---:|
| 1 | 183 |
| 2 | 38 |
| 3 | 25 |
| 4 | 5 |
| 5 | 2 |

## 3. Authoritative protocol

- official JMM ETH biwi_eth_agentformer.txt;
- data SHA256 d7cdcedd6472ebaa794bc56e037542e350dfe2ee992f818ab8c417e9fc2a17ad;
- 253 sliding windows and 364 complete agent instances;
- 8 observed and 12 predicted steps;
- 2.5 Hz (dt=0.4 s), stride one source frame;
- world coordinates in metres;
- exactly 20 predicted joint futures per scene;
- minADE/minFDE select the best sample separately per agent, then aggregate
  over 364 agents;
- minJADE/minJFDE select one common sample index for every agent in a scene,
  then aggregate over 253 scenes.

The joint metrics never mix one agent from sample k1 with another from k2.

## 4. Frozen artifacts and provenance

| Artifact | Value |
|---|---|
| implementation commit | 05378c7b5cb589d64efc3c00391a89f90cb64194 |
| implementation tracked tree | clean at cache build and inference |
| checkpoint | Stage-B V2-A best, epoch 5 |
| checkpoint SHA256 | e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73 |
| frozen Goal U-Net SHA256 | 126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950 |
| official data SHA256 | d7cdcedd6472ebaa794bc56e037542e350dfe2ee992f818ab8c417e9fc2a17ad |
| evaluation seed | 2035 |
| model candidate bank | K=21 |
| exported joint futures | K=20 |
| sampler | exact lexicographic persistent tie |
| cache record-set SHA256 | 4e56bbafd999b84037100665713ead761e4b8ac48c0fc2882472b4b19858448d |
| authors' evaluator commit | 894182b5ea7dac3c93bbe0bcc18c156c932398b7 |

The deployment cache contains no future supervision and no teacher sidecar. It
locks the source Goal checkpoint hash, official data hash, candidate seed/order,
dt, K and sparse graph configuration. Inference fails closed on manifest,
record-set, window, candidate-shape or source-checkpoint mismatch.

## 5. Implementation

The official evaluator now has an explicit JDV2 cache phase:

1. build the exact 253-window future-free deployment cache;
2. validate its manifest and every record;
3. export 20 joint futures without computing metrics;
4. score the completed frozen export in a separate command.

The default training and inference paths were not modified. No model, loss,
sampler, checkpoint or learned weight changed.

Changed files:

- tools/evaluate_jmm_official.py
- tests/test_jmm_protocol.py

## 6. Exact commands

Interpreter:

~~~bash
PY=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
~~~

Cache:

~~~bash
$PY tools/evaluate_jmm_official.py build-jdv2-cache \
  --run-dir outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v2a_eth_seed2035 \
  --cache-dir outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/cache/eth_official_goal_126acf_seed2035_commit05378c7 \
  --device cuda:0 --seed 2035
~~~

Inference:

~~~bash
$PY tools/evaluate_jmm_official.py infer \
  --run-dir outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v2a_eth_seed2035 \
  --checkpoint best \
  --jdv2-cache-dir outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/cache/eth_official_goal_126acf_seed2035_commit05378c7 \
  --jdv2-cache-seed 2035 \
  --output-dir outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7 \
  --device cuda:0 --seed 2035
~~~

Score:

~~~bash
$PY tools/evaluate_jmm_official.py score \
  --trajectory-root outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7/trajectories/gdts \
  --output-json outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7/metrics.json
~~~

## 7. Independent scorer cross-check

The same files were evaluated with the authors' original evaluator at commit
894182b5ea7dac3c93bbe0bcc18c156c932398b7.

| Metric | Internal | Authors' scorer | Absolute difference | Count |
|---|---:|---:|---:|---:|
| minADE | 0.413709296249 | 0.413709288043 | 8.21e-9 | 364 agents |
| minFDE | 0.594281744287 | 0.594281752404 | 8.12e-9 | 364 agents |
| minJADE | 0.480336172549 | 0.480336168302 | 4.25e-9 | 253 scenes |
| minJFDE | 0.763341155542 | 0.763341146076 | 9.47e-9 | 253 scenes |

Status: **PASS**. The tiny differences are float64/float32 accumulation, not
protocol or aggregation differences.

## 8. Context against the two papers

| Method | ETH minJADE | ETH minJFDE |
|---|---:|---:|
| AgentFormer (Joint Metrics Matter) | 0.482 | 0.794 |
| Joint AgentFormer (Joint Metrics Matter) | 0.485 | 0.798 |
| GFTD (Posterior Sampling) | 0.505 | 0.873 |
| GFTD + RePaint (Posterior Sampling) | 0.514 | 0.906 |
| JDV2 Stage-B V2-A, seed 2035 | 0.480 | 0.763 |

The JDV2 row must not be advertised as an input-controlled SOTA claim. Metric,
data, horizon, density and K contracts match, but the JMM fair-comparison table
explicitly withheld images and semantic maps while this frozen GDTS/JDV2 run
uses semantic-map context.

## 9. Compliance matrix

| Requirement | Status | Evidence |
|---|---|---|
| official ETH file/hash | PASS | exact source hash |
| 253 scenes / 364 agents / density 1.438735 | PASS | strict parser |
| 8 observed / 12 future / 2.5 Hz | PASS | builder and run config |
| complete-agent filtering | PASS | official window builder |
| world metres | PASS | bridge assertion and export |
| K=20 predictions | PASS | 20 complete samples per scene |
| common scene sample for JADE/JFDE | PASS | coupled tensor and scorer |
| correct marginal/joint weighting | PASS | 364 vs 253 counts |
| future-free JDV2 cache | PASS | manifest and record checks |
| authors' scorer agreement | PASS | maximum error below 1e-8 |
| JMM CRmean/CRJADE | NOT IMPLEMENTED | no substitute metric claimed |
| JMM Table-1 input fairness | NOT SATISFIED | semantic map is used |
| multi-seed uncertainty | NOT RUN | requested seed-2035 K=20 run |

## 10. Tests and artifacts

Validation:

- python -m compileall -q .: PASS
- pytest -q: 476 passed, 12 warnings
- git diff --check: PASS
- JMM-specific tests: 8 passed
- cache: 253/253 records
- inference: 253/253 scenes
- official scorer cross-check: PASS

Artifacts:

- cache manifest:
  outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/cache/eth_official_goal_126acf_seed2035_commit05378c7/manifest.json
- evaluation manifest:
  outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7/manifest.json
- metrics:
  outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7/metrics.json
- official cross-check:
  outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/stage_b_v2a_epoch005_seed2035_commit05378c7/official_scorer_crosscheck.json

## 11. Final assessment

The code now satisfies the official ETH(1.4) displacement evaluation conditions
shared by both papers: official windows, density, horizons, world units, K=20,
coupled scene samples and correct marginal/joint aggregation. The frozen V2-A
seed-2035 result is reproducible and independently cross-validated.

Two claims remain outside this result:

1. full *Joint Metrics Matter* reporting additionally requires its exact
   CRmean/CRJADE collision implementation;
2. direct fair-comparison claims against its Table 1 require a
   no-image/no-semantic-map input-controlled run.

Neither limitation invalidates the reported displacement metrics, but both must
be disclosed in a paper table or benchmark claim.
