# ZARA1 matched-GDTS asset recipe

The ZARA1 preparation pipeline uses the fixed legacy GDTS implementation at
commit `092ed91f1f4117f50aa835ec8e47cd2c9ce74a98` as a read-only dependency.
The executable values are frozen in
`configs/asset_preparation/zara1_gdts.yaml`.

- Goal U-Net: seed 3101, 150 epochs, Adam at `1e-3`, ExponentialLR 0.99,
  batch 64, FP32, validation every epoch, no augmentation, deterministic batch
  order. The strict-earlier minimum validation Goal BCE checkpoint initializes
  EBJD's trainable map U-Net.
- Matched GDTS: fresh non-goal modules and optimizer state, initialized only
  from the selected Goal U-Net; seed 3101, 250 epochs, Adam at `1e-3`,
  ExponentialLR 0.99, batch 64, FP32, original augmentation and train shuffle,
  validation every 10 epochs. Selection is the native minimum validation ADE.
- Binding evaluation: the selected full GDTS checkpoint is evaluated with 20
  candidates and 20 DDIM steps on synchronized ZARA1 windows. Seed 2035 binds
  the EBJD validation selector. Seeds 2035--2039 are completed after EBJD
  training for the final comparison.

The independent fragment cache used to train GDTS is not treated as the joint
metric manifest. Before matched evaluation, the pipeline creates the existing
GDTS synchronized-scene cache and requires full coordinate, scene-identity and
semantic-map parity against the paper manifest. The official ETH/UCY package
mirrors validation and test files; their identities remain recorded separately.

The adapter adds complete Python, NumPy, CPU/CUDA Torch, and loader-generator
RNG state to epoch-boundary resume checkpoints. This is an execution/recovery
addition only; it does not change the GDTS model, objective, optimizer or data
order. Best checkpoints remain weights for selection rather than resume points.
