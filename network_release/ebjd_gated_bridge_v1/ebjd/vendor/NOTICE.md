# Vendored component notice

`unet.py` is a clean, local implementation of the channel/topology contract
specified for EBJD. It does not import or modify the repository's legacy GDTS
U-Net. The topology is conventional U-Net (Ronneberger et al., 2015); no
third-party source code was copied.

The accepted initialization tensor layout was inspected from the repository's
fold-matched goal checkpoint produced at base lineage
`092ed91f1f4117f50aa835ec8e47cd2c9ce74a98`; its exact artifact and SHA256 are
recorded in `docs/IMPLEMENTATION_RECEIPT.md`. This clean implementation and its
local metric adapter are distributed under the repository root MIT License.
