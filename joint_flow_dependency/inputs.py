"""Reuse the verified original-asset adapter without invoking the closed selector."""
from pathlib import Path
from bcep.contracts import digest
from scene_pair_selector.inputs import Inputs, ROOT, BASE, CHECKPOINT, CHECKPOINT_SHA, BANK_HASH, LIMITS, RECIPE, FORMAL

SOURCE = Path(__file__).resolve().parents[1]
RUN = ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122'
DOC = SOURCE/'docs/research/continuous_joint_flow/zara2_20261122'
ARCHIVE = '203b83c51f414beca340d62be27dc146cf11f3b0'
ARCHIVE_DIR = 'docs/research/scene_pair_selector/zara2_20261122/'


def source_hashes():
    return {str(p.relative_to(SOURCE)): digest(p)
            for folder in ('joint_flow_dependency', 'scene_pair_selector', 'bcep_v2', 'bcep_joint', 'bcep_formal', 'bcep', 'ebjd', 'selector_control', 'bcep_confirmation')
            for p in sorted((SOURCE/folder).rglob('*.py'))}
