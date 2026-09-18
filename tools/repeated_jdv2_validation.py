#!/usr/bin/env python3
"""Repeat JDV2 validation for one checkpoint under fixed stochastic seeds."""

import argparse
import json
from pathlib import Path

from src.jdv2_audit import repeated_validation
from tools.audit_jdv2_stage_a import _active_evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[2035, 2036, 2037, 2038, 2039])
    cli = parser.parse_args()
    output = Path(cli.output).resolve()
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / 'evaluation', cli.device)
    result = {
        'checkpoint': str(Path(cli.checkpoint).resolve()),
        'checkpoint_epoch': int(epoch),
        'split': 'valid',
        'sampling_policy': 'stochastic deployed policy (not MAP)',
        'seeds': cli.seeds,
        **repeated_validation(evaluator, epoch, cli.seeds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
