"""One isolated GPU stage; repeated invocations resume the same stage directory."""
import argparse
import json
from pathlib import Path
import traceback
import sys

import torch
from bcep_joint.common import read, write
from bcep_v2.runtime import configure_cuda
from .common import status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('upstream', 'dependency', 'test'))
    parser.add_argument('--fold', required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    folder = args.run/args.fold
    result_path = folder/('UPSTREAM_FROZEN.json' if args.stage == 'upstream' else
                         'B_FROZEN.json' if args.stage == 'dependency' else 'TEST_RESULT.json')
    torch.set_num_threads(1)
    try:
        if args.stage == 'upstream':
            from .upstream import prepare_and_run_fold
            result = prepare_and_run_fold(args.fold, folder/'upstream', 'cuda')
        else:
            configure_cuda()
            upstream = read(folder/'UPSTREAM_FROZEN.json')
            if args.stage == 'dependency':
                from .train import run_training
                result = run_training(upstream, folder/'dependency', 'cuda')
            else:
                from .test import run_test
                result = run_test(upstream, read(folder/'B_FROZEN.json'),
                    read(args.run/'FROZEN_MODELS.json'), folder/'test', 'cuda')
        write(result_path, result)
        print(json.dumps(dict(stage=args.stage, fold=args.fold, result=result), ensure_ascii=False), flush=True)
    except BaseException as error:
        from .upstream import MissingUpstreamAssets
        if isinstance(error, MissingUpstreamAssets):
            write(folder/'MISSING_ASSETS.json', dict(status='MISSING_ORIGINAL_ASSETS',
                fold=args.fold, missing=error.missing, error=repr(error)))
            raise SystemExit(3)
        status(folder, 'STAGE_INTERRUPTED_RESUME_SAME_RUN', stage=args.stage,
               error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
