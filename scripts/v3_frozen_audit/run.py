"""Optional explicit fresh-output wrapper for reproducing this read-only audit."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['replay','analyze'])
    p.add_argument('--root',type=Path,required=True,help='separate audit outputs; replay refuses an existing directory')
    p.add_argument('--docs',type=Path,help='new report-data directory; required for analyze')
    a=p.parse_args()
    import replay
    replay.OUT=a.root.resolve()
    if a.phase=='replay':
        replay.main()
    else:
        if a.docs is None:p.error('--docs is required for analyze')
        import analyze
        analyze.OUT=replay.OUT;analyze.DOC=a.docs.resolve()
        analyze.main()


if __name__=='__main__':main()
