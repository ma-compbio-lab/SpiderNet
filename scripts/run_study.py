"""Run an unchanged study runner with this checkout's explicit data locations."""
import argparse
import os
import subprocess
import sys
from release_environment import ROOT, environment, configure_manifests

STUDIES=('AgingBrain','HGSOC','Pancancer','PerturbFISH','Simulation',
         'Coupling_benchmark','Ablation_study','Robustness_stability')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=STUDIES)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    configure_manifests()
    extra=args.arguments
    if extra[:1]==['--']:extra=extra[1:]
    if not extra:
        parser.error('Supply an explicit mode, for example --check --plot-only or --plot-only')
    if args.study=='Robustness_stability' and '--package-root' not in extra:
        extra=['--package-root',str(ROOT/'SpiderNet'),*extra]
    env=os.environ.copy();env.update(environment())
    return subprocess.call([sys.executable,'-B','run_benchmarks.py',*extra],
        cwd=ROOT/'Tutorial'/args.study,env=env)

if __name__=='__main__':raise SystemExit(main())
