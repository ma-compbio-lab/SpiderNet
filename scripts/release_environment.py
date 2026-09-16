"""Relocate only filesystem inputs for the saved-result distribution."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def native(path):
    value = str(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return value

def environment(root=ROOT):
    root = Path(root).resolve()
    work = root / '.spidernet' / 'workspace'
    results, data = work / 'Results', work / 'Data'
    values = {key: str(work) for key in ('SPIDERNET_ROOT', 'SPIDERNET_WORKSPACE_ROOT',
        'SPIDERNET_WORKSPACE_DIR', 'SPIDERNET_WORKSPACE')}
    values.update({
        'SPIDERNET_DATA_ROOT': str(data), 'SPIDERNET_RESULTS_ROOT': str(results),
        'SPIDERNET_PACKAGE_ROOT': str(root/'SpiderNet'),
        'SPIDERNET_PROJECT_DIR': str(root/'SpiderNet'),
        'SPIDERNET_AGINGBRAIN_DATA_ROOT': str(data/'AgingBrain'),
        'SPIDERNET_AGINGBRAIN_RESULTS_ROOT': str(results/'AgingBrain'),
        'SPIDERNET_AGINGBRAIN_OUTPUT': str(root/'Tutorial/AgingBrain/output'),
        'HGSOC_DATA_ROOT': str(data/'HGSOC'), 'HGSOC_RESULTS_ROOT': str(results/'HGSOC'),
        'HGSOC_PROCESSED_DATA': str(results/'HGSOC/ProcessedData'),
        'HGSOC_RUN_DIR': str(results/'HGSOC/V1/SpiderNet_Result_dim15'),
        'HGSOC_OUTPUT': str(root/'Tutorial/HGSOC/output'),
        'SPIDERNET_PERTURBFISH_DATA_ROOT': str(data/'PerturbFISH'),
        'SPIDERNET_PERTURBFISH_OUTPUT_ROOT': str(results/'PerturbFISH'),
        'SPIDERNET_PERTURBFISH_PROCESSED_ROOT': str(results/'PerturbFISH/ProcessedData'),
        'SPIDERNET_PERTURBFISH_RESULTS_DIR': str(results/'PerturbFISH/V1/SpiderNet_Result_dim23'),
        'SPIDERNET_PERTURBFISH_PLOT_DIR': str(root/'Tutorial/PerturbFISH/output'),
        'SPIDERNET_PANCANCER_DATA': str(data/'Pancancer'),
        'SPIDERNET_PANCANCER_PROCESSED': str(results/'Pancancer/ProcessedData_entire'),
        'SPIDERNET_PANCANCER_RESULTS': str(results/'Pancancer/V1/SpiderNet_Result_dim11'),
        'SPIDERNET_PANCANCER_OUTPUT': str(root/'Tutorial/Pancancer/output'),
        'SIMULATION_DATA_ROOT': str(data/'Simulation'),
        'SIMULATION_RESULT_ROOT': str(results/'Simulation'),
        'SPIDERNET_ROBUSTNESS_WORK_DIR': str(results/'MI_robustness_S31_S32'),
        'SPIDERNET_ROBUSTNESS_OUTPUT_DIR': str(root/'Tutorial/Robustness_stability/output'),
        'PYTHONPATH': os.pathsep.join([str(root/'SpiderNet'), str(root/'scripts')]),
        'PYTHONDONTWRITEBYTECODE': '1',
    })
    return {key: (native(value) if key not in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE') and not key.startswith('SPIDERNET_PANCANCER_') else value)
            for key,value in values.items()}

def configure_manifests(root=ROOT):
    """Generate location-only manifests; original run metadata is archived separately."""
    root = Path(root).resolve()
    results = root / '.spidernet/workspace/Results'
    for study, dim in [('AgingBrain',30),('HGSOC',15),('PerturbFISH',23),('Pancancer',11)]:
        dest = results/study/'run_dirs.json'
        run = results/study/'V1'/f'SpiderNet_Result_dim{dim}'
        payload={'run_dir':native(run), 'model_dir':native(run/'Model')}
        if dest.exists():
            previous=json.loads(dest.read_text(encoding='utf8'))
            if {k:native(v) for k,v in previous.items()} != payload:
                raise ValueError(f'Refusing to overwrite a different run manifest: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')

if __name__ == '__main__':
    import argparse, shlex
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell',choices=['powershell','bash','json'],default='json')
    parser.add_argument('--configure',action='store_true')
    args=parser.parse_args()
    if args.configure: configure_manifests()
    for key,value in environment().items():
        if args.shell=='powershell': print(f"$env:{key} = '"+value.replace("'","''")+"'")
        elif args.shell=='bash': print(f'export {key}={shlex.quote(value)}')
    if args.shell=='json':print(json.dumps(environment(),indent=2))
