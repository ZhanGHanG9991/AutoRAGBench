"""Stage the portable preparation tree and runtime for one AutoRAGBench task."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

def overlay(src,dst):
    if src.exists(): shutil.copytree(src,dst,dirs_exist_ok=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--tune',type=Path,required=True,help='200 public training rows in upstream custom-exam JSON format')
    a=p.parse_args();base=Path(__file__).resolve().parents[1];ws=a.workspace.resolve()
    framework='autorag' if base.name=='agentic_autorag' else 'flashrag'
    rows=json.loads(a.tune.read_text());assert len(rows)==200
    assert all('question' in x and 'canonical_answer' in x and 'id' in x for x in rows)
    assert not (ws/'submission').exists(), 'use a fresh build workspace'
    prep=ws/'prep';(prep/'jobs').mkdir(parents=True,exist_ok=True)
    overlay(base/'service/submission',prep)
    overlay(base/'service/configs'/a.task/'submission',prep)
    # Build import paths match the original packaging layout.
    overlay(prep/'runtime',prep/'submission'/framework)
    for f in (base/'build').glob('*.py'):
        if f.name!='prepare.py':shutil.copy2(f,prep/'jobs'/f.name)
    overlay(base/'build/configs/autorag',prep/'configs/autorag')
    if framework=='autorag' and a.task in ('hotpotqa','pubmedqa','webqsp'):
        shutil.copy2(base/'build/configs/additional_tasks/autorag_optimize.py',prep/'jobs/autorag_optimize.py')
    if framework=='autorag' and a.task in ('cwq','musique','multihop_rag'):
        shutil.copy2(base/'build/configs/revised_logging/autorag_optimize.py',prep/'jobs/autorag_optimize.py')
    if framework=='flashrag':
        if a.task in ('hotpotqa','pubmedqa','webqsp'):
            shutil.copy2(base/'build/configs/additional_tasks/build_bm25_index.py',prep/'jobs/build_bm25_index.py')
        if a.task=='multihop_rag':overlay(base/'build/configs/multihop_rag',prep/'jobs')
        if a.task=='webqsp':
            shutil.copy2(base/'build/configs/webqsp/platform_chat.py',prep/'lib/exp3lib/platform_chat.py')
    else:
        # The optimizer package writer copies this directory although it is unused by AutoRAG.
        (prep/'flashrag').mkdir(exist_ok=True)
    (prep/'splits'/a.task).mkdir(parents=True,exist_ok=True)
    shutil.copy2(a.tune,prep/'splits'/a.task/'tune.json')
    runtime=ws/'runtime';runtime.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(runtime/'env')],check=True)
    overlay(a.models,runtime/'models')
    if framework=='autorag':
        overlay(a.upstream/'agentic_autorag',runtime/'vendor/agentic_autorag')
    else:
        (runtime/'vendor').mkdir(exist_ok=True)
    print('Prepared',a.task,'at',ws)

if __name__=='__main__':main()
