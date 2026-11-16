"""Aggregate all registered repetitions without selecting a best run."""
import argparse,csv,json,pathlib,statistics
from .efficiency import build_stats
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);a=ap.parse_args()
    folder=ROOT/'records/multireader'/a.campaign;plan=json.loads((folder/'plan.json').read_text());rows=[];states=[]
    for run in plan['runs']:
        rec=ROOT/'records'/run['run_id'];result=rec/'run_result.json';scores=rec/'multireader_scores.json'
        outcome=json.loads(result.read_text()).get('outcome') if result.exists() else 'pending_or_running'
        build_rec=ROOT/'records'/run.get('build_run_id',run['run_id'])
        data=json.loads(scores.read_text()) if scores.exists() else {}
        preparation=data.get('retrieval',{})
        states.append({**run,'outcome':outcome,**build_stats(build_rec),
                       'shared_packing_seconds':preparation.get('packing_seconds'),
                       'shared_tokenizer_load_seconds':preparation.get('tokenizer_load_seconds'),
                       'evaluation_wall_seconds_this_attempt':data.get('elapsed_seconds_this_attempt')})
        if not data:continue
        retrieval=data['retrieval']['extraction']
        for s in data['readers']:
            attempts=s.get('n_generation_attempts',s['n']-data['retrieval']['n_errors'])
            rows.append({**run,'reader':s['reader'],'primary_metric':s['primary_metric'],'score':s['primary'],'n':s['n'],'n_invalid':s['n_invalid'],
                         'retrieval_latency_mean':retrieval.get('latency_mean'),'retrieval_mean_tokens':retrieval.get('mean_tokens'),
                         'retrieval_mean_calls':retrieval.get('mean_calls'),'reader_latency_mean':s.get('latency_mean'),
                         'reader_latency_p50':s.get('latency_p50'),'reader_latency_p95':s.get('latency_p95'),
                         'reader_questions_per_second':attempts/s['elapsed_seconds'],'reader_generation_attempts':attempts,
                         'reader_skipped_retrieval_invalid':s['n']-attempts,'reader_prompt_tokens':s['usage']['prompt_tokens'],
                         'reader_completion_tokens':s['usage']['completion_tokens'],'reader_generation_seconds':s['elapsed_seconds'],
                         'reader_startup_seconds':s.get('model_startup_seconds'),'reader_concurrency':s['concurrency'],
                         'evidence_sha256':s['evidence_sha256'],'job':s.get('slurm_job'),'node':s.get('node')})
    with (folder/'scores.csv').open('w',newline='') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (folder/'build_costs.csv').open('w',newline='') as f:
        if states:w=csv.DictWriter(f,fieldnames=list(states[0]));w.writeheader();w.writerows(states)
    tasks=list(dict.fromkeys(x['task'] for x in plan['runs']));panel=plan['readers'];means={};coverage={}
    for reader in panel:
        means[reader]={};coverage[reader]={}
        for task in tasks:
            xs=[r['score'] for r in rows if r['reader']==reader and r['task']==task]
            need=sum(r['task']==task for r in plan['runs']);coverage[reader][task]={'done':len(xs),'planned':need}
            means[reader][task]=statistics.mean(xs) if len(xs)==need else None
    macros={reader:(statistics.mean(means[reader].values()) if all(v is not None for v in means[reader].values()) else None) for reader in panel}
    summary={'campaign':a.campaign,'protocol':plan['protocol'],'states':states,'coverage':coverage,'task_means':means,'nine_task_macro':macros,
             'aggregation':'Per-task mean across all planned repeats, then unweighted task macro. Incomplete/flagged runs remain explicit; no best-run selection or imputed scores.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=[f'# {a.campaign}','',f"Protocol: {plan['protocol']}; builder: {plan['agent']}; build budget: {plan['build_seconds']/3600:g} h.",
           '','Each case uses the same sealed evidence for all five readers. Retrieval and reader timing are separate. Reader request latency is measured at concurrency 4; model startup is excluded from generation throughput. Failed retrieval rows score zero and remain in the scoring denominator; they are excluded from reader generation attempts and throughput.',
           '', 'Build budget is a maximum deadline, not a claim of continuous agent activity for that duration. The existing harness resumes an ended session only when at least 15 minutes remain. Actual build wall time and active fraction are reported separately in build_costs.csv.',
           '', '| Benchmark | '+' | '.join(panel)+' |','|---|'+'---:|'*len(panel)]
    def display(value,task=None):
        if value is not None:return f'{value:.4f}'
        relevant=[s for s in states if task is None or s['task']==task]
        return 'pending' if any(s['outcome']=='pending_or_running' for s in relevant) else 'N/A'
    for task in tasks:lines.append('| '+task+' | '+' | '.join(display(means[r][task],task) for r in panel)+' |')
    lines.append('| Nine-task macro | '+' | '.join(display(macros[r]) for r in panel)+' |')
    lines+=['','All raw run rows and reader/retrieval efficiency measurements are in scores.csv. Build costs and shared evidence preparation times are in build_costs.csv, one row per delivery: do not sum them once per reader. Packing and tokenizer loading are measured separately from reader inference; older runs without these measurements remain blank. Qwen build calls exclude preflight calls before agent_start. Missing usage is left blank, not treated as zero. Builder cached input is a component of input tokens; do not add them. Tokens are observed usage, not a monetary cost estimate.',
            '', 'Different benchmarks use their own registered metrics; the macro is a descriptive mean of those normalized metrics, not one common accuracy. N/A means the registered attempts ended without a complete score set; flagged/failed attempts are retained and never replaced with the best successful score.',
            '', 'Run status:']
    lines += [f"- {x['run_id']}: {x['outcome']}" for x in states]
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'campaign':a.campaign,'completed_rows':len(rows),'planned_rows':len(plan['runs'])*len(panel),'macro':macros}))

if __name__=='__main__':main()
