"""Run patched upstream optimizer and export the winning trial's actual Lance store.
Only runtime hooks are installed; the shared checkout is unchanged.
"""
import argparse,asyncio,json,logging,os,time,signal,subprocess,sys
from pathlib import Path
from exp3lib.autorag_export import export_winner

async def run(a):
    import yaml
    from agentic_autorag.config.models import ProjectConfig
    from agentic_autorag.litellm_runtime import configure_litellm_runtime, resolve_model
    import agentic_autorag.orchestrator as om
    from agentic_autorag.engine.index_builder import IndexBuilder
    prep=Path(os.environ["EXP3_PREP"]);root=Path(a.run)
    template="search_space_large.yaml" if a.bench=="cwq" else "search_space_loose.yaml"
    raw=yaml.safe_load((prep/"configs/autorag"/template).read_text())
    raw["meta"].update(project_name="x9-"+a.bench,corpus_path=os.environ["EXP3_CORPUS"],output_dir=str(root/"out"))
    exam=prep/"splits"/a.bench/"tune.json"
    if a.smoke:
        raw["meta"]["max_trials"]=2
        exam=root/"smoke-tune.json"
        exam.write_text(json.dumps(json.loads((prep/"splits"/a.bench/"tune.json").read_text())[:2]))
    raw["examiner"]["custom_exam_path"]=str(exam)
    raw["agent"]["concurrency"]=1
    # Optimizer and examiner/judge use separate providers.
    assert raw["agent"]["optimizer_model"]=="deepseek/deepseek-flash"
    g61_target=os.environ.get("G61_LITELLM_MODEL","openai/gpt-6.1-sol")
    raw["agent"]["optimizer_model"]=g61_target
    # Preserve the optimizer output cap used by the evaluated build.
    g61_max_tokens=int(os.environ.get("G61_MAX_TOKENS","16384"))
    raw["model_aliases"][g61_target]={"model":g61_target,"api_base":os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),"max_tokens":g61_max_tokens}
    # Only the local platform's non-secret placeholder key is in this object.
    # Provider credentials remain in environment variables.
    raw["model_aliases"]["generator"]={"model":"openai/generator","api_base":os.environ["LLM_BASE_URL"],"api_key":"EMPTY","extra_body":{"chat_template_kwargs":{"enable_thinking":False}}}
    # Endpoint verification resolves this alias once before the runtime resolver.
    raw["model_aliases"]["openai/generator"]=dict(raw["model_aliases"]["generator"])
    models=Path(os.environ["EXP3_MODELS_DIR"])
    for part in ["embedding","reranker"]:
        raw["search_space"][part]["models"]=[str(models/Path(x).name) if x != "none" else x for x in raw["search_space"][part]["models"]]
    raw["model_aliases"]["deepseek/deepseek-flash"]={"model":"deepseek/deepseek-flash","api_base":os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com")}
    config=ProjectConfig.model_validate(raw)
    om.load_config=lambda _:config
    configure_litellm_runtime(config.model_aliases)
    import litellm
    litellm.register_model({g61_target:{"litellm_provider":"openai","mode":"chat","supports_reasoning":True,"max_input_tokens":272000,"max_output_tokens":128000,"input_cost_per_token":0.0,"output_cost_per_token":0.0}})
    g61_resolved,g61_extras=resolve_model(config.agent.optimizer_model)
    assert g61_resolved==g61_target and g61_extras=={"api_base":os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),"max_tokens":g61_max_tokens}, "optimizer routing invariant failed"
    target,extras=resolve_model(config.resolve_alias("generator"))
    assert target=="openai/generator", "local generator model routing invariant failed"
    assert extras.get("api_base")==os.environ["LLM_BASE_URL"], "local generator endpoint routing invariant failed"
    assert extras.get("api_key")=="EMPTY", "local generator placeholder routing invariant failed"
    mapping={};original=IndexBuilder.build
    async def tracked(self,documents,structural,corpus_hash,doc_ids=None):
        index=await original(self,documents,structural,corpus_hash,doc_ids)
        key=json.dumps(structural.model_dump(mode="json"),sort_keys=True)
        mapping[key]=index.vector_store.exp3_lance_path
        (root/"trial-index-map.json").write_text(json.dumps(mapping,indent=2))
        return index
    IndexBuilder.build=tracked
    orch=om.Orchestrator("memory-only-config",debug_eval_samples=0,seed=20261027,skip_final_report=True)
    # Suppress framework score/question log output; train artifacts remain local.
    orch.logger.setLevel(logging.WARNING)
    original_setup=orch.setup
    async def setup():
        await original_setup()
        import json as j
        with Path(os.environ["EXP3_CORPUS"]).open() as fh:
            orch.evaluator.titles={str(d["doc_id"]):d.get("title","") for d in map(j.loads,fh)}
    orch.setup=setup
    def salvage_on_term(signum, frame):
        record=orch.history.get_best()
        if record is None: raise SystemExit(3)
        if not (root/"serve-index").exists():
            export_winner(record,mapping,root,config.search_space.generator.reasoning_effort)
        (root/"optimizer_status.json").write_text(json.dumps({"completed_trials":len(orch.history.records),"winning_trial":record.trial_number,"optimizer_timeout":True,"termination_export":True,"test_evaluation":False,"model":"gpt-6.1-sol"},indent=2))
        if not (root/"submission/start.sh").exists():
            subprocess.run([sys.executable,str(Path(os.environ["EXP3_PREP"])/"jobs/package_submission.py"),"--framework","autorag","--bench",a.bench,"--run",str(root),"--mode","smoke" if a.smoke else "formal"],check=True)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,salvage_on_term)
    timed_out=False
    try:
        await asyncio.wait_for(orch.run(),timeout=min(float(os.environ.get("EXP3_OPTIMIZE_SECONDS","2700" if a.smoke else "19800")),max(30,int(os.environ.get("ARB_DEADLINE",str(int(time.time())+21600)))-time.time()-300)))
    except asyncio.TimeoutError:
        timed_out=True
    record=orch.history.get_best()
    if record is None:raise RuntimeError("no completed trial available for export")
    export_winner(record,mapping,root,config.search_space.generator.reasoning_effort)
    (root/"optimizer_status.json").write_text(json.dumps({"completed_trials":len(orch.history.records),"winning_trial":record.trial_number,"optimizer_timeout":timed_out,"test_evaluation":False,"model":"gpt-6.1-sol"},indent=2))
    print("AutoRAG export complete; completed trials",len(orch.history.records),"winner trial",record.trial_number,flush=True)

def main():
    if not os.environ.get("SLURM_JOB_ID"):raise SystemExit("Slurm allocation required")
    ap=argparse.ArgumentParser();ap.add_argument("--bench",required=True);ap.add_argument("--run",required=True);ap.add_argument("--smoke",action="store_true");a=ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    try:asyncio.run(run(a))
    except Exception as exc:
        # Do not render provider exceptions, which can include request credentials.
        Path(a.run,"build_failure.json").write_text(json.dumps({"component":"autorag","exception_type":type(exc).__name__}))
        raise SystemExit("AutoRAG build failed: "+type(exc).__name__) from None
if __name__=="__main__":main()
