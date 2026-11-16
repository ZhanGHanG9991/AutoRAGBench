"""Export a completed winning trial by moving its existing on-disk Lance store."""
import json,shutil
from pathlib import Path

def export_winner(record,index_paths,run,reasoning_effort="medium"):
    key=json.dumps(record.config.to_structural().model_dump(mode="json"),sort_keys=True)
    source=Path(index_paths[key]);run=Path(run);dest=run/"serve-index"
    if not source.is_dir():raise FileNotFoundError("winning trial Lance directory missing")
    if dest.exists():raise FileExistsError("serve-index already exported")
    shutil.move(str(source),str(dest))
    payload={"trial_number":record.trial_number,"embedding_model":record.config.embedding_model,
             "index_type":record.config.index_type.value,"runtime":record.config.to_runtime(reasoning_effort=reasoning_effort).model_dump(mode="json"),
             "export_method":"reuse_trial_index_move_no_reembedding"}
    (run/"winning-trial.json").write_text(json.dumps(payload,indent=2))
    return dest
