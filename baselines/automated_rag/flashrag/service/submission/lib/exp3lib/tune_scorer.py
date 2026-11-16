"""The fixed-evidence reader contract and task-native metric; tune only."""
from functools import lru_cache
from exp3lib.scoring import load_task_hooks,qwen_tokenizer,primary_of,gold_row
from exp3lib.passages import build_passages
from exp3lib.budget import CallBudget
from exp3lib.platform_chat import platform_complete
@lru_cache(maxsize=1)
def tokenizer(): return qwen_tokenizer()
def score_hits(row,hits,corpus,task):
    from multireader.contract import pack_for_readers,messages,clean_answer
    hook=load_task_hooks(task);tok=tokenizer()
    passages=build_passages(hits,corpus,row.get("meta") or {})
    packed=pack_for_readers(row["question"],passages,hook["answer_format"],tok,[tok])
    prompt=messages(row["question"],packed,hook["answer_format"])[0]["content"]
    answer=clean_answer(platform_complete(prompt,CallBudget(max_calls=1,max_tokens=32000,max_seconds=60),max_tokens=512))
    gold=gold_row(row["id"],row["canonical_answer"],row.get("answer_variants") or [])
    return primary_of(hook["score_one"],answer,gold,hook["primary_name"])
