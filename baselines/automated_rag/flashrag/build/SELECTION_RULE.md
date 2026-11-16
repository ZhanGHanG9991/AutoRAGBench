# Registered selection rule

The immutable candidate order is `service/submission/flashrag/candidates.json`.
Use 200 public train-tune examples, split seed 20261027. The separate 200-example holdout and any development split do not select candidates; test data is never used.
Score source-exact evidence with the common fixed-evidence Qwen3.5-4B reader and the task-native metric. The reader call has its own budget.
Maximize the mean tune metric; ties prefer fewer registered maximum generator calls, then earlier registered order.

The evaluated standard policy stops the entire search at the first retrieval-token overflow. Only fully evaluated preceding candidates can win; partial rows are discarded. Non-token errors retain their failure behavior.
MultiHop-RAG uses the 131,072-token exception: skip only a token/context-overflow candidate, discard its partial rows, and continue through the original order. Select among fully completed candidates with the same tie break.
WebQSP retains the standard stop policy, with pre-send token-budget checks and context-length errors converted to token-budget errors.

There are ten candidates: sequential BM25/e5/BGE, those same three with BGE reranking, three-round iterative BM25, four-call IRCOT BM25, and fixed single-/multi-step adaptive BM25. No learned adaptive classifier or final generated answer is served. Candidate names ending in `_3` retain three retrieval rounds and two query-generation calls.
