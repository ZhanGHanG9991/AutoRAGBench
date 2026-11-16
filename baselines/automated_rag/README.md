# Automated RAG baselines

[Agentic AutoRAG](agentic_autorag/) optimizes a parameterized retrieval pipeline with a reasoning model; [FlashRAG](flashrag/) searches a registered set of retrieval pipelines on public training examples.
Both receive the same six-hour (21,600-second) build budget as the agents, including index construction and configuration search. Frozen services return corpus evidence through `POST /retrieve`; Qwen3.5-4B is the only retrieval generator at evaluation.
The method READMEs describe task coverage, the failed FlashRAG HotpotQA build, and the MultiHop-RAG budget exception. Model weights, indexes, training examples, and evaluation outputs are not bundled.
