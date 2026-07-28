# Stage 7A protocol note

Stage 7A is development-only.

The 24-question sample is selected before answer generation and is balanced by
BioASQ question type and retrieval-uncertainty stratum. The graph-suitable and
control labels are deterministic sampling roles based on question wording; they
are not observed performance labels.

The smoke test compares four conditions using the same generator model and the
frozen Stage 6 verifier:

1. BGE text-only RAG
2. BM25+BGE hybrid text-only RAG
3. Graph-reranked RAG
4. Risk-adaptive agentic GraphRAG

The graph-reranked condition must alter the evidence package on at least some
graph-suitable cases. The sealed test partition is not read.
