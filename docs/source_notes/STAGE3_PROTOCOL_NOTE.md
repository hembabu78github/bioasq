# Stage 3 protocol note

## Split correction

The provisional Stage 2 split is superseded because the original 80-question
pilot existed before the split. The v2 split pins all pilot duplicate-groups to
development and creates a new untouched test partition.

## Dense models

Two frozen off-the-shelf embedding models are evaluated during development:

- BAAI/bge-small-en-v1.5: efficient general English embedding model.
- NeuML/pubmedbert-base-embeddings: biomedical literature embedding model.

No dense model is fine-tuned in Stage 3.

## Hybrid retrieval

BM25 and dense ranked lists are combined by reciprocal rank fusion. Fusion does
not use gold relevance at retrieval time.

## Risk-adaptive status

The current query-only evidence-risk prior is exploratory. It will not be the
final routing policy unless it predicts retrieval failure or route utility on
development data beyond question type.

The final route policy may use:

- query form and relation/complexity cues;
- BM25 and dense score margins;
- lexical/dense agreement;
- evidence redundancy;
- graph path availability;
- claim-verification conflict;
- estimated latency/resource cost.

No gold label is available at deployment time.
