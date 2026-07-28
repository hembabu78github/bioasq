# Stage 6 protocol decisions

## Development boundary

All adversarial claims are derived from Stage 5 development claims and displayed
development evidence. The sealed test set is not accessed.

## Why generate two variants per supported source claim?

- The contradiction variant supplies directly refutable claims that were absent
  from the original human audit.
- The insufficient-evidence variant supplies plausible inferential leaps and
  unsupported qualifiers, which were the dominant Stage 5 verifier weakness.

## Why human-label generated adversarial claims?

The generator's intended label is not a gold label. Generation can fail to
produce a genuine contradiction or may accidentally create a supported claim.
Two independent human labels are therefore required before evaluation.

## Why batch by question?

All claims sharing the same question/evidence package are verified in one call.
This reduces the verifier comparison from 228 claim-level calls to approximately
36 grouped calls.

## Why not select by accuracy alone?

The Stage 5 verifier achieved high overall accuracy while missing three of four
insufficient-evidence claims. The selection rule therefore prioritizes
insufficient-evidence recall and false-support rate.
