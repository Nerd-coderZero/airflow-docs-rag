# Running the chunking / retrieval comparison locally

This reproduces the chunking-strategy and retrieval-method comparison on
your own machine, against the same pinned corpus and eval set already in
your Project 5 folder.

## Where these files go

Place this `scripts/` folder directly inside `Project 5`, so the layout is:

```
Project 5/
  corpus/           (already there)
  eval/
    questions.json  (already there)
  scripts/
    requirements.txt
    src/
      chunking.py
      retrieval.py
      run_comparison.py
```

## Setup

From inside `Project 5/scripts/`:

```
python -m venv venv
venv\Scripts\activate          (Windows)
pip install -r requirements.txt
```

## Running it

```
cd src
python run_comparison.py ../../corpus ../../eval/questions.json
```

This will:

1. Build both chunk sets (structure-aware and fixed-window) from the real
   corpus files.
2. Download the embedding model `BAAI/bge-small-en-v1.5` the first time
   (about 130MB, cached afterward -- no re-download on later runs).
3. Embed every chunk, build a FAISS index and a BM25 index for each
   chunking strategy.
4. Score dense, BM25 and hybrid retrieval against the 15-question dev set
   in `eval/questions.json` (hit@5 and MRR).
5. Log all 6 runs to a local MLflow store at `Project 5/mlflow.db`, and
   write a summary table to `Project 5/results/dev_comparison.json`.

## What to expect

This is CPU-bound work -- embedding several thousand real text chunks with
a transformer model. Expect it to take somewhere between a few minutes and
around 20 minutes depending on your machine's core count; it took about
20 minutes in a 2-core cloud sandbox. More cores should be noticeably
faster. There is no GPU requirement for this step.

## Viewing the MLflow results afterward

```
cd ..
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the printed local URL in a browser to see all 6 runs compared
side by side, with their parameters and metrics.

## Reference: results from the cloud sandbox run

For comparison once your own run finishes. If your numbers differ
noticeably, that is worth investigating (a different corpus content,
different code, or something specific to CPU/library versions) rather
than assumed away.

| run | chunks | avg words | hit@5 | MRR |
|---|---|---|---|---|
| structure-dense | 6186 | 93.4 | 0.833 | 0.688 |
| structure-bm25 | 6186 | 93.4 | 0.417 | 0.354 |
| structure-hybrid | 6186 | 93.4 | 0.833 | 0.554 |
| fixed-dense | 4171 | 168.7 | 0.917 | 0.722 |
| fixed-bm25 | 4171 | 168.7 | 0.667 | 0.408 |
| fixed-hybrid | 4171 | 168.7 | 0.917 | 0.644 |

Scored against the 12 dev-set questions with a real expected source (the
other 3 dev questions are adversarial and are correctly excluded from
hit@k/MRR scoring, since there is no source chunk for them to hit).
