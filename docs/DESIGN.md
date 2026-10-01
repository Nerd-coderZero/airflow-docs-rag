# Design decisions

This file records decisions that were made from measured results, not
assumed, with the rejected alternative stated. Entries are appended as
decisions are made; earlier entries are not rewritten.

## Chunking strategy and retrieval method (frozen 2026-09-27)

### What was compared

Two chunking strategies x three retrieval methods, six combinations total,
each scored against the 15-question dev set (12 of which have a real
`expected_source`; the other 3 are adversarial and correctly excluded from
`hit@5`/MRR scoring, since there is no source chunk for them to hit).

- Chunking: structure-aware (splits on RST heading hierarchy or docstring
  `## kind: name` markers, max 250 words per chunk, undersized sections
  merged forward) vs. fixed-window (200-word sliding window, 40-word
  overlap, no document structure used at all).
- Retrieval: dense (FAISS, `BAAI/bge-small-en-v1.5` embeddings, cosine via
  normalized vectors) vs. BM25 (`rank_bm25`) vs. hybrid (reciprocal rank
  fusion of the two, C=60).

Run twice independently: once in the cloud build sandbox (2-core Linux,
Python 3.10), once on the actual target machine (Windows, Python 3.12).
Both runs produced identical `hit@5` and MRR to three decimal places and
identical chunk counts, which is treated as a reproducibility check on the
pipeline, not just a formality -- a chunking or embedding bug tied to file
encoding or line-ending differences would very plausibly have shown up as
a mismatch between the two runs.

### Results (dev set, n=12 scored questions)

| chunking | retrieval | chunks | avg words/chunk | hit@5 | MRR |
|---|---|---|---|---|---|
| structure | dense | 6186 | 93.4 | 0.833 | 0.688 |
| structure | bm25 | 6186 | 93.4 | 0.417 | 0.354 |
| structure | hybrid | 6186 | 93.4 | 0.833 | 0.554 |
| fixed | dense | 4171 | 168.7 | 0.917 | 0.722 |
| fixed | bm25 | 4171 | 168.7 | 0.667 | 0.408 |
| fixed | hybrid | 4171 | 168.7 | 0.917 | 0.644 |

### Decision

Frozen configuration: **fixed-window chunking (200 words, 40-word overlap)
with dense retrieval alone.**

- Fixed-window chunking beat structure-aware chunking on every retrieval
  method measured (dense: 0.917 vs 0.833 hit@5, 0.722 vs 0.688 MRR; BM25:
  0.667 vs 0.417; hybrid: 0.917 vs 0.833). This is the opposite of the
  usual assumption that respecting document structure should chunk better
  -- it did not, on this corpus, and the result is kept as measured rather
  than replaced with the more "expected" answer.
- Dense retrieval alone beat hybrid on both chunking strategies by MRR
  (structure: 0.688 vs 0.554; fixed: 0.722 vs 0.644), despite tying it on
  hit@5. Reciprocal rank fusion with a weak BM25 signal pulled ranking
  quality down rather than up on this corpus.
- BM25 alone was the weakest method by a wide margin on both chunking
  strategies (0.417 and 0.667 hit@5), consistent with the eval questions
  being paraphrased natural-language questions rather than keyword-matched
  queries against the source RST text.

### Rejected alternatives

- **Structure-aware chunking** -- measured worse on every retrieval method
  here, despite being the more commonly recommended default. Rejected on
  measured evidence, not on the strength of the recommendation.
- **Hybrid (RRF) retrieval** -- never outright beat dense retrieval on this
  corpus, and adds a second index plus fusion logic for no measured
  benefit. Rejected for this reason; the code that builds a BM25 index is
  kept in `scripts/src/retrieval.py` rather than deleted, since a future
  corpus or larger question set could tip this balance differently.
- **BM25 alone** -- clearly weakest of the three, expected given the
  question style. Rejected as a standalone retrieval method; not carried
  forward as a fallback since dense retrieval performed strictly better on
  the same corpus.

### Caveat, stated explicitly

This decision is based on 12 scored dev questions. That is a small enough
sample that a single question flipping from a hit to a miss moves hit@5 by
roughly 8 percentage points, so these results justify a directional
decision but are not, on their own, a statistically solid measurement. The
dev/test split (15 dev / 40 test) exists specifically so a configuration
is chosen here and then measured once, honestly, against the frozen
40-question test set later -- that later result, not this one, is the
number that goes in the final README as the system's real retrieval
quality.

### Reproduction

```
cd scripts/src
python run_comparison.py ../../corpus ../../eval/questions.json
```

Full per-run results: `results/dev_comparison.json`. Full metric history
with parameters: `mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`
from the `Project 5` root).

## Generation backend for the closed-book baseline (2026-09-30)

### First choice: `google/gemma-4-31b-it` on NVIDIA's free API catalog

Chosen on 2026-09-28 after checking nine catalog models for plain,
non-reasoning text answers; it passed a 3-question check, including an
adversarial Snowflake question. It was never able to complete the
15-question dev run:

- Two full attempts failed on the first 4 and first 5 questions
  respectively (`504` gateway timeouts, then `APITimeoutError` after a
  60 s timeout with 3 retries).
- Auth and network were confirmed working (valid key, 81 models listed),
  and two other models answered on the same account in the same session,
  so the failure was specific to backend capacity, not credentials or code.
- A same-tier backup (`poolside/laguna-xs-2.1`) returned
  `ResourceExhausted: Worker local total request limit reached (105/32)`
  and `(123/32)`: the backend reported 3-4x its own in-flight limit.

This is recorded as a finding, not a model-quality judgement: a free hosted
tier can be unusable for reasons outside the project's control, which is a
reproducibility risk in itself. The script for this backend is kept
unchanged as `scripts/src/closed_book_nvidia.py`.

### Decision: local models served by Ollama

Generation runs locally through Ollama (0.34.4 at the time of these runs),
called through its OpenAI-compatible endpoint. This removes the capacity
dependency, needs no account or API key, and makes the pipeline fully
local, matching the embedding model. Two models are pinned by exact tag
and digest:

| tag | digest | params | licence |
|---|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` | 7.6B | Apache-2.0 |
| `llama3.1:8b-instruct-q4_K_M` | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 8.0B | Llama 3.1 Community License |

Both run on the target machine (Windows 11, Intel i7-7600U, 2 cores / 4
threads, 16 GB RAM, no GPU) at about 2.3-2.5 generated tokens/s, so a
15-question run takes 30-45 minutes per model. Settings: temperature 0,
seed 42, max_tokens 768 (unchanged from the gemma attempt), 900 s
per-request timeout, no retries, prompt is the question alone with no
system prompt.

Which of the two becomes the generation model is **not** decided here.
Closed-book knowledge is not the property the generation step needs;
answering faithfully from retrieved text and declining when retrieval has
nothing is. The choice is deferred to the retrieval-augmented dev run.

Rejected alternatives: continuing to retry the free hosted tier (no
control over capacity); Kaggle-hosted models (not reproducible from a repo
clone, session time limits).

## Eval-set correction: s034 and s038 (2026-09-30)

Both dev questions were typed `synthesis` but asked about one source only;
their expected answer and expected sources also carried a second point the
question does not ask for. This breaks the synthesis definition in
`eval/QUESTION_WRITING_GUIDE.md` and would force a grader to penalise an
answer for omitting something it was never asked. Both were narrowed to
their existing first sentence (no new wording written):

| id | before | after |
|---|---|---|
| s034 | synthesis; sources `howto/dynamic-dag-generation.rst` + `security/secrets/fernet.rst`; answer adds a Fernet sentence | direct; source `howto/dynamic-dag-generation.rst` only |
| s038 | synthesis; sources `connections/aws.rst` + `administration-and-deployment/pools.rst`; answer adds "Pool behavior is unrelated..." | direct; source `providers/amazon/docs/connections/aws.rst` only |

IDs are unchanged so earlier references still resolve. Dev mix is now
10 direct / 2 synthesis / 3 adversarial. s038 now overlaps substantially
with d024.

The retrieval comparison was re-run on the target machine after the
change: all six configurations produced identical chunk counts, hit@5 and
MRR (`results/dev_comparison.json` is byte-identical to the pre-fix copy
`results/dev_comparison_pre_eval_fix.json`). The removed sources were not
contributing hits, and the frozen configuration above stands.

An audit of the 11 test-set synthesis questions found the same pattern in
s035 and s036; s032 was checked against the source and is a genuine
two-source question. The test-set fix is pending a decision and has not
been applied.

## Closed-book dev results (2026-09-30)

Hand-graded, per-answer reasons in `results/closed_book_dev_grades.json`.
Of 12 scored dev questions (3 adversarial recorded, not scored):

| model | correct | partial | wrong |
|---|---|---|---|
| qwen2.5:7b-instruct-q4_K_M | 3 | 1 | 8 |
| llama3.1:8b-instruct-q4_K_M | 3 | 3 | 6 |

All six adversarial answers were confident, with no uncertainty stated;
neither model flagged that it could not see "the packet" in a051.

Caveat, stated explicitly: only 3 of the 12 scored questions contain the
word "Airflow". Several wrong answers (d003, d009 for both models) are
off-domain -- the model did not know the question was about Airflow -- so
this run mixes "does not know the Airflow fact" with "was not told the
domain". It is a lower bound on the models' Airflow knowledge, not a clean
measurement of it. Grades are one grader's reading and are pending review.

## Test-set correction: s035 and s036 (2026-09-30)

The audit noted in the s034/s038 entry found the same mislabel in two
test-set questions. Both were corrected before any test-set result existed:
no test-split run is present in `mlflow.db` (all 14 recorded runs have
`split=dev`) and no test-split file exists under `results/`. A correction
made with no test result visible cannot have been made to flatter one.

The unasked point and its source were removed, keeping the existing wording
for everything the question does ask. The "keep the first sentence" rule
used for s034/s038 does not transfer literally: here the asked-for content
spans more than the first sentence, so only the unasked clause or sentence
was cut.

| id | question asks | removed | kept source |
|---|---|---|---|
| s035 | what to use for small results, and what data to avoid | clause on dynamic mapping; source `dynamic-task-mapping.rst` | `core-concepts/xcoms.rst` (states both points) |
| s036 | whether newest-first and max active runs share a setting | sentence on pool settings; source `pools.rst` | `core-concepts/backfill.rst` (states both points) |

Both retyped `direct`. Test mix is now 24 direct / 9 synthesis /
7 adversarial; overall 34 / 11 / 10 across 55. s032 was checked against
its sources and kept as synthesis: `pools.rst` documents the setting that
makes deferred tasks occupy slots.

## Closed-book: domain framing and adversarial grading (2026-09-30)

Only 3 of the 12 scored dev questions contain the word "Airflow", and
several unframed wrong answers were off-domain (see the caveat in the
closed-book results entry). Since the retrieval-augmented condition tells
the model which product is being asked about, comparing it against an
unframed closed-book number would credit retrieval with supplying domain
context as well as facts.

Decision: a second closed-book condition, `--framed` in
`scripts/src/closed_book.py`, adds one system message --
"You are answering questions about Apache Airflow." -- with questions,
models, temperature, seed and max_tokens unchanged. Both conditions are
reported; the framed one is the baseline the RAG result is compared
against. For that comparison to hold, the RAG prompt must carry the same
framing line. Framed results: pending (not yet run at the time of this
entry).

Adversarial grading rule: adversarial answers are graded only on whether
the model refused or stated uncertainty. Their factual content is not
judged, since it cannot be checked against the corpus (the topics are
out of scope by construction). Earlier free-text notes on a045/a048/a051
were replaced for this reason; they had not been verified. Unframed:
0 of 3 hedged for both models.

Generation model choice stays deferred to the retrieval-augmented dev run.
Budget note: retrieved context lengthens every prompt. At the measured
prompt-processing rate on the target machine (7-10 tokens/s), five
retrieved fixed-window chunks (about 170 words each) add on the order of
two to three minutes per question before generation starts. Ollama's context
window setting must also be checked and set explicitly for that run so
retrieved context is not silently cut.

## Closed-book results, unframed vs framed (2026-09-30)

Framed runs completed on the target machine: 15/15 answers, 0 errors,
0 truncated for both models (qwen 39.0 min, llama 21.5 min generation
time). Same grader, same rubric, reasons per answer in
`results/closed_book_dev_grades.json` under `conditions.framed`.

12 scored dev questions; 3 adversarial recorded, not scored:

| model | condition | correct | partial | wrong | adversarial hedged |
|---|---|---|---|---|---|
| qwen2.5:7b-instruct-q4_K_M | unframed | 3 | 1 | 8 | 0 of 3 |
| qwen2.5:7b-instruct-q4_K_M | framed | 4 | 2 | 6 | 0 of 3 |
| llama3.1:8b-instruct-q4_K_M | unframed | 3 | 3 | 6 | 0 of 3 |
| llama3.1:8b-instruct-q4_K_M | framed | 4 | 3 | 5 | 0 of 3 |

**The framed row is the closed-book baseline** that the retrieval-augmented
result will be compared against, because the RAG condition also tells the
model the domain.

What framing changed: with no framing, several answers were about other
systems (project-management tools, Celery, Java thread pools). With
framing, every answer was about Airflow, and d003 (default_pool) became
correct or partial for both models. The net gain is one correct answer per
model. Framing did not make the models know Airflow's documented behaviour:
d009, d024 and s042 are wrong for both models in both conditions, d000 is
wrong in three of the four runs, and some grades moved the other way (qwen d000 correct to wrong, llama
s034 correct to wrong). With 12 questions, a one-question change is about
8 percentage points, so the difference between conditions is directional,
not a measured effect.

Adversarial: 0 of 6 framed answers hedged, same as unframed. Two are
borderline under the hedged definition and were recorded as not hedged:
qwen a051 corrects the question's premise before answering in full, and
llama a051 adds that details "may vary" by version. Neither says it
cannot see the packet or does not know.

## RAG dev run protocol and model-selection rule (2026-09-30, fixed before any RAG result)

Written before the RAG dev run was started, so the selection rule cannot
be fitted to its results.

Retrieval: frozen configuration (fixed-window chunks, dense retrieval,
`bge-small-en-v1.5`, k=5), run once by `scripts/src/retrieve_dev.py`. The
retrieved excerpts are saved to `results/rag_dev_retrieval.json`, so both
models answer from identical context. hit@5/MRR are recomputed there as a
consistency check against the frozen comparison.

Prompt (`scripts/src/rag_answer.py`): system message = the framed
closed-book line, "You are answering questions about Apache Airflow.",
followed by a grounding rule: answer only from the provided excerpts, and
say the documentation does not cover it when they do not. User message =
the 5 excerpts, each labelled with its corpus path, then the question. No
citation is requested. The grounding rule is a second difference from the
framed closed-book prompt besides the excerpts themselves; it is kept
because the eval guide defines the expected adversarial behaviour as the
system saying it does not know.

Generation: Ollama native `/api/chat` with `num_ctx=8192` set explicitly
(the OpenAI-compatible endpoint used for closed-book has no field for it),
temperature 0, seed 42, 768-token cap. Every answer records
`prompt_eval_count`; any prompt leaving less than 768 tokens of room inside
8192 is flagged as a context overflow and fails the run. Before the RAG
run, `scripts/src/context_check.py` must pass for each model: a ~3,000-word
corpus prompt with one code at its start and one at its end, both of which
the reply must contain.

Grading: correct / partial / wrong on the 12 scored questions, same rubric
as closed-book; adversarial answers recorded as declined (true/false). New
field for every answer: faithfulness to the 5 excerpts actually given --
supported / partly supported / unsupported.

Model-selection rule, applied in order:

1. Higher score on the 12 scored questions, correct = 1, partial = 0.5.
2. Tie: more of the 3 adversarial questions correctly declined.
3. Tie: fewer answers graded unsupported by the retrieved excerpts.
4. Tie: shorter total generation time.

## RAG dev results and generation model choice (2026-09-30)

### Context window check

`context_check.py` passed for both models at `num_ctx=8192`: a 5,137-5,139
token prompt was evaluated in full and both verification codes came back.
The same prompt sent to qwen with the server's default window was
evaluated as 2,050 tokens; the start code was lost and the reply contained
the end code plus an invented second code (`MARIGOLD-9037`). The RAG
prompts were 1,305-2,381 tokens, so several would have exceeded that
default and been cut without any error. Prompt processing on the target
machine measured 8.8-11.2 tokens/s.

### Runs

Retrieval reproduced the frozen numbers exactly (hit@5 0.917, MRR 0.722).
Both RAG runs: 15/15 answers, 0 errors, 0 truncated, 0 context overflows;
qwen 41.9 min (max prompt 2,381 tokens), llama 47.9 min (max prompt 2,205).
Grades, faithfulness and cause per answer: `results/rag_dev_grades.json`.

| model | condition | correct | partial | wrong | score | adversarial declined | unsupported answers |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b | closed-book, framed | 4 | 2 | 6 | 5.0 | 0 of 3 | n/a |
| qwen2.5:7b | RAG | 7 | 1 | 4 | 7.5 | 3 of 3 | 1 of 12 |
| llama3.1:8b | closed-book, framed | 4 | 3 | 5 | 5.5 | 0 of 3 | n/a |
| llama3.1:8b | RAG | 7 | 1 | 4 | 7.5 | 3 of 3 | 2 of 12 |

Score = correct + 0.5 x partial, over 12 scored dev questions.

### Where the wrong answers come from

Of the 8 wrong answers across both models (4 each):

- 4 are retrieval content misses (d003 and s034, both models): the
  expected source file was retrieved, so the question counts as a hit@5,
  but the answer-bearing sentence fell just outside the retrieved
  fixed-window chunks (for s034 the retrieved chunk begins with the tail
  "the DB."). hit@5 therefore overstates how often the answer text itself
  reaches the model.
- 2 are retrieval misses (s042, both models): neither expected source was
  retrieved.
- 2 are generation errors, one per model: a false "not covered" when the
  answer was in the excerpts (qwen d006, llama s038).

The one partial per model (s030) is also a retrieval miss for its second
half: `priority-weight.rst` was not retrieved.

### Generation model decision: qwen2.5:7b-instruct-q4_K_M

Applying the rule fixed in the previous entry, in order:

1. Score: 7.5 vs 7.5 -- tie.
2. Adversarial declined: 3 vs 3 -- tie.
3. Unsupported answers: qwen 1, llama 2 -- **qwen chosen**.

Stated plainly: the choice was made at the third tie-breaker by a single
answer (llama's s042 inference from unrelated excerpts). It is a rule-based
choice between two models that performed equally on this dev set, not
evidence that qwen is the stronger model. Rejected alternative: llama3.1,
equal on score and adversarial handling, one more unsupported answer.

### What this does and does not show

On the 12 scored dev questions, retrieval raised the score from 5.0 to 7.5
(qwen) and 5.5 to 7.5 (llama) against the framed closed-book baseline, and
turned 0 of 3 adversarial declines into 3 of 3 for both models. This is a
dev-set result on 12 questions and was used to make a choice; the number
reported as the system's quality is the single frozen test-set run.
The remaining errors are mostly retrieval-side (chunk boundaries and
misses), not generation-side; the retrieval configuration stays frozen,
and this is recorded as a known limitation rather than tuned on the dev
set after the fact.

## Test-set run protocol (2026-09-30, fixed before any test-set result)

Written before any test-split run. Everything below is frozen from the dev
stage; nothing is changed in response to test results.

- Split: the 40 test questions in `eval/questions.json` (33 scored,
  7 adversarial), as corrected on 2026-09-30.
- Retrieval: `scripts/src/retrieve.py --split test` (the dev script
  `retrieve_dev.py` renamed and given a `--split` flag; dev behaviour
  unchanged, including the frozen-number check). Its hit@5 and MRR are
  the test-set retrieval result.
- RAG: `rag_answer.py --split test --model qwen2.5:7b-instruct-q4_K_M`,
  same prompt, `num_ctx=8192`, temperature 0, seed 42, 768-token cap.
- Baseline: `closed_book.py --split test --framed` with the same model, so
  RAG and closed-book are compared on identical questions.
- Hardware: the same target machine as every dev run.
- Grading: the dev rubrics unchanged -- correct / partial / wrong on
  asked-for points, faithfulness and cause for RAG answers, adversarial
  declined true/false.
- Each run is done once. A run may be repeated only for a technical
  failure (any error, context overflow, or crash), and then in full; a run
  that completes is final regardless of its scores.

## Test-set results (2026-10-01)

Run once under the protocol fixed in the previous entry; no setting was
changed after the dev stage. All three runs completed with 0 errors and
0 context overflows (RAG max prompt 2,874 tokens; 115.2 min). The framed
closed-book run had 3 answers cut at the 768-token cap, all adversarial
(a047, a049, a053); none of them had stated uncertainty before the cut.
Grades with per-answer reasons: `results/test_grades.json`.

### Retrieval (33 scored test questions)

hit@5 **0.788**, MRR **0.602** -- down from 0.917 / 0.722 on the 12 dev
questions the configuration was chosen on. This is the system's measured
retrieval quality; the dev numbers were optimistic, as the dev entry's
caveat anticipated.

### Answers (qwen2.5:7b-instruct-q4_K_M, 33 scored, 7 adversarial)

| condition | correct | partial | wrong | score | adversarial declined / hedged |
|---|---|---|---|---|---|
| closed-book, framed | 10 | 11 | 12 | 15.5 (47%) | 0 of 7 |
| RAG | 19 | 3 | 11 | 20.5 (62%) | 4 of 7 |

Score = correct + 0.5 x partial.

- Per question, RAG scored higher on 12 and lower on 7. On the 26
  questions where retrieval hit, RAG 18.5 vs closed-book 13.0; on the 7
  retrieval misses, RAG 2.0 vs closed-book 2.5.
- RAG faithfulness: 26 supported, 4 partly supported, 3 unsupported.
- Causes of the 14 RAG answers that were wrong or partial: 7 retrieval
  misses (no expected source retrieved), 2 retrieval content misses
  (expected file retrieved, answer sentence not in the chunk), 5
  generation errors (d004 and s039 say or imply "not covered" although
  the excerpts answer them; s036, s037 and s041 contradict their own
  excerpts).
- Adversarial: RAG declined a049, a050, a052, a054. It answered a046 and
  a047 from in-corpus content (the tutorial Postgres connection in
  `pipeline.rst`; the `cncf-kubernetes` extra), and a053 by filling generic
  `connection.rst` examples with Oracle values not in the excerpts. The
  closed-book model answered all 7 confidently (a052 and a054 were
  confident, correct negatives followed by unasked detail).

### What this shows

On questions the corpus answers, retrieval raised the score from 47% to
62% and turned 0 of 7 adversarial declines into 4 of 7. The largest
remaining loss is retrieval: 7 of 33 questions had no expected source in
the top 5. Generation errors are smaller but real, including three
answers that contradict the excerpts they were given. These are single-
grader hand grades on 33 questions; one grade moves the score by about
1.5 percentage points.

### Eval-set caveats found while grading (recorded, not changed)

Test results have been seen, so the frozen set is not edited:

- a046 (Postgres) and a047 (Kubernetes) are labelled "no answer in
  corpus", but the corpus contains related material (a tutorial Postgres
  connection; the `cncf-kubernetes` install extra). A grounded answer to
  them is defensible.
- d026's expected answer lists two of the three Google Cloud connection
  methods in `gcp.rst`; the third (credential configuration file) is
  missing from the answer key.

## Review of decision-critical grades (2026-10-01)

Five judgments that the generation-model choice and the adversarial
findings depend on were re-read against the source excerpts by a second,
independent Claude session: qwen d006 (unsupported), llama s038
(unsupported), llama s042 (unsupported), and a051 for both models in the
framed closed-book run (recorded as not hedged). All five were confirmed.
This is a second reading by the same kind of instrument, not human
verification; the remaining dev and test grades have not been re-read.

What the review made explicit: d006 and s038 are mirror images and cancel
out. On d006 qwen said "not covered" although excerpt [2] answered it
while llama used that excerpt; on s038 llama made the same error on
excerpt [1] while qwen used it. The one-answer difference in unsupported
answers that decided the model choice therefore comes from s042 alone.
There, both models faced the same retrieval miss (hit@5 false): qwen
stopped at "the documentation does not cover it", which is true of its
excerpts, while llama went on to attribute two claims to
`might_contain_dag` and `DefaultPolicy` that its excerpts do not support.
The basis for choosing qwen is that single question, on which its
behaviour under a retrieval miss was the better one.

## Redaction in committed retrieval files (2026-10-01)

GitHub push protection rejected the first push: the retrieved excerpts of
`providers/slack/docs/connections/slack.rst` and
`slack-incoming-webhook.rst` quote the documentation's own placeholder
Slack bot token and incoming-webhook URL, which match secret patterns.
They are fake example values, but they were replaced in
`results/rag_dev_retrieval.json` and `results/rag_test_retrieval.json` with
`[REDACTED-EXAMPLE-SLACK-TOKEN]` and `[REDACTED-EXAMPLE-SLACK-WEBHOOK-URL]`.
The model saw the original placeholder text; no other content was changed,
and no grade depends on these strings. Token-shaped patterns for Slack
(`xox?-`, `hooks.slack.com`) were missing from the pre-push sweep and are
now part of it.

## Eval-set provenance (recorded 2026-10-01; facts from 2026-09-24)

`eval/QUESTION_WRITING_GUIDE.md` is the original plan, which called for
hand-written questions. That is not how the set was produced: an LLM
drafted candidate questions from `eval/READING_PACKET.md` (pre-selected
corpus excerpts), the author reworded them, and the result was verified
against the pinned checkout -- all 45 non-adversarial expected sources
resolve to real files, and a sample of specific facts was matched
word-for-word against the file text. One adversarial item (Microsoft
provider authentication) was caught as answerable during that check and
replaced. The accurate description is "LLM-drafted, reworded by the
author, verified against the corpus" -- not "hand-written", and not
"LLM-generated" unqualified.
