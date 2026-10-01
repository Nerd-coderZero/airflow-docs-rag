# Project 5 — writing the 55 eval questions

Plain-language working sheet. Keep this open while you write.

## What you are actually doing

You are setting an exam for a new hire who has the Airflow manual on their
desk and nothing else. You write the questions, and you write the answer key.

Later, your RAG pipeline sits the exam. The score tells you whether your
retrieval choices (chunking, dense vs BM25 vs hybrid) actually help, instead
of you asserting that they do.

## The four fields per question

    id        q01, q02, ...
    question  written the way a real person would type it
    type      A, B, or C
    answer    one or two sentences, in your own words
    sources   the corpus file(s) the answer comes from (empty for type C)

That is it. No scoring rubric, no difficulty rating. Four fields.

## The three types

### Type A — answer is on one page (write 25)

Tests: does it find the right page at all.

Worked examples (demonstrations only, do not use these):

    q: What does setting catchup=False on a DAG do?
    type: A
    answer: It stops the scheduler from creating runs for the intervals
            between the start date and now when the DAG is first enabled.
            Only the current interval onwards runs.

    q: Which Airflow component is responsible for parsing DAG files?
    type: A
    answer: The DAG processor. In older setups this ran inside the
            scheduler; it can also run as its own component.

### Type B — answer needs two pages (write 20)

Tests: does it find both pieces, or find one and confidently answer half.

The trick to writing these: ask a question with two halves, where the halves
live in different parts of the docs.

Worked examples (demonstrations only, do not use these):

    q: A DAG has catchup=True and a start date six months ago. When I unpause
       it, what does the scheduler create, and which setting limits how many
       of those run at the same time?
    type: B
    answer: It creates one run per missed interval since the start date.
            max_active_runs on the DAG caps how many run concurrently.
    (catchup is in the scheduling docs; max_active_runs is in the DAG
     parameters docs — neither page answers the whole question)

    q: How do I pass a value from one task to the next, and what size limit
       applies to whatever mechanism you name?
    type: B
    answer: XComs. The practical size limit comes from the metadata database
            column the XCom backend writes into, so large data should be
            written to external storage with only a reference passed.

### Type C — answer is NOT in the corpus (write 10)

Tests: does it admit it doesn't know, or invent an answer.

This is the most valuable type and the easiest to skip. Write them last,
they go fast.

The strongest ones target the 71 providers you deliberately excluded from
the corpus, because a retriever will happily hand back the Amazon or Google
provider docs as a near match and sound confident doing it.

Worked examples (demonstrations only, do not use these):

    q: Which Snowflake provider hook supports key-pair authentication?
    type: C
    answer: Not answerable from this corpus. The Snowflake provider is not
            among the five providers extracted.
    sources: (none)

    q: What is the task concurrency limit on AWS MWAA?
    type: C
    answer: Not answerable from this corpus. MWAA is a managed service; its
            limits are not documented in the Airflow repository.
    sources: (none)

Good sources of type C questions:
  - the 71 excluded providers (Snowflake, Postgres, HTTP, Salesforce, ...)
  - managed services (MWAA, Cloud Composer, Astronomer) and their limits
  - pricing, support, commercial terms
  - roadmap / future-version questions
  - anything that would live in a GitHub issue or discussion (deliberately
    excluded from the corpus)

## How to produce them without staring at a blank page

Do not try to invent 55 questions from nothing. Work file by file.

1. Open one doc file from the corpus. Read it for two minutes.
2. Write 2 type A questions it answers on its own.
3. Write 1 type B question that needs this file plus another one you
   remember seeing.
4. Next file.

Three questions per file. Fifteen files gets you 45. Then do the 10 type C
questions in one sitting at the end — they need no reading, just a list of
things you know are not in the corpus.

## The one rule that makes or breaks the set

**Write the question the way a user would ask it, THEN go find the answer.**

Do not read a sentence in the docs and rephrase it into a question. If you
do, the question and the document end up sharing wording, both BM25 and
dense retrieval score it far too easily, and your results look good for a
reason that has nothing to do with your pipeline.

A real user asks "why is my DAG not running" — they do not ask "what is the
effect of the catchup parameter on DagRun creation."

## Sequencing

- All 55 exist before any retrieval result is scored.
- The dev/test split (15 / 40) is assigned randomly once all 55 exist, not
  by picking which ones feel like good tuning questions.
- The 40-question test set is scored exactly once, after every choice about
  chunking, embedding and retrieval is fixed.
- Building the pipeline (chunking, FAISS, BM25, hybrid) does not need any
  questions and can proceed in parallel right now.
