# Writing the eval question set

55 questions total, hand-written against the actual corpus content (not
LLM-generated). 15 held out as a dev set for tuning chunking, embedding and
retrieval choices. 40 frozen as a test set, scored exactly once after every
pipeline choice is fixed -- this mirrors Project 4's validation-week /
evaluation-week split.

## Format, one entry per question

```json
{
  "id": "core-001",
  "question": "...",
  "expected_source": "airflow-core/docs/core-concepts/xcoms.rst",
  "expected_answer_or_criterion": "...",
  "type": "direct | synthesis | adversarial",
  "corpus_area": "core | provider:amazon | provider:google | provider:microsoft | provider:databricks | provider:slack",
  "split": "dev | test"
}
```

- **direct**: answer is stated plainly in one section of one file.
- **synthesis**: answer requires combining two or more sections/files
  (e.g. a scheduling behaviour described in core-concepts, qualified by a
  caveat in administration-and-deployment).
- **adversarial**: the corpus genuinely does not answer this (a plausible-
  sounding question about something out of scope, or about a provider not
  in the 5-provider sample). The expected behaviour is the system says it
  doesn't know, not that it guesses.

## Suggested mix across 55

- ~30 direct, mostly core (this is the bulk, checks basic retrieval works)
- ~15 synthesis (checks retrieval finds multiple relevant chunks, and
  generation actually combines them rather than just answering from the
  first hit)
- ~10 adversarial (checks the system doesn't hallucinate when the corpus
  has no answer -- this is the finding that a real RAG project needs to
  show, not just hide)

## Areas seeded from the actual corpus, to write against directly

Core -- core-concepts: xcoms.rst, taskflow.rst, sensors.rst, dag-run.rst,
dynamic-task-mapping (authoring-and-scheduling), assets.rst, deferring.rst,
params.rst, variables.rst, backfill.rst.

Core -- administration-and-deployment: scheduler.rst, pools.rst,
priority-weight.rst, cluster-policies.rst, plugins.rst, kubernetes.rst,
dag-serialization.rst.

Core -- security: fernet.rst, jwt_token_authentication.rst,
local-filesystem-secrets-backend.rst, security_model.rst,
mask-sensitive-values.rst.

Core -- howto: custom-operator.rst, dynamic-dag-generation.rst,
setup-and-teardown.rst, notifications.rst, usage-cli.rst.

Providers -- pick 2-3 questions per provider (amazon, google, microsoft,
databricks, slack) about something specific in that provider's docs: a
named operator, hook, connection type or a documented gotcha, not a
generic "what does this provider do."

Adversarial candidates: ask about a provider NOT in the 5-sample (this
tests whether the system correctly says "not in my corpus" rather than
hallucinating from training knowledge); ask about an Airflow version or
feature that doesn't appear in this corpus; ask something plausible-
sounding but fabricated (a config option that doesn't exist).

## Two sittings

Sitting 1 (~20 questions): work through core-concepts, authoring-and-
scheduling, administration-and-deployment. This covers the bulk of direct
and synthesis questions.

Sitting 2 (~20-35 questions): security, howto, the 5 providers, and the
adversarial set. Finish by splitting the full set into dev (15) and test
(40) -- pick the dev 15 to span all three types and both core/provider
areas, so tuning decisions aren't blind to any question shape.

Write into `eval/questions.json` as a single JSON array of the objects
above.
