# Corpus source and licence

## Source

[Apache Airflow](https://github.com/apache/airflow), pinned to commit
`2f846065252579d4ef8729ba8e42d96c5ab1b611` on `main`, checked out and
extracted 2026-09-20.

## Licence

Apache License, Version 2.0. Confirmed directly from the `LICENSE` file at
the pinned commit (copy kept at `docs/AIRFLOW_LICENSE.txt`), not assumed
from the repository's public label. The `NOTICE` file (`docs/AIRFLOW_NOTICE.txt`)
is included as required by the licence's attribution terms.

One subcomponent exception noted in the LICENSE file: two bundled fonts
(JetBrains Mono, Plus Jakarta Sans) are under the SIL Open Font License 1.1.
Neither is part of this corpus -- nothing here touches font assets.

## What is included, and why

| Category | Source path | Files | Words |
|---|---|---|---|
| Core docs | `airflow-core/docs/` | 155 | 226,893 |
| Core newsfragments (changelog) | `airflow-core/newsfragments/` | 33 | 2,620 |
| Core docstrings | `airflow-core/src/` (AST-extracted) | 633 | 89,475 |
| Provider docs: amazon | `providers/amazon/` | 116 | 82,961 |
| Provider docs: google | `providers/google/` | 118 | 97,508 |
| Provider docs: microsoft | `providers/microsoft/` | 78 | 42,420 |
| Provider docs: databricks | `providers/databricks/` | 23 | 20,609 |
| Provider docs: slack | `providers/slack/` | 17 | 8,172 |
| **Total** | | **1,173** | **570,658** |

Full per-file provenance (source path, destination path, word count, SHA-256)
is in `corpus/manifest.json`. Every file can be traced back to its exact path
in the pinned commit.

Airflow uses towncrier for its changelog: there is no single CHANGELOG file
in `airflow-core`; instead each pull request adds a short fragment under
`newsfragments/`, aggregated into release notes at release time. Those
fragments are indexed as-is.

Core docs word count above (226,893) is the actual content under
`airflow-core/docs/`. An earlier exploratory count of 217 files conflated
this with `.rst` files elsewhere in `airflow-core` (ADRs, etc.); the 155
figure here is the corrected, reproducible count, verified against the
category split in `corpus/corpus_summary.json`.

Docstring word count (89,475) is extracted per-symbol via Python's `ast`
module (module, class and function docstrings only) -- not raw source file
size, which would overstate prose content roughly 5x on a sampled check.

## What is deliberately excluded

**GitHub issues and discussions.** These are user-generated content governed
by GitHub's own Terms of Service, not the repository's Apache-2.0 grant.
Including them in a corpus described as "Apache-2.0 licensed" would misstate
what licence actually covers the material, so they are excluded entirely,
not filtered or sampled.

**71 of 76 available providers.** Only amazon, google, microsoft, databricks
and slack are sampled. All 76 providers' docs total 802,663 words across
1,185 files -- indexing all of them would make this an indexing-at-scale
project rather than a retrieval-quality project, and would leave most
providers with zero eval questions written against them. The five sampled
span both size (amazon/google large, databricks/slack small) and domain
(cloud infra vs. a messaging API), which is what the retrieval comparison
in `docs/DESIGN.md` needs.

## Reproducing this corpus

```
git clone https://github.com/apache/airflow.git
cd airflow
git checkout 2f846065252579d4ef8729ba8e42d96c5ab1b611
python extract_corpus.py   # see mlops/ for the extraction script
```

## What is committed

The extracted corpus (`corpus/`) is not committed to this repository; it is
regenerated from the pinned commit by `mlops/extract_corpus.py`, which
writes `corpus/manifest.json` and checks the commit. Excerpts of the
Apache-2.0 documentation do appear in committed files (retrieved chunks in
`results/`, the eval reading packet in `eval/`), so the Airflow `LICENSE`
and `NOTICE` are committed in `docs/`.
