# Reading packet for the eval question set

Real excerpts pulled directly from the pinned corpus (commit
`2f846065252579d4ef8729ba8e42d96c5ab1b611`). Read through this one document
instead of opening files yourself. For each excerpt that has something you'd
plausibly ask about, jot a question. You don't need all of them -- skip
anything that doesn't spark a question, use what does.

Each entry has: the source file path (this is what goes in
`expected_source`), the real text, and a suggested question angle so you
have a starting point rather than a blank line -- but write it your own way,
or write a different question about the same excerpt if something else
strikes you.

Target: ~55 questions total across everything below plus your own additions.
Roughly 30 direct (one excerpt), 15 synthesis (two excerpts that connect --
a few pairs are flagged below), 10 adversarial (no excerpt needed -- see the
note at the end).

---

## 1. XComs -- `airflow-core/docs/core-concepts/xcoms.rst`

> XComs are explicitly "pushed" and "pulled" to/from their storage using the
> xcom_push and xcom_pull methods... they are only designed for small
> amounts of data; do not use them to pass around large values, like
> dataframes.
>
> xcom_pull() without a task_ids argument pulls only from the current task.
> In Airflow 2, the same call would search all tasks and return the most
> recent value. Always specify task_ids explicitly when pulling from other
> tasks.

Angle: what changed in xcom_pull's default behaviour between Airflow 2 and
now, and what breaks if you don't know that.

## 2. Pools -- `airflow-core/docs/administration-and-deployment/pools.rst`

> Note that if tasks are not given a pool, they are assigned to a default
> pool default_pool, which is initialized with 128 slots and can be
> modified through the UI or CLI (but cannot be removed).
>
> As slots free up, queued tasks start running based on the priority-weight
> of the task and its descendants.

Angle: what pool does an unassigned task land in, and how many slots does
it start with.

## 3. Priority weight -- `airflow-core/docs/administration-and-deployment/priority-weight.rst`

> priority_weight defines priorities in the executor queue. The default
> priority_weight is 1... By default, Airflow's weighting method is
> downstream. The effective weight of the task is the aggregate sum of all
> downstream descendants.

Angle (pairs with #2 above -- SYNTHESIS): pools.rst says queueing order
depends on priority-weight but doesn't explain it; this file does. A
question combining "why did my queued task run before another one" needs
both.

## 4. Deferrable operators -- `airflow-core/docs/authoring-and-scheduling/deferring.rst`

> By default, tasks in a deferred state don't occupy pool slots. If you
> would like them to, you can change this by editing the pool in question.
>
> Airflow 3.2 also supports Python-native async tasks... For guidance on
> when to use deferred operators versus async tasks, see Deferred vs Async
> Operators.

Angle: does a deferred task hold its pool slot while waiting, and what's
the newer async-task alternative mentioned.

## 5. Dynamic task mapping -- `airflow-core/docs/authoring-and-scheduling/dynamic-task-mapping.rst`

> Unlike a Python for-loop executed at DAG parse time, dynamic task mapping
> defers task creation until runtime, allowing the scheduler to determine
> the exact number of task instances based on upstream task outputs.
>
> Only keyword arguments are allowed to be passed to expand().

Angle: why can't you just use a for-loop to generate a variable number of
tasks based on an upstream result, and what's the argument-passing
restriction on expand().

## 6. Backfill -- `airflow-core/docs/core-concepts/backfill.rst`

> Backfill does not make sense for Dags that don't have a time-based
> schedule.
>
> You can set max_active_runs on a backfill... applied independently of the
> Dag max_active_runs setting.
>
> You can run your backfill in reverse, i.e. latest runs first. The CLI
> option is --run-backwards.

Angle: does a backfill's max_active_runs share a limit with the DAG's own
max_active_runs setting (no -- independent), and how do you backfill
newest-first.

## 7. Fernet encryption -- `airflow-core/docs/security/secrets/fernet.rst`

> Airflow uses Fernet to encrypt passwords in the connection configuration
> and the variable configuration... The first time Airflow is started, the
> airflow.cfg file is generated with the default configuration and the
> unique Fernet key.
>
> export AIRFLOW__CORE__FERNET_KEY=your_fernet_key

Angle: where does the Fernet key come from on first startup, and how do you
override it via environment variable (note the double underscore syntax).

## 8. Dynamic DAG generation -- `airflow-core/docs/howto/dynamic-dag-generation.rst`

> If you want to use variables to configure your code, you should always
> use environment variables in your top-level code rather than Airflow
> Variables. Using Airflow Variables in top-level code creates a connection
> to the metadata DB of Airflow to fetch the value, which can slow down
> parsing and place extra load on the DB.

Angle (pairs with #12 below on Variables -- SYNTHESIS possible): why is
using an Airflow Variable at the top level of a DAG file discouraged, in
favour of environment variables.

## 9. Slack connection -- `providers/slack/docs/connections/slack.rst`

> Authenticate to Slack using a Slack API token... The default Slack API
> Connection ID is slack_api_default.

Angle: what credential type does the Slack provider need, and what's the
default connection ID if you don't specify one.

## 10. AWS connection testing -- `providers/amazon/docs/connections/aws.rst`

> During this test components of Amazon Provider invoke AWS Security Token
> Service API GetCallerIdentity. This service can only check if your
> credentials are valid... it is not possible to validate if credentials
> have access to specific AWS service or not.
>
> If you use the Amazon Provider to communicate with AWS API compatible
> services (MinIO, LocalStack, etc.) test connection failure doesn't mean
> that your connection has wrong credentials.

Angle: if the AWS connection's "test connection" button passes, does that
mean your credentials have the permissions your task needs (no) -- and why
might test connection fail against MinIO/LocalStack even with correct
credentials.

## 11. Google Cloud connection -- `providers/google/docs/connections/gcp.rst`

> There are three ways to connect to Google Cloud using Airflow: 1.
> Application Default Credentials, 2. a service account by specifying a key
> file... Key can be specified as a path to the key file, as a key payload,
> or as secret in Secret Manager. Only one way of defining the key can be
> used at a time.

Angle: what are the three ways to authenticate the Google Cloud connection,
and can you combine key-file-path with key-payload (no, only one at a
time).

## 12. Databricks copy-into operator -- `providers/databricks/docs/operators/copy_into.rst`

> The only required parameters are: table_name, file_location, file_format
> (Supported formats are CSV, JSON, AVRO, ORC, PARQUET, TEXT, BINARYFILE).
> One of sql_endpoint_name or http_path.

Angle: what file formats does DatabricksCopyIntoOperator support, and what
are the minimum required parameters.

---

## Adversarial candidates (no excerpt needed -- write directly)

These test whether the system says "I don't know" instead of guessing. Pick
a few, or write your own along the same lines:

- Ask about a provider that is NOT in the 5 sampled (amazon, google,
  microsoft, databricks, slack) -- e.g. Snowflake, Postgres, Kubernetes
  provider specifically, Salesforce, Datadog.
- Ask about an Airflow version or feature you don't see mentioned anywhere
  above (make sure it's actually absent, not just absent from this packet
  -- if unsure, ask me to check).
- Ask about a plausible-sounding but invented config option, e.g. "what
  does the `auto_retry_backoff_multiplier` setting do" (not a real
  Airflow setting, as far as this packet shows).
- Ask something specific to a provider's docs but about a DIFFERENT
  provider than the one that actually documents it (e.g. ask about AWS
  Secrets Manager key rotation when only GCP Secret Manager appears above).

---

## How to turn this into `eval/questions.json`

For each question you write, note: the question text, the file path from
above (or "none" for adversarial), a short expected answer pulled from the
excerpt (or "no answer in corpus" for adversarial), and whether it's
direct/synthesis/adversarial. Plain text or a rough list is fine -- send it
back in whatever form is fastest for you and it'll get formatted into JSON.
