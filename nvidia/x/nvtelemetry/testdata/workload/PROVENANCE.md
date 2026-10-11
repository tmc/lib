# Workload fixture provenance

`provenance.json` records raw fixture SHA-256 values and the source-file hashes
retained with the lane receipt. The following upstream versioned files define
the supported mappings; current/latest documentation alone was not used to
claim compatibility with pinned versions.

- Triton core `r25.06` [metrics.cc](https://github.com/triton-inference-server/core/blob/r25.06/src/metrics.cc): counter/gauge families, cache lookups and first-response histogram.
- vLLM `v0.8.5` [engine/metrics.py](https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/engine/metrics.py): legacy-engine counters, finish-reason labels, KV-cache fractions and histogram families. V1-specific metrics are a separate pending mapping.
- Ray `ray-2.44.0` [metric_defs.cc](https://github.com/ray-project/ray/blob/ray-2.44.0/src/ray/stats/metric_defs.cc), [metric_defs.h](https://github.com/ray-project/ray/blob/ray-2.44.0/src/ray/stats/metric_defs.h) and [reporter_agent.py](https://github.com/ray-project/ray/blob/ray-2.44.0/python/ray/dashboard/modules/reporter/reporter_agent.py): logical resource gauges and selected reporter GPU gauges.
- MLflow `v2.22.0` [service.proto](https://github.com/mlflow/mlflow/blob/v2.22.0/mlflow/protos/service.proto) and [proto_json_utils.py](https://github.com/mlflow/mlflow/blob/v2.22.0/mlflow/utils/proto_json_utils.py): metric-history fields and native integer-number JSON serialization.
- W&B `v0.19.11` [handler.py](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/handler.py) and [wandb_run.py](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/wandb_run.py): history scalars, explicit history steps and timestamp context.
- ClearML `v1.18.0` [v2_9/events.py](https://github.com/clearml/clearml/blob/v1.18.0/clearml/backend_api/services/v2_9/events.py) and [callresult.py](https://github.com/clearml/clearml/blob/v1.18.0/clearml/backend_api/session/callresult.py): scalar-event schema, response-data envelope and successful result code.
- PyTorch `v2.7.0` [test_profiler.py](https://github.com/pytorch/pytorch/blob/v2.7.0/test/profiler/test_profiler.py) and Kineto `v0.4.0` [output_json.cpp](https://github.com/pytorch/kineto/blob/v0.4.0/libkineto/src/output_json.cpp): user annotations and Chrome-trace timestamp/duration serialization. No claim that a bounded profile is a live workload metric stream.

`mlflow-native.json` is an actual vendor SDK file-store history/response
serialization captured on 2026-10-10 with `mlflow-skinny==2.22.0`. The receipt
records the real run ID and exact hash. `generate_mlflow.py` repeats the native
SDK/serializer path in an isolated environment; it requires explicit new store
and output directories and performs no endpoint or account operations.

All other fixtures are hand-constructed protocol controls grounded in the
versioned sources above. They are not production traces or genuine captures from
those pinned running frameworks. Their pending native-export gates are listed
in `WORKLOADS.md`. `mlflow.json` is also a protocol control, separate from the
native fixture. Large integer, dynamic-batch, failure, cache and abort fields in
controls test preservation/refusal behavior, not measured application outcomes.
