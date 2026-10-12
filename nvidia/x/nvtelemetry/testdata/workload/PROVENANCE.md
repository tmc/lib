# Workload fixture provenance

`provenance.json` records raw fixture SHA-256 values and the source-file hashes
retained with the lane receipt. The following upstream versioned files define
the supported mappings; current/latest documentation alone was not used to
claim compatibility with pinned versions.

- Triton core `r25.06` [metrics.cc](https://github.com/triton-inference-server/core/blob/r25.06/src/metrics.cc): counter/gauge families, cache lookups and first-response histogram.
- vLLM `v0.8.5` [engine/metrics.py](https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/engine/metrics.py): legacy-engine counters, finish-reason labels, KV-cache fractions and histogram families. V1-specific metrics are a separate pending mapping.
- Ray `ray-2.44.0` [metric_defs.cc](https://github.com/ray-project/ray/blob/ray-2.44.0/src/ray/stats/metric_defs.cc), [metric_defs.h](https://github.com/ray-project/ray/blob/ray-2.44.0/src/ray/stats/metric_defs.h) and [reporter_agent.py](https://github.com/ray-project/ray/blob/ray-2.44.0/python/ray/dashboard/modules/reporter/reporter_agent.py): logical resource gauges and selected reporter GPU gauges.
- SGLang `v0.5.21` peeled commit `e00930c5489053f26d86b179cee0d087f846acbb` [metrics_collector.py](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/observability/metrics_collector.py): exact metric families, types, label names and family-specific units. SHA256 `b1821b90ad18ce82d75648a1da76f850b5d5c7c3fadc1b39a7cc694d00533136`.
- Dynamo `v1.5.1` commit `2093662a4d4e5fd318732935213dde08dc7be818` [metrics catalog](https://github.com/ai-dynamo/dynamo/blob/2093662a4d4e5fd318732935213dde08dc7be818/docs/fern/pages/reference/observability/metrics-catalog.mdx), [metric labels](https://github.com/ai-dynamo/dynamo/blob/2093662a4d4e5fd318732935213dde08dc7be818/docs/fern/pages/reference/observability/metric-labels.mdx): runtime family types, meanings, scopes and labels. Snapshot SHA256 values are `255672389599da2493aab1206101851d54f42a98132475c310089297d3768d43` (catalog), `1534082dbc8a1eebdfbb9186110c3b16a8c99f806c09c7e25f0ace1b6459407b` (labels), and `38e7078609d29ce26e8edf1fc15336e79bb7c81999f56860be765596c9d82c8a` (runtime Dockerfile). The tagged runtime Dockerfile uses `${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG}` and does not pin SGLang, so this does not establish a compatible binary pairing.
- The Dynamo engine gauges `dynamo_component_total_blocks`, `dynamo_component_gpu_cache_usage_percent`, and `dynamo_component_model_load_time_seconds` are defined by the pinned implementation at `components/src/dynamo/common/utils/prometheus.py` and names at `lib/runtime/src/metrics/prometheus_names.rs`, commit `2093662a4d4e5fd318732935213dde08dc7be818`. Retained snapshot SHA256 values are `44965a5ff37346c2e6456f87c7f3a454ef0148e1777fbf44fb38f4afb530851c` and `f1ffbad29088e6512add04208364db766b67dee0635747795c9450e2f3fefdef`. The implementation requires `model` and `dp_rank` in addition to hierarchy labels for total blocks and GPU cache usage, and `model` in addition to hierarchy labels for model-load time. Its explicit `component` label is the injected `dynamo_component` label.

The added fixtures are deterministic, source-derived constructed controls:

- `sglang-v0.5.21.prom.control`: SHA256 `0f7d475df6ab85ed6efb24d5d8e9ce6366c8afd89955ce62c63b00a8a4a0ba12`.
- `dynamo-v1.5.1.prom.control`: SHA256 `70830e5dcfeb9d0ba289e4a80a4390fbf5b61c015a01cffbac62d3598fa39ace`.

The values, label combinations and timing are controlled data and are not an
authentic producer scrape. The reader recomputes and preserves each raw hash.
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
versioned sources above. `sglang-v0.5.21.prom.control` and
`dynamo-v1.5.1.prom.control` are constructed controls, not production traces or
genuine captures from those pinned running frameworks. Their pending
authentic-export gates are listed
in `WORKLOADS.md`. `mlflow.json` is also a protocol control, separate from the
native fixture. Large integer, dynamic-batch, failure, cache and abort fields in
controls test preservation/refusal behavior, not measured application outcomes.
