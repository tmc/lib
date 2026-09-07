# Pretrained PJRT reference

`generate.py` runs a local checkpoint through Transformers 4.57.1 and PyTorch
2.9.0 with eager attention and float32 weights by default. It records every output logit
and greedy predictions for four prompts, including 24 decode steps after a longer
prefill. It uses tokenizers 0.22.2 and
loads no remote Python code. Install these versions in a separate Python
environment before running it.

```sh
python generate.py /path/to/checkpoint /path/to/reference
```

The output includes SHA-256 hashes for checkpoint assets and logit files.
Keep the model weights and generated reference outside the repository.
From `examples`, run the native comparison on Linux with a CPU PJRT plugin:

```sh
GOWORK=off \
XLA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \
PJRT_PRETRAINED_CHECKPOINT=/path/to/checkpoint \
PJRT_PRETRAINED_REFERENCE=/path/to/reference \
go test -tags=tpu -race -timeout=20m -run TestNativePJRTPretrainedReference -v ./llama-infer
```

The test checks input IDs, every logit, greedy IDs, and streamed text. The logit
comparison allows `0.001 + 0.0001*abs(reference)` error for float32 reduction
order differences; greedy IDs must match exactly. This tests the supplied model
and CPU plugin. It does not establish TPU correctness or comparative performance.

Use `--prompts prompts.json` to supply a JSON list of objects such as
`{"prompt":"The capital of France is","steps":5}`. `--capacity` sets the
context capacity (default 96). Output directories must be new.

For precision measurements, first generate a free-running float32 reference.
Then generate a half-precision baseline on that reference's token history:

```sh
python generate.py /path/to/checkpoint /path/to/hf-f16 \
  --dtype float16 --teacher /path/to/reference
```

From `examples`, record the Go implementation on the same history:

```sh
GOWORK=off \
XLA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \
PJRT_PRETRAINED_CHECKPOINT=/path/to/checkpoint \
PJRT_PRETRAINED_REFERENCE=/path/to/reference \
PJRT_PRECISION=f16 \
PJRT_PRECISION_OUTPUT=/path/to/native-f16 \
go test -tags=tpu -timeout=20m -run '^TestNativePJRTPrecisionDiagnostic$' -v ./llama-infer
```

The diagnostic's PASS means it recorded finite logits. Evaluate those logits
separately, using the Python environment above:

```sh
python llama-infer/testdata/pretrained/compare.py \
  /path/to/reference /path/to/native-f16 --baseline /path/to/hf-f16 \
  --require-not-worse
```

The comparator checks hashes, token histories, finite values, and recorded
argmax IDs. It reports absolute error, RMSE, and greedy matches against float32.
With `--require-not-worse`, it exits 1 after printing the report if the candidate
has greater RMSE or fewer greedy matches than the baseline. Without that flag,
the numerical results are diagnostic. Invalid inputs exit 2.

For BF16, use `--dtype bfloat16` and `PJRT_PRECISION=bf16` with fresh output
directories. The Go diagnostic also accepts `f32`. In teacher-forced outputs,
`teacher_ids` holds the forced history and `generated_ids` holds each row's
actual argmax; these choices do not form a free-running text sample. The teacher
reference's JSON hash ties each output to its history. These measurements concern
numerical fidelity on the supplied prompts, not answer quality or performance.
