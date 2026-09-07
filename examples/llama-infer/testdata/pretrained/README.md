# Pretrained PJRT reference

`generate.py` runs a local checkpoint through Transformers 4.57.1 and PyTorch
2.9.0 with eager attention and float32 weights. It records every output logit
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
