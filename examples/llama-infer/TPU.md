# TPU inference

On Linux, build with `-tags=tpu`. `GetOrCompileTPUEngine` loads libtpu and
creates a single-device f32 decoder. Close the engine after its last use;
all calls require external synchronization. The engine owns its PJRT client,
weights, history and programs. `Free` attempts all cleanup and returns errors.

For a synthetic byte-tokenizer run on a configured TPU host, from `examples`:

```sh
go run -tags=tpu ./llama-infer -model tiny -prompt hi -gen-tokens 4
```

Synthetic weights require `Config.TestMode`; the `tiny` preset enables it.
This checks the inference path and does not measure pretrained model quality.
`RunTPUSmokeTest` remains a separate matrix-multiply check.

Checkpoint loading accepts unscaled, full-attention LLaMA and Qwen2 models
with SiLU MLPs. It checks config options, tensor roles, shapes, source dtypes
and exact payload lengths. Unknown config fields and unsupported variants
return errors. Required dimensions must be supplied, including Qwen2 KV heads.
Scaled RoPE, partial rotary, sliding attention, output/MLP biases and integer
quantized formats are not supported. F16 and BF16 checkpoint values are widened
to f32; half-precision execution is not implemented. Rotary theta retains its
JSON float64 precision in the decoder. The context limit comes from the config
and may be reduced by the caller's `MaxSeqLen`.

`GenerateFromTokenIDs` runs greedy generation from exact prompt IDs. The caller
includes BOS and chat-template tokens. EOS IDs come from config.json, with
missing metadata read from generation_config.json. Generation parameters such
as temperature in that file do not change the greedy algorithm. The callback
receives each generated ID, including EOS, and can stop by returning false.
A request resets prior logical history. A zero token limit performs no inference.
A full-capacity prompt can still predict one token without appending it.

`Generate` and `GenerateTokenIDs` load the checkpoint's `tokenizer.json` using
`x/hftokenizer`. Supported byte-level BPE pipelines supply their own BOS template;
per-call BOS and EOS defaults do not override checkpoint metadata. Supply rendered
text: chat templates in `tokenizer_config.json` are not interpreted. Unsupported
pipelines and token IDs outside the model vocabulary return errors.

The CLI validates config and tokenizer before loading weights or opening PJRT.
Programmatic engines load the tokenizer lazily from their construction path, so
raw-ID generation requires no tokenizer file. Treat checkpoint assets as immutable.
Each text callback receives one ID and a valid UTF-8 fragment, which may be empty.
Normal completion flushes incomplete final bytes as replacement characters;
cancellation or a callback stop discards an incomplete trailing character.
Explicit synthetic models retain the byte tokenizer. Serving and mesh modes
are not implemented by this backend.

`ReadLogits` returns a copy of the last row. `ResetKV` clears logical history
and stored logits. Readback failures or cancellation after the decoder commits
close the model, because retrying that step would use advanced history.
`WeightReport.Bytes` is the sum of uploaded parameter byte sizes, not allocator
telemetry; PJRT's compiled kernel names are not reported.

The portable CPU checks run from `examples` with an installed CPU PJRT plugin:

```sh
XLA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \
  go test -race -run 'TestPJRT|TestNativePJRT' ./llama-infer
```

On Linux, the production adapter can also be tested with the CPU plugin using
`-tags=tpu -run TestNativeTPUModelEngineWithCPU`. CPU execution and Linux
cross-compilation do not establish physical TPU correctness or throughput.
Set `RUN_TPU_HARDWARE_TESTS=1` when running the TPU hardware tests so unavailable
hardware cannot turn a requested hardware check into a skip.
