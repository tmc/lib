# Independent upstream feasibility audit

Scope: read-only static feasibility of exact F32 Llama-3.2-1B, 8 input tokens, 32 output tokens, B1 and true B4 decode/serving calibration against unmodified upstream. No GPU or producer analysis executed. Completed within 15 active minutes. Source was read with `git show HEAD:path`; tracked local changes were inspected separately and are not upstream.

Repository: `/Users/tmc/tmp/lib-excellence-goal/parity-20260904-luminal`.
Pinned HEAD verified live: `598e5ca7aa48c40a0efcc703d41ac207107de043`.

## Finding

Neither shipped entrypoint runs the exact registered workload unchanged. A new caller can reuse upstream primitives, but that is not calibration against the unmodified shipped application. Under the literal registered requirement to calibrate the instrumented comparator against unmodified upstream on a matching timer/workload, the exact-cell reference prerequisite is absent. This is a source-level interface mismatch, not a measured claim about speed or proof that building a broader comparator is impossible.

## CUDA

Upstream `examples/llama/src/main.rs:22-32` selects 8B model IDs, 500 generated tokens, a fixed natural-language prompt, 500 search graphs and repetition penalty 1.05. Argument handling at `:64-94` accepts BF16/FP8/help only and rejects all other flags. The runtime instantiates the default model at `:299-304`, and obtains the prompt by chat-template tokenization at `:283-289`. These concrete call sites preclude requesting F32 1B/8/32 by runtime configuration.

The underlying `examples/llama/src/model.rs` does have F32 and custom dimensions: `:21-49`, `:58-65`, `:161-168`. It would therefore be wrong to claim that upstream cannot represent a small F32 model at all. However, its model initializer independently names `lm_head.weight` at `:190-191`; the local tracked patch changes that to `model.embed_tokens.weight` for the campaign checkpoint's tied embeddings. That patch is a semantic weight-binding change, not only timing instrumentation. An adapter using it must not be described as unmodified upstream. A reconstructed checkpoint that adds an alias is also a changed checkpoint representation whose equivalence must be proven, not an automatic match to the pinned file manifest.

CUDA exposes one token vector and one shared context vector (`main.rs:293-298`), one context length and one next token (`:409-421`), and a serial autoregressive loop supplying exactly one current token (`:463-497`). The cache has shape `(num_slots, kv_dim)` per layer (`model.rs:111-127`). Attention uses a shared gathered context and a causal triangle indexed by query position (`:794-821`), with no request/block identity to prevent four independent sequences attending to one another. The sampling state is one shared vocabulary mask (`:227-239`). Setting the sequence dimension to four is four positions of one context, not four independent decode requests. True B4 requires new request-separated state/masking or separate graphs/runtimes plus explicit batching semantics.

## Metal

Upstream `crates/luminal_metal/examples/llama_1b.rs:20-38` is already the 1B model family, with F32 cache storage established at `:483-486`, and 100 search graphs rather than one. It nevertheless hardcodes 96 generated tokens and a different prose prompt. `main` has no argument parsing; it downloads the fixed repo (`:40-44`, `:417`), chat-tokenizes the fixed prompt (`:420-426`) and uses `GEN_TOKENS` throughout (`:454`, `:516-522`, `:566`). There is no runtime exact-token/model-path override in HEAD.

The local tracked diff adds LIB_COMPARE_MODEL/TOKENS/REPEATS/MANAGED controls and changes GEN_TOKENS and SEARCH_GRAPHS to 1. Those are historical modified comparator facilities, not upstream features. Upstream's 100-candidate search capacity is real, but it does not imply that the local search-limit-1 artifact is tuned.

Metal holds one cache set (`:98-118`), one request state (`:509-522`), a one-token decode call (`:573-589`), and one generated stream (`:592-611`). The shared `(s,c)` attention mask is causal for one history (`:86-95`, `:428-442`). Independent B4 requests require caller/model construction. Running the existing binary four times provides four B1 programs; it does not establish matched true B4 execution.

## Serving and timer boundary

Both concrete entrypoints initialize one model, generate one response, print timing statistics and exit. CUDA's request progression is `main.rs:409-520`; Metal's is `llama_1b.rs:509-632`. Neither accepts a client request, dispatches a queue, or maintains per-request arrival/finish timestamps. Both use locally generated prompts and write decoded text to stdout. This executable control flow, not filename absence, establishes that a serving adapter and client harness must be built for the registered end-to-end requirement.

A bounded HEAD search for Rust HTTP/socket server APIs found `docs/company/src/main.rs`. Reading its full 43-line implementation shows only static HTML/favicon/redirect routes; it is not an inference server. This corroborates, but does not by itself prove, the absence of serving machinery elsewhere.

The existing reported timing is also not the requested client-through-final-token interval. Metal's `run_model_step` closes its timer before returning (`:400-408`); CPU greedy sampling and stdout decode/flush occur afterward (`:545-562`, `:592-611`), while TTFT/TPOT sum only step profiles (`:615-619`). CUDA's `run_model_step` returns an on-device selected token (`main.rs:221-265`), but decode/stdout work still happens after measured steps, and reported TPOT additionally skips the first decode duration (`:535`). Comparing these summary fields with a serving timer would violate matching scope.

## Implications for the bounded campaign

- A tuned B1 correctness probe can help test whether the newly constructed adapter faithfully implements the checkpoint. It cannot alone supply the missing unmodified exact-workload reference or end-to-end calibration.
- Do not change the frozen 10% calibration rule after observing timing. If it remains literally tied to an unmodified upstream application, close exact CUDA and Metal calibration as unavailable and identify the hardcoded workload/entrypoint interface as the missing prerequisite.
- A separate, explicitly prospective interpretation that permits a new harness around unchanged library APIs would need its own declared reference boundary and source identity before timing. This report does not authorize that reinterpretation, certify its feasibility within remaining budget, or accept any samples.
- B4 and serving require concrete implementation, not argument selection. Their construction must fit the existing budget or close with those prerequisites. This report does not assert that Luminal is intrinsically incapable of either workload.
