// Ds4-benchstat converts ds4-bench measurements of the ds4.c oracle into the Go
// benchmark format that benchstat reads, so oracle rates and ds4-min's own
// benchmarks can be compared with one tool.
//
// ds4-bench writes one CSV row per context length:
//
//	ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes
//	2048,2048,310.54,128,28.59,34.049,127,28.61,0
//
// Older CSVs carry only the first five columns and kvcache_bytes; columns are
// read by header name, so both shapes work.
//
// Each row becomes up to four benchmarks, named by context length:
//
//	BenchmarkDS4Prefill/ctx=2048          iterations = prefill tokens, one op = one prefilled token
//	BenchmarkDS4Generate/ctx=2048         iterations = generated tokens, one op = one generated token
//	BenchmarkDS4GenerateSteady/ctx=2048   generation excluding the first cycle
//	BenchmarkDS4KVCache/ctx=2048          one op = holding the context
//
// Usage:
//
//	ds4-benchstat [-backend name] [-host name] [file...]
//	ds4-benchstat -receipt RESULT.json...
//
// With no files, the CSV is read from standard input. Output goes to standard
// output.
//
// # Receipts
//
// The -receipt form reads the RESULT.json receipts written for qualified oracle
// runs, taking the CSV fields from the receipt's csv object and the backend,
// host, oracle revision and input hashes from its metadata. A receipt whose
// result is not qualified is refused rather than converted: a run that failed
// its after-qualification is not admissible as a rate, and silently turning one
// into a benchmark line would launder it into a comparison.
//
// # What the numbers do and do not carry
//
// ds4-bench measures each context length once, so a converted file has n=1 per
// benchmark and benchstat will report no confidence interval. Two files may be
// compared for a ratio, but the ratio has no significance behind it.
//
// A kvcache_bytes of 0 means the field was not reported, not that the cache was
// empty; those rows emit no BenchmarkDS4KVCache line. Likewise a rate of 0 is
// ds4-bench's sentinel for an unmeasured phase, and emits no line.
package main
