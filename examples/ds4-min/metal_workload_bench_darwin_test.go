//go:build darwin && arm64

package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// These benchmarks run the real model through the real runtime, which the
// per-kernel benchmarks in metal_bench_darwin_test.go deliberately do not: a
// kernel benchmark on synthetic buffers cannot see how many command buffers a
// token costs, and that count is the whole prefill gap against the C oracle.
//
// The numbers they report were previously reachable only as a t.Logf line from
// the gated TestOracleWorkload, which runs once and so has no variance. These
// are ordinary benchmarks, so benchstat compares them:
//
//	export DS4_MODEL=/path/to/model.gguf
//	go test -c -o ds4-min.test
//	memguard -max-wired-gib 100 -max-wired-rise-gib -1 -- \
//	  ./ds4-min.test -test.bench='BenchmarkMetalPrefill|BenchmarkMetalDecode' \
//	  -test.benchtime=1x -test.count=6 -test.run='^$' | tee new.txt
//	benchstat old.txt new.txt
//
// The model load dominates process time and is shared: the runtime is built
// once per process and reused by every sub-benchmark, so a single invocation
// should ask for every case it wants compared.
//
// What is measured is throughput and submission structure, not numerics. The
// token ids are synthetic (see benchWorkloadTokens) and the context is smaller
// than the registered workload's, so a rate here is not the registered
// oracle-workload rate and must not be reported as one.

// benchWorkloadContext is the session context these benchmarks allocate. The
// registered oracle workload uses 65665; a benchmark that only needs a few
// thousand positions pays for the KV cache of every one it declares, so this
// defaults smaller and DS4_BENCH_CONTEXT raises it when a deeper sweep needs it.
const benchWorkloadContext = 4096

var benchWorkload struct {
	once    sync.Once
	runtime *metalRuntime
	err     error
}

// benchWorkloadRuntime returns the process-wide model runtime, loading it on
// first use. It is never closed: the process exits soon after the last
// benchmark, and closing it between sub-benchmarks would pay an 80 GiB load
// again for no measurement.
func benchWorkloadRuntime(b *testing.B) *metalRuntime {
	b.Helper()
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		b.Skip("DS4_MODEL is not set")
	}
	context := benchWorkloadContext
	if v := os.Getenv("DS4_BENCH_CONTEXT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			b.Fatalf("DS4_BENCH_CONTEXT: %v", err)
		}
		context = n
	}
	benchWorkload.once.Do(func() {
		benchWorkload.runtime, benchWorkload.err = newMetalRuntimeLogged(path, context, io.Discard)
	})
	if benchWorkload.err != nil {
		b.Fatal(benchWorkload.err)
	}
	return benchWorkload.runtime
}

// benchWorkloadTokens returns a deterministic token sequence.
//
// The ids are synthetic. Routing selects six of 256 experts per token whatever
// the ids are, and every other kernel runs the same shapes, so the dispatch
// count and the arithmetic per token do not depend on them. What does depend on
// them is the output, which is why these are benchmarks and the correctness
// gates run on the registered oracle tokenization instead.
func benchWorkloadTokens(n int) []int {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = (i*7919 + 13) % 129280
	}
	return tokens
}

// benchWorkloadSizes returns a sweep, letting an environment variable narrow it.
//
// Go's -bench pattern matches path elements positionally, so one pattern cannot
// select prefill chunks and decode depths independently: "MetalDecode/ctx0"
// filters the prefill sub-benchmarks out along with the deep decodes. These
// overrides let a plain `go test -bench` command name the cases it can afford
// -- DS4_BENCH_DEPTHS=0,512 -- without a wrapper script deciding for it.
func benchWorkloadSizes(b *testing.B, key string, sweep []int) []int {
	b.Helper()
	v := os.Getenv(key)
	if v == "" {
		return sweep
	}
	var sizes []int
	for _, field := range strings.Split(v, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil {
			b.Fatalf("%s: %v", key, err)
		}
		if n < 0 {
			b.Fatalf("%s: negative size %d", key, n)
		}
		sizes = append(sizes, n)
	}
	if len(sizes) == 0 {
		b.Fatalf("%s: no sizes", key)
	}
	return sizes
}

// reportWorkload converts a run into the per-token metrics benchstat compares.
//
// cbufs/tok is the structural one: it is a count, not a timing, so unlike the
// rates it is unaffected by measurement overhead and comparable against any
// other implementation of the same model.
//
// ops/tok counts operations at the batch API's level, not GPU dispatches: a
// call that encodes four dispatches for one logical operation advances it by
// one (metal_batch_darwin.go documents that convention), and a token-major call
// that encodes one dispatch per token advances it by one for the whole batch.
// It was called dispatch/tok until the token-major encoders landed, which made
// the two diverge by a factor of the token count -- and in the flattering
// direction, since batching N tokens into one call would have shown the metric
// falling while the GPU ran exactly as many dispatches as before. Nothing here
// counts GPU dispatches; do not read this as though something did.
//
// gpu_ms/tok and %gpu come from the union of the command-buffer spans, not
// their sum. Buffers overlap in wall time by design -- commit ed187ec commits a
// layer's opening buffer without waiting so the closing buffer can be encoded
// against it -- and a sum counts an overlapped interval once per buffer, so it
// is an upper bound on GPU busy time and can exceed wall time outright. The sum
// is still reported as gpumax_ms/tok, since the gap between the two is the work
// the overlap put in flight together; %overlap states that gap as a fraction of
// the sum. host_ms/tok is the wall time the union does not cover, which is
// where submission cost, host round trips and copies land.
//
// There is no %idle here. Spans accumulate across benchmark iterations while
// wall does not include the untimed reset between them, so the extent of the
// whole span set is not a duration these metrics may divide by.
func reportWorkload(b *testing.B, tokens int, spans []metalRegion, subs metalSubmissions, wall time.Duration) {
	b.Helper()
	if tokens == 0 {
		b.Fatal("no tokens evaluated")
	}
	if len(spans) == 0 {
		b.Fatal("no command buffer spans recorded")
	}
	st := analyzeSpans(spans)
	if st.Count != subs.Buffers {
		b.Fatalf("recorded %d command buffer spans but created %d command buffers; one of the two paths is missing a buffer", st.Count, subs.Buffers)
	}
	buffers, commits, waits, encoded := subs.perToken(tokens)
	seconds := wall.Seconds()
	perToken := seconds / float64(tokens)
	gpuPerToken := st.Union / float64(tokens)
	b.ReportMetric(float64(tokens)/seconds, "tok/s")
	b.ReportMetric(buffers, "cbufs/tok")
	b.ReportMetric(commits, "commits/tok")
	b.ReportMetric(waits, "waits/tok")
	b.ReportMetric(encoded, "ops/tok")
	b.ReportMetric(gpuPerToken*1e3, "gpu_ms/tok")
	b.ReportMetric(st.Sum/float64(tokens)*1e3, "gpumax_ms/tok")
	b.ReportMetric((perToken-gpuPerToken)*1e3, "host_ms/tok")
	b.ReportMetric(st.Union/seconds*100, "%gpu")
	b.ReportMetric(st.Overlap/st.Sum*100, "%overlap")
}

// BenchmarkMetalPrefill evaluates a prompt chunk from a fresh session.
//
// It sweeps the chunk size because prefill cost per token is not flat: the
// attention each position runs covers the positions before it. A chunk that
// grows while the reported metrics stay flat is itself the finding, since it
// says the run is paying per-token submission rather than per-token arithmetic.
func BenchmarkMetalPrefill(b *testing.B) {
	for _, chunk := range benchWorkloadSizes(b, "DS4_BENCH_CHUNKS", []int{32, 128, 512}) {
		b.Run(fmt.Sprintf("chunk%d", chunk), func(b *testing.B) {
			r := benchWorkloadRuntime(b)
			if chunk >= r.ctx {
				b.Skipf("chunk %d does not fit context %d", chunk, r.ctx)
			}
			tokens := benchWorkloadTokens(chunk)
			var spans []metalRegion
			var subs metalSubmissions
			var wall time.Duration
			evaluated := 0
			for b.Loop() {
				b.StopTimer()
				if err := r.Reset(); err != nil {
					b.Fatal(err)
				}
				log := &metalRegionLog{}
				r.ops.regions = log
				// The counter is bracketed at the same two points the region
				// log is installed and removed, so the two cover the same
				// buffers and reportWorkload can require them to agree. Reset
				// falls outside deliberately: it is untimed setup.
				beforeSubs := r.ops.submissions
				b.StartTimer()

				start := time.Now()
				err := r.EvalBatch(tokens, 0)
				elapsed := time.Since(start)

				b.StopTimer()
				r.ops.regions = nil
				subs = subs.add(r.ops.submissions.sub(beforeSubs))
				if err != nil {
					b.Fatal(err)
				}
				wall += elapsed
				evaluated += chunk
				spans = append(spans, log.all()...)
				b.StartTimer()
			}
			reportWorkload(b, evaluated, spans, subs, wall)
		})
	}
}

// BenchmarkMetalDecode evaluates single tokens at a given context depth.
//
// The depth is prefilled once, outside the timer, because decode cost depends
// on how much context attention covers and a decode benchmark from an empty
// session measures the cheapest step the runtime can take rather than a
// representative one. Depth then drifts upward by one per iteration, which is
// what generation does.
func BenchmarkMetalDecode(b *testing.B) {
	for _, depth := range benchWorkloadSizes(b, "DS4_BENCH_DEPTHS", []int{0, 512, 2048}) {
		b.Run(fmt.Sprintf("ctx%d", depth), func(b *testing.B) {
			r := benchWorkloadRuntime(b)
			if depth >= r.ctx {
				b.Skipf("depth %d does not fit context %d", depth, r.ctx)
			}
			if err := r.Reset(); err != nil {
				b.Fatal(err)
			}
			if depth > 0 {
				if err := r.EvalBatch(benchWorkloadTokens(depth), 0); err != nil {
					b.Fatal(err)
				}
			}
			tokens := benchWorkloadTokens(r.ctx)
			log := &metalRegionLog{}
			r.ops.regions = log
			defer func() { r.ops.regions = nil }()
			beforeSubs := r.ops.submissions

			var wall time.Duration
			evaluated := 0
			b.ResetTimer()
			for b.Loop() {
				if r.pos+1 >= r.ctx {
					b.Fatalf("decode reached context %d; raise DS4_BENCH_CONTEXT or lower -test.benchtime", r.ctx)
				}
				start := time.Now()
				err := r.Eval(tokens[r.pos], r.pos)
				wall += time.Since(start)
				if err != nil {
					b.Fatal(err)
				}
				evaluated++
			}
			b.StopTimer()
			reportWorkload(b, evaluated, log.all(), r.ops.submissions.sub(beforeSubs), wall)
		})
	}
}
