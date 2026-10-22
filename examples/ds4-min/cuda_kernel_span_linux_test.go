//go:build linux

package main

import (
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// kernelSpan accumulates one kernel name's device occupancy.
type kernelSpan struct {
	name     string
	launches int
	spanNS   uint64
	maxNS    uint64
	regs     uint16
}

// busyNS returns the union of the intervals, which is the device's busy time.
// The sum of the spans is not that number: kernels on different streams
// overlap, so the sum can exceed the wall clock it was measured inside.
func busyNS(intervals [][2]uint64) uint64 {
	if len(intervals) == 0 {
		return 0
	}
	sort.Slice(intervals, func(i, j int) bool { return intervals[i][0] < intervals[j][0] })
	var total uint64
	start, end := intervals[0][0], intervals[0][1]
	for _, iv := range intervals[1:] {
		if iv[0] > end {
			total += end - start
			start, end = iv[0], iv[1]
			continue
		}
		if iv[1] > end {
			end = iv[1]
		}
	}
	return total + (end - start)
}

// TestCUDAKernelSpan ranks the prefill's kernels by the device time they hold.
//
// It exists because the PC-sampling ranking cannot answer this. A PC sample says
// where a warp's program counter was when the sampler looked, so a kernel's share
// of samples is a share of resident warp-cycles. A kernel that runs at low
// occupancy for a long time and one that saturates the SMs briefly can trade
// places between the two measures. Anything chosen for optimization on the sample
// ranking alone is chosen on an unvalidated proxy for the thing that matters,
// which is the device time it would give back.
//
// The two rankings disagreeing is the finding, not a fault. Agreement is what
// licenses acting on either.
func TestCUDAKernelSpan(t *testing.T) {
	path := os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if path == "" {
		t.Skip("DS4_CUDA_PREFILL_TOKENS is required")
	}
	input, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("prefill input has %d tokens, want 2048", len(tokens))
	}
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("cuda device 0 not available: %v", err)
	}
	defer dev.Close()
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		t.Skipf("retain primary context: %v", err)
	}
	defer cuCtx.Release()

	r, err := newCUDARuntime(modelPath, 4096)
	if err != nil {
		t.Fatalf("newCUDARuntime: %v", err)
	}
	defer r.Close()

	// Warm up before attaching. PC sampling had to attach first, because it
	// needed the module's compilation in its own view; this does not, and a
	// ledger that counted the JIT's launches would be measuring the compiler.
	if err := r.Eval(100, 0); err != nil {
		t.Fatalf("warmup step: %v", err)
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if err := r.EvalBatch(tokens[:2], 0); err != nil {
		t.Fatalf("prefill warmup: %v", err)
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("reset after prefill warmup: %v", err)
	}
	if err := r.device.WithContext(func() error { return driver.CtxSynchronize() }); err != nil {
		t.Fatalf("synchronize after warmup: %v", err)
	}

	session, err := cuptiactivity.Start(cuptiactivity.Config{
		Kinds:       []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel},
		BufferBytes: 8 << 20,
		MaxBuffers:  256,
		MaxRecords:  4 << 20,
	})
	if err != nil {
		t.Skipf("start activity collection: %v", err)
	}

	start := time.Now()
	if err := r.EvalBatch(tokens, 0); err != nil {
		session.Close()
		t.Fatalf("prefill: %v", err)
	}
	// Close drains buffers but does not synchronize application work. An
	// unsynchronized close reports the kernels that happened to have landed.
	if err := r.device.WithContext(func() error { return driver.CtxSynchronize() }); err != nil {
		session.Close()
		t.Fatalf("synchronize: %v", err)
	}
	wall := time.Since(start)
	if err := session.Close(); err != nil {
		t.Fatalf("close activity session: %v", err)
	}
	result := session.Result()

	// A result is usable only if nothing was lost. These are the counters the
	// package keeps precisely so a truncated ledger cannot read as a complete
	// one, and a ranking taken from a truncated ledger is a ranking of what
	// survived.
	t.Logf("activity: %d records, closed=%v pending=%d refused=%d incomplete=%d discarded=%d dropped=%d errors=%d",
		len(result.Records), result.Closed, result.PendingBuffers, result.RefusedBuffers,
		result.IncompleteRecords, result.DiscardedRecords, result.DroppedRecords, result.CallbackErrors)
	if result.FirstError != nil {
		t.Fatalf("activity collection error: %v", result.FirstError)
	}
	if result.DroppedRecords != 0 || result.RefusedBuffers != 0 || result.PendingBuffers != 0 {
		t.Fatalf("activity collection lossy: dropped=%d refused=%d pending=%d",
			result.DroppedRecords, result.RefusedBuffers, result.PendingBuffers)
	}

	// Every record is accounted for in exactly one of three counts, and they are
	// printed together so a filter that quietly excluded a population shows up
	// as a sum that does not close.
	var kernels, nonKernel, incomplete int
	spans := make(map[string]*kernelSpan)
	intervals := make([][2]uint64, 0, len(result.Records))
	for _, rec := range result.Records {
		if rec.Kind != cupti.ActivityKindConcurrentKernel && rec.Kind != cupti.ActivityKindKernel {
			nonKernel++
			continue
		}
		if !rec.Complete || rec.End < rec.Start {
			incomplete++
			continue
		}
		kernels++
		d := rec.End - rec.Start
		s := spans[rec.Name]
		if s == nil {
			s = &kernelSpan{name: rec.Name}
			spans[rec.Name] = s
		}
		s.launches++
		s.spanNS += d
		if d > s.maxNS {
			s.maxNS = d
		}
		s.regs = rec.RegistersPerThread
		intervals = append(intervals, [2]uint64{rec.Start, rec.End})
	}
	if kernels+nonKernel+incomplete != len(result.Records) {
		t.Fatalf("record accounting: %d kernel + %d non-kernel + %d incomplete != %d total",
			kernels, nonKernel, incomplete, len(result.Records))
	}
	if kernels == 0 {
		t.Fatal("no complete kernel records: the ledger is empty, not the device idle")
	}

	ranked := make([]*kernelSpan, 0, len(spans))
	var totalSpan uint64
	for _, s := range spans {
		ranked = append(ranked, s)
		totalSpan += s.spanNS
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].spanNS > ranked[j].spanNS })

	busy := busyNS(intervals)
	t.Logf("prefill wall %v, device busy %v (%.1f%% of wall), summed kernel span %v across %d launches in %d kernels",
		wall.Round(time.Millisecond), time.Duration(busy).Round(time.Microsecond),
		100*float64(busy)/float64(wall.Nanoseconds()),
		time.Duration(totalSpan).Round(time.Microsecond), kernels, len(ranked))
	t.Logf("%-58s %10s %8s %12s %10s %5s", "kernel", "span", "%span", "launches", "max", "regs")
	for i, s := range ranked {
		if i >= 25 {
			t.Logf("... and %d more holding %.1f%%", len(ranked)-i,
				100*float64(totalSpan-cumulative(ranked[:i]))/float64(totalSpan))
			break
		}
		t.Logf("%-58s %10s %7.2f%% %12d %10s %5d",
			truncName(s.name, 58), time.Duration(s.spanNS).Round(time.Microsecond),
			100*float64(s.spanNS)/float64(totalSpan), s.launches,
			time.Duration(s.maxNS).Round(time.Nanosecond), s.regs)
	}
}

func cumulative(s []*kernelSpan) uint64 {
	var n uint64
	for _, k := range s {
		n += k.spanNS
	}
	return n
}

func truncName(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

// TestCUDADecodeKernelSpan ranks the decode step's kernels by the device time
// they hold, and separates the kernels that cannot fill the device from the
// kernels that can.
//
// It is the decode counterpart of TestCUDAKernelSpan, and it exists for two
// reasons that the prefill ranking and the PC-sampling ranking each fail on.
//
// The PC-sampling profile cannot answer this at all. A kernel's share of PC
// samples is a share of resident warp-cycles, so a kernel that holds the device
// at low occupancy collects few samples however long it runs. In the decode
// profile taken on this host every kernel launched with a one-block grid fell
// below the 0.5% reporting floor -- not because those kernels are cheap, but
// because the measure is biased against exactly the shape under suspicion. A
// ranking by device span has no such bias: it is the time the device was busy
// with that kernel, which is the time an optimization could give back.
//
// The prefill ranking does not transfer. Prefill runs 2048 token rows through
// the same kernels, so grids that are one block wide in decode are wide there,
// and the geometry defect this test looks for is invisible in it.
//
// Graphs must be on. The single-token path takes its captured graph only when
// r.phaseMark is nil (cuda_session_linux.go:1787), so any capture that tags a
// phase -- which is how the PC-sampling profile selects decode -- silently
// measures a configuration the shipping binary never runs. This test never sets
// a phase: it brackets the region itself. That is not assumed. Kernels launched
// from a graph carry a nonzero GraphID, so the ledger is asked whether the
// graphs engaged, and the test fails if they did not.
func TestCUDADecodeKernelSpan(t *testing.T) {
	const decodeSteps = 32

	path := os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if path == "" {
		t.Skip("DS4_CUDA_PREFILL_TOKENS is required")
	}
	input, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("prefill input has %d tokens, want 2048", len(tokens))
	}
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("cuda device 0 not available: %v", err)
	}
	defer dev.Close()
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		t.Skipf("retain primary context: %v", err)
	}
	defer cuCtx.Release()

	r, err := newCUDARuntime(modelPath, 4096)
	if err != nil {
		t.Fatalf("newCUDARuntime: %v", err)
	}
	defer r.Close()
	if r.disableGraphs {
		t.Fatal("DS4_DISABLE_CUDA_GRAPH is set: this test measures the graphed decode path")
	}

	// Prefill outside the measured region. Decode reads a populated KV cache and
	// a real hidden state; decoding from a cold cache would measure a workload
	// the binary never runs. The warmup steps also force the graph captures, so
	// the measured region replays graphs instead of capturing them.
	if err := r.Eval(100, 0); err != nil {
		t.Fatalf("warmup step: %v", err)
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if err := r.EvalBatch(tokens, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	for i := 0; i < 4; i++ {
		if err := r.Eval(tokens[0], len(tokens)+i); err != nil {
			t.Fatalf("decode warmup step %d: %v", i, err)
		}
	}
	if err := r.device.WithContext(func() error { return driver.CtxSynchronize() }); err != nil {
		t.Fatalf("synchronize after warmup: %v", err)
	}

	session, err := cuptiactivity.Start(cuptiactivity.Config{
		Kinds:       []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel},
		BufferBytes: 8 << 20,
		MaxBuffers:  256,
		MaxRecords:  4 << 20,
	})
	if err != nil {
		t.Skipf("start activity collection: %v", err)
	}

	// A fixed token is fed for every step. Which token is decoded changes which
	// expert rows are gathered, not which kernels run or the geometry they run
	// with, and this test ranks kernels and geometry. It is not a correctness
	// run and does not gate on the output digest.
	base := len(tokens) + 4
	start := time.Now()
	for i := 0; i < decodeSteps; i++ {
		if err := r.Eval(tokens[0], base+i); err != nil {
			session.Close()
			t.Fatalf("decode step %d: %v", i, err)
		}
	}
	if err := r.device.WithContext(func() error { return driver.CtxSynchronize() }); err != nil {
		session.Close()
		t.Fatalf("synchronize: %v", err)
	}
	wall := time.Since(start)
	if err := session.Close(); err != nil {
		t.Fatalf("close activity session: %v", err)
	}
	result := session.Result()

	t.Logf("activity: %d records, closed=%v pending=%d refused=%d incomplete=%d discarded=%d dropped=%d errors=%d",
		len(result.Records), result.Closed, result.PendingBuffers, result.RefusedBuffers,
		result.IncompleteRecords, result.DiscardedRecords, result.DroppedRecords, result.CallbackErrors)
	if result.FirstError != nil {
		t.Fatalf("activity collection error: %v", result.FirstError)
	}
	if result.DroppedRecords != 0 || result.RefusedBuffers != 0 || result.PendingBuffers != 0 {
		t.Fatalf("activity collection lossy: dropped=%d refused=%d pending=%d",
			result.DroppedRecords, result.RefusedBuffers, result.PendingBuffers)
	}

	// smCount is the device's SM count. A grid narrower than this cannot occupy
	// the device however long it runs, so its span is time the rest of the
	// device was idle underneath it. This is the population the test exists to
	// size; it is reported as a total so that it is a number and not an anecdote.
	const smCount = 48

	type decodeSpan struct {
		kernelSpan
		minGrid, maxGrid int64
		block            int64
		graphed          int
	}
	var kernels, nonKernel, incomplete int
	var graphedLaunches int
	// Starvation is accumulated per RECORD, not per kernel name. A kernel
	// launched with more than one geometry would otherwise be classified
	// entirely by whichever launch happened to be seen last.
	var starvedSpan, starvedLaunches uint64
	spans := make(map[string]*decodeSpan)
	intervals := make([][2]uint64, 0, len(result.Records))
	for _, rec := range result.Records {
		if rec.Kind != cupti.ActivityKindConcurrentKernel && rec.Kind != cupti.ActivityKindKernel {
			nonKernel++
			continue
		}
		if !rec.Complete || rec.End < rec.Start {
			incomplete++
			continue
		}
		kernels++
		d := rec.End - rec.Start
		s := spans[rec.Name]
		if s == nil {
			s = &decodeSpan{kernelSpan: kernelSpan{name: rec.Name}}
			spans[rec.Name] = s
		}
		s.launches++
		s.spanNS += d
		if d > s.maxNS {
			s.maxNS = d
		}
		s.regs = rec.RegistersPerThread
		grid := int64(rec.Grid[0]) * int64(rec.Grid[1]) * int64(rec.Grid[2])
		if s.minGrid == 0 || grid < s.minGrid {
			s.minGrid = grid
		}
		if grid > s.maxGrid {
			s.maxGrid = grid
		}
		s.block = int64(rec.Block[0]) * int64(rec.Block[1]) * int64(rec.Block[2])
		if grid < smCount {
			starvedSpan += d
			starvedLaunches++
		}
		if rec.GraphID != 0 {
			s.graphed++
			graphedLaunches++
		}
		intervals = append(intervals, [2]uint64{rec.Start, rec.End})
	}
	if kernels+nonKernel+incomplete != len(result.Records) {
		t.Fatalf("record accounting: %d kernel + %d non-kernel + %d incomplete != %d total",
			kernels, nonKernel, incomplete, len(result.Records))
	}
	if kernels == 0 {
		t.Fatal("no complete kernel records: the ledger is empty, not the device idle")
	}
	// The graph gate, asked of the device rather than of the predicate that
	// selects it. Zero graphed launches means this ran the un-graphed path and
	// every number below describes a configuration the binary does not ship.
	if graphedLaunches == 0 {
		t.Fatalf("no kernel carried a graph id across %d launches: the graphed decode path did not engage", kernels)
	}

	ranked := make([]*decodeSpan, 0, len(spans))
	var totalSpan uint64
	for _, s := range spans {
		ranked = append(ranked, s)
		totalSpan += s.spanNS
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].spanNS > ranked[j].spanNS })

	busy := busyNS(intervals)
	perToken := func(n uint64) time.Duration { return time.Duration(n / decodeSteps) }
	t.Logf("decode %d steps: wall %v (%v/token), device busy %v (%.1f%% of wall), summed span %v across %d launches in %d kernels",
		decodeSteps, wall.Round(time.Millisecond), (wall / decodeSteps).Round(time.Microsecond),
		time.Duration(busy).Round(time.Microsecond), 100*float64(busy)/float64(wall.Nanoseconds()),
		time.Duration(totalSpan).Round(time.Microsecond), kernels, len(ranked))
	t.Logf("graphed launches: %d of %d (%.1f%%)", graphedLaunches, kernels,
		100*float64(graphedLaunches)/float64(kernels))
	t.Logf("grid < %d SMs: %v/token across %d launches/token, %.1f%% of summed span",
		smCount, perToken(starvedSpan), starvedLaunches/decodeSteps,
		100*float64(starvedSpan)/float64(totalSpan))
	t.Logf("%-52s %10s %10s %7s %8s %8s %6s %6s", "kernel", "span", "span/tok", "%span", "launch/t", "grid", "block", "graph")
	for i, s := range ranked {
		if i >= 30 {
			var rest uint64
			for _, o := range ranked[i:] {
				rest += o.spanNS
			}
			t.Logf("... and %d more holding %v/token (%.1f%%)", len(ranked)-i,
				perToken(rest), 100*float64(rest)/float64(totalSpan))
			break
		}
		grid := fmt.Sprintf("%d", s.maxGrid)
		if s.minGrid != s.maxGrid {
			grid = fmt.Sprintf("%d-%d", s.minGrid, s.maxGrid)
		}
		t.Logf("%-52s %10s %10s %6.2f%% %8d %8s %6d %5d%%",
			truncName(s.name, 52), time.Duration(s.spanNS).Round(time.Microsecond),
			perToken(s.spanNS), 100*float64(s.spanNS)/float64(totalSpan),
			s.launches/decodeSteps, grid, s.block,
			100*s.graphed/max(s.launches, 1))
	}
}
