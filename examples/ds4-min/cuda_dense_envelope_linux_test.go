//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

// The reference routes a dense Q8_0 GEMM to its dense kernel only inside a
// shape envelope (ds4_cuda.cu:14862-14863); our gate is a plain on/off, and
// eleven of our fifteen dense sites fail that envelope. This measures whether
// honouring it helps, hurts, or does nothing.
//
// The design is registered here rather than in a script, so the sequence, the
// block size, the firing control and the outcome classes are fixed before any
// timing exists:
//
//   - Three arms. BASE ships what we ship. ATTNOUT applies the envelope only
//     in the attention-output block, whose two shapes both fail it. FULL adds
//     the query/key-value block, where attnQA and attnKV fail and attnQB
//     passes. ATTNOUT-BASE and FULL-ATTNOUT are therefore disjoint: the first
//     is the nine attnOutA/attnOutB sites, the second the two qkv sites.
//   - Block size six, sequence BASE ATTNOUT FULL FULL ATTNOUT BASE. Each arm
//     holds positions summing to seven within a block, so any drift linear in
//     position cancels inside one block. Six is the smallest position-balanced
//     sequence over three arms. Draws on this host are not exchangeable
//     (availGiB has moved 32.5 GiB inside one sweep), which is why the arms
//     interleave at this granularity instead of running arm after arm.
//   - Every run reports a Wald-Wolfowitz runs z over the whole time series in
//     run order, so the blocking's premise is checked rather than assumed.
//   - The firing control is the kernel's own launch count, not the timing.
//     Fifteen sites over 43 layers is 645 launches; the envelope leaves
//     attnQB, gate, up and down, which is 4*43 = 172, and ATTNOUT leaves
//     those plus attnQA and attnKV, 6*43 = 258. An arm whose count does not
//     move did not fire, and no timing from it means anything.
//   - Outcome classes are disjoint and cross-tabulated: OK, WRONGTOKEN,
//     NONFINITE, COUNT (the launch-count control failed), ERROR.
func TestCUDADenseShapeEnvelope(t *testing.T) {
	if os.Getenv("DS4_CUDA_DENSE_ENVELOPE_GATE") != "1" {
		t.Skip("DS4_CUDA_DENSE_ENVELOPE_GATE is not set")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("a GPU lock is required")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("DS4_MODEL and DS4_CUDA_PREFILL_TOKENS are required")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("input contains %d tokens, want 2048", len(tokens))
	}
	blocks := 10
	if s := os.Getenv("DS4_CUDA_DENSE_ENVELOPE_BLOCKS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			t.Fatalf("DS4_CUDA_DENSE_ENVELOPE_BLOCKS=%q", s)
		}
		blocks = n
	}

	t.Setenv("DS4_CUDA_DENSE_D2R", "1")
	t.Setenv("DS4_CUDA_MOE_D2R", "1")
	t.Setenv("DS4_CUDA_ATTENTION_HMMA", "1")
	t.Setenv("DS4_CUDA_DENSE_D2R_ENVELOPE", "")

	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()

	type arm struct {
		name  string
		env   string
		// want is the expected value of denseQ8D2RLaunches, which counts HOST
		// submissions after the launch API returns (cuda_dense_d2r_linux.go).
		// It is evidence that this encoder was reached the expected number of
		// times. It is not a device-maintained counter and not a CUPTI
		// census, and under graph capture the two diverge.
		want  uint64
		sites string
	}
	arms := []arm{
		{"BASE", "", 645, "all 15"},
		{"ATTNOUT", "attnout", 258, "15 - 8 attnOutA - attnOutB"},
		{"FULL", "1", 172, "attnQB, gate, up, down"},
	}
	order := []int{0, 1, 2, 2, 1, 0}

	type draw struct {
		block, pos int
		arm        string
		ms         float64
		launches   uint64
		class      string
		digest     string
		avail      float64
	}
	var draws []draw

	run := func(a arm) (float64, uint64, string, string, float64) {
		if err := os.Setenv("DS4_CUDA_DENSE_D2R_ENVELOPE", a.env); err != nil {
			t.Fatal(err)
		}
		if err := r.Reset(); err != nil {
			return 0, 0, "ERROR", err.Error(), memAvailableGiB()
		}
		before := r.denseQ8D2RLaunches
		start := time.Now()
		evalErr := r.EvalBatch(tokens, 0)
		if evalErr == nil {
			evalErr = syncStream(r.stream())
		}
		ms := float64(time.Since(start).Nanoseconds()) / 1e6
		got := r.denseQ8D2RLaunches - before
		avail := memAvailableGiB()
		if evalErr != nil {
			return ms, got, "ERROR", evalErr.Error(), avail
		}
		logits := r.Logits()
		raw := make([]byte, len(logits)*4)
		for i, v := range logits {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		digest := fmt.Sprintf("%x", sha256.Sum256(raw))[:16]
		// Every check runs, and all of their answers are carried. The class
		// below is a priority label for the cross-tab, not an independent
		// observation: while it was the only thing recorded, a COUNT row read
		// as "the count was wrong" and silently also meant "nobody looked at
		// whether the logits were finite". A joint census needs every
		// observable, and the ones an ERROR draw never reaches must be
		// explicitly unobserved rather than absent.
		bad := nonFiniteExtent(logits).bad()
		tok := greedyToken(logits)
		t.Logf("    checks: launches=%d/%d nonfinite=%v token=%d/%d",
			got, a.want, bad, tok, 15342)
		switch {
		case got != a.want:
			return ms, got, "COUNT", digest, avail
		case bad:
			return ms, got, "NONFINITE", digest, avail
		case tok != 15342:
			return ms, got, "WRONGTOKEN", digest, avail
		}
		return ms, got, "OK", digest, avail
	}

	// Two discarded warmups: the first prefill in a process has paid a
	// first-slot penalty on this host before.
	for i := range 2 {
		ms, got, class, digest, _ := run(arms[0])
		t.Logf("warmup %d: %.1f ms launches=%d %s %s", i, ms, got, class, digest)
	}

	for b := range blocks {
		for p, ai := range order {
			ms, got, class, digest, avail := run(arms[ai])
			draws = append(draws, draw{b, p, arms[ai].name, ms, got, class, digest, avail})
			t.Logf("block=%d pos=%d arm=%-7s %8.1f ms launches=%-4d %-10s %s availGiB=%.1f",
				b, p, arms[ai].name, ms, got, class, digest, avail)
		}
	}

	// Outcome classes, disjoint, cross-tabulated by arm.
	classes := []string{"OK", "WRONGTOKEN", "NONFINITE", "COUNT", "ERROR"}
	t.Logf("--- outcome classes by arm (disjoint) ---")
	for _, a := range arms {
		counts := map[string]int{}
		for _, d := range draws {
			if d.arm == a.name {
				counts[d.class]++
			}
		}
		line, total := "", 0
		for _, c := range classes {
			line += fmt.Sprintf(" %s=%d", c, counts[c])
			total += counts[c]
		}
		t.Logf("%-7s n=%d%s  sites=%s", a.name, total, line, a.sites)
	}

	// Timing over OK draws only; every other class is reported above rather
	// than folded into a mean.
	t.Logf("--- timing, OK draws only ---")
	for _, a := range arms {
		var v []float64
		for _, d := range draws {
			if d.arm == a.name && d.class == "OK" {
				v = append(v, d.ms)
			}
		}
		if len(v) == 0 {
			t.Logf("%-7s no OK draws", a.name)
			continue
		}
		sort.Float64s(v)
		sum := 0.0
		for _, x := range v {
			sum += x
		}
		mean := sum / float64(len(v))
		sd := 0.0
		for _, x := range v {
			sd += (x - mean) * (x - mean)
		}
		if len(v) > 1 {
			sd = math.Sqrt(sd / float64(len(v)-1))
		}
		t.Logf("%-7s n=%d median=%.1f mean=%.1f sd=%.1f min=%.1f max=%.1f",
			a.name, len(v), median(v), mean, sd, v[0], v[len(v)-1])
	}

	// Paired within-block differences: each block holds two draws of each arm,
	// so the block mean removes whatever the block's covariates were.
	t.Logf("--- within-block paired differences (positive = envelope slower) ---")
	pairDiff := func(x, y string) {
		var ds []float64
		for b := range blocks {
			var xs, ys []float64
			for _, d := range draws {
				if d.block != b || d.class != "OK" {
					continue
				}
				if d.arm == x {
					xs = append(xs, d.ms)
				}
				if d.arm == y {
					ys = append(ys, d.ms)
				}
			}
			if len(xs) != 2 || len(ys) != 2 {
				continue
			}
			ds = append(ds, (ys[0]+ys[1])/2-(xs[0]+xs[1])/2)
		}
		if len(ds) == 0 {
			t.Logf("%s -> %s: no complete blocks", x, y)
			return
		}
		sum, pos := 0.0, 0
		for _, d := range ds {
			sum += d
			if d > 0 {
				pos++
			}
		}
		mean := sum / float64(len(ds))
		sd := 0.0
		for _, d := range ds {
			sd += (d - mean) * (d - mean)
		}
		if len(ds) > 1 {
			sd = math.Sqrt(sd / float64(len(ds)-1))
		}
		sort.Float64s(ds)
		se := sd / math.Sqrt(float64(len(ds)))
		tstat := 0.0
		if se > 0 {
			tstat = mean / se
		}
		t.Logf("%s -> %s: blocks=%d mean=%+.1f ms median=%+.1f sd=%.1f t=%+.2f sign=%d/%d positive",
			x, y, len(ds), mean, median(ds), sd, tstat, pos, len(ds))
	}
	pairDiff("BASE", "ATTNOUT")
	pairDiff("ATTNOUT", "FULL")
	pairDiff("BASE", "FULL")

	// Acceptance. Everything above this point reports; nothing above it
	// refuses. Until this block existed a run in which every draw returned
	// COUNT -- the launch-count control failing on every single draw -- still
	// exited 0, because the class was recorded, printed, and then excluded
	// from the timings it would have contaminated. Reporting a bad outcome
	// and asserting against it are different acts, and this test did only
	// the first while the ledger cited its exit status as the second.
	//
	// Diagnostics are retained above regardless; this only decides the
	// exit status. DS4_CUDA_DENSE_ENVELOPE_ALLOW_BAD accepts a budget of
	// non-OK draws for a deliberately hostile run.
	budget := 0
	if s := os.Getenv("DS4_CUDA_DENSE_ENVELOPE_ALLOW_BAD"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			t.Fatalf("DS4_CUDA_DENSE_ENVELOPE_ALLOW_BAD=%q", s)
		}
		budget = n
	}
	var bad []draw
	for _, d := range draws {
		if d.class != "OK" {
			bad = append(bad, d)
		}
	}
	if len(bad) > budget {
		for _, d := range bad {
			t.Errorf("block=%d pos=%d arm=%s class=%s launches=%d digest=%s",
				d.block, d.pos, d.arm, d.class, d.launches, d.digest)
		}
		t.Errorf("%d of %d draws are not OK, budget %d", len(bad), len(draws), budget)
	}
	// An arm with no OK draws has no timing, and pairDiff prints "no complete
	// blocks" and returns -- which read as a quiet pass.
	for _, a := range arms {
		n := 0
		for _, d := range draws {
			if d.arm == a.name && d.class == "OK" {
				n++
			}
		}
		if n == 0 {
			t.Errorf("arm %s has no OK draws; its contrasts are empty, not null", a.name)
		}
	}

	// Wald-Wolfowitz runs test on the whole series in run order, against its
	// own median. A large negative z means neighbouring draws agree more than
	// chance allows, i.e. the covariate drifted and the blocking was needed.
	t.Logf("--- runs test, all draws in run order ---")
	all := make([]float64, len(draws))
	for i := range draws {
		all[i] = draws[i].ms
	}
	t.Logf("%s", runsZ(all))
	for _, a := range arms {
		var v []float64
		for _, d := range draws {
			if d.arm == a.name {
				v = append(v, d.ms)
			}
		}
		t.Logf("%-7s %s", a.name, runsZ(v))
	}

	// The same statistic over availGiB, which is the series the drift
	// argument is actually about. An earlier version fed runsZ only d.ms and
	// the ledger then cited its z as evidence that headroom did not drift --
	// a statistic about latency quoted as one about memory. They are
	// different series and either can be flat while the other is not.
	avail := make([]float64, len(draws))
	for i := range draws {
		avail[i] = draws[i].avail
	}
	t.Logf("availGiB %s", runsZ(avail))
	lo, hi := avail[0], avail[0]
	for _, v := range avail {
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	// Endpoints alone say nothing about the path between them, so print the
	// range too. Note availGiB is sampled AFTER the draw, so it describes the
	// state each draw left, not the state it ran under.
	t.Logf("availGiB first=%.1f last=%.1f min=%.1f max=%.1f (sampled after each draw)",
		avail[0], avail[len(avail)-1], lo, hi)
}

// median of a sorted slice. Both reporting sites used v[len(v)/2], which for
// an even n is the upper of the two middle samples rather than their mean --
// small against these effects, but a label that names a convention it does
// not follow. runsZ keeps the order statistic deliberately: there it is a
// threshold for classifying samples, not a reported value.
func median(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

// runsZ reports the Wald-Wolfowitz runs statistic of v against its median.
func runsZ(v []float64) string {
	if len(v) < 3 {
		return "n<3"
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	med := s[len(s)/2]
	var signs []int
	for _, x := range v {
		switch {
		case x > med:
			signs = append(signs, 1)
		case x < med:
			signs = append(signs, -1)
		}
	}
	n1, n2, runs := 0, 0, 0
	for i, s := range signs {
		if s > 0 {
			n1++
		} else {
			n2++
		}
		if i == 0 || s != signs[i-1] {
			runs++
		}
	}
	if n1 == 0 || n2 == 0 {
		return "degenerate"
	}
	n := float64(n1 + n2)
	mu := 2*float64(n1)*float64(n2)/n + 1
	varr := (mu - 1) * (mu - 2) / (n - 1)
	if varr <= 0 {
		return "degenerate"
	}
	return fmt.Sprintf("n=%d runs=%d expected=%.1f z=%+.2f", len(signs), runs, mu, (float64(runs)-mu)/math.Sqrt(varr))
}
