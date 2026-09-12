//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"
)

func TestCUDADeepDecodeBenchmark(t *testing.T) {
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}

	// 1. Load the registered oracle 2048 prefill tokens
	tokensPath := "../../scripts/testdata/oracle-workload-tokens2048.json"
	tokensBytes, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatalf("read oracle tokens: %v", err)
	}
	var prefillTokens []int
	if err := json.Unmarshal(tokensBytes, &prefillTokens); err != nil {
		t.Fatalf("unmarshal oracle tokens: %v", err)
	}
	if len(prefillTokens) != oracleWorkloadPrefill {
		t.Fatalf("oracle token count = %d, want %d", len(prefillTokens), oracleWorkloadPrefill)
	}

	// Verify token hash matches registered oracle workload
	hashData := make([]byte, 4*len(prefillTokens))
	for i, tok := range prefillTokens {
		binary.LittleEndian.PutUint32(hashData[4*i:], uint32(tok))
	}
	tokenHash := fmt.Sprintf("%x", sha256.Sum256(hashData))
	if tokenHash != oracleWorkloadTokenHash {
		t.Fatalf("token hash = %s, want %s", tokenHash, oracleWorkloadTokenHash)
	}

	t.Logf("loaded and verified %d oracle prefill tokens (hash %s)", len(prefillTokens), tokenHash[:16])

	// 2. Instantiate runtime at ctx = 65,665 (oracle workload context)
	t.Logf("creating CUDA session at ctx = %d...", oracleWorkloadContext)
	r, err := newCUDARuntime(modelPath, oracleWorkloadContext)
	if err != nil {
		t.Fatalf("newCUDARuntime: %v", err)
	}
	defer r.Close()

	stream := r.device.DefaultStream()

	// 3. Prefill the 2,048 tokens
	t.Logf("prefilling %d tokens in chunks of %d...", len(prefillTokens), cudaPrefillBatchWidth)
	prefillStart := time.Now()
	if err := r.EvalBatch(prefillTokens, 0); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	prefillDur := time.Since(prefillStart)
	t.Logf("prefill complete in %.2fs (%.1f tok/s)", prefillDur.Seconds(), float64(len(prefillTokens))/prefillDur.Seconds())

	// 4. Assert and record live attn.n and indexer.n from runtime state
	evenLayerIdx := 2 // Layer 2 is an even layer carrying ratio=4 compressor and indexer
	layerState := &r.state.layers[evenLayerIdx]
	liveAttnN := layerState.attn.n
	liveIndexerN := layerState.indexer.n
	t.Logf("post-prefill live state at layer %d: attn.n = %d, indexer.n = %d (cap = %d)",
		evenLayerIdx, liveAttnN, liveIndexerN, layerState.indexer.cap)

	if liveAttnN != 512 {
		t.Fatalf("live attn.n = %d, want 512 (2048 / ratio 4)", liveAttnN)
	}
	if liveIndexerN != 512 {
		t.Fatalf("live indexer.n = %d, want 512 (2048 / ratio 4)", liveIndexerN)
	}

	// 5. Evaluate positions 2048..2063 with per-phase CUDA events
	const numDecodeTokens = 16
	decodeTokens := make([]int, numDecodeTokens)
	for i := range decodeTokens {
		decodeTokens[i] = 1000 + i*17
	}

	metrics := make([]stepMetrics, numDecodeTokens)

	t.Logf("benchmarking decode steps at positions %d..%d...", oracleWorkloadPrefill, oracleWorkloadPrefill+numDecodeTokens-1)

	for step := 0; step < numDecodeTokens; step++ {
		pos := oracleWorkloadPrefill + step
		var events []decodePhaseEvt

		r.phaseMark = func(name string, position, layer int) error {
			ev, err := r.device.NewEvent()
			if err != nil {
				return err
			}
			if err := ev.Record(stream); err != nil {
				ev.Destroy()
				return err
			}
			events = append(events, decodePhaseEvt{name: name, position: position, layer: layer, event: ev})
			return nil
		}

		startEvt, err := r.device.NewEvent()
		if err != nil {
			t.Fatal(err)
		}
		endEvt, err := r.device.NewEvent()
		if err != nil {
			t.Fatal(err)
		}

		if err := startEvt.Record(stream); err != nil {
			t.Fatal(err)
		}
		hostStart := time.Now()
		if err := r.Eval(decodeTokens[step], pos); err != nil {
			t.Fatalf("eval pos %d: %v", pos, err)
		}
		hostElapsed := time.Since(hostStart)
		if err := endEvt.Record(stream); err != nil {
			t.Fatal(err)
		}

		r.phaseMark = nil

		if err := endEvt.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}

		streamElapsed, err := endEvt.ElapsedSince(startEvt)
		if err != nil {
			t.Fatal(err)
		}

		phaseTotals := make(map[string]time.Duration)
		for i := 0; i+1 < len(events); i++ {
			d, err := events[i+1].event.ElapsedSince(events[i].event)
			if err != nil {
				t.Fatal(err)
			}
			phaseTotals[events[i].name] += d
		}

		startEvt.Destroy()
		endEvt.Destroy()
		for _, e := range events {
			e.event.Destroy()
		}

		var sumPhases time.Duration
		for _, d := range phaseTotals {
			sumPhases += d
		}
		gap := streamElapsed - sumPhases
		if gap < 0 {
			gap = 0
		}

		metrics[step] = stepMetrics{
			streamTime:       float64(streamElapsed.Microseconds()) / 1000.0,
			hostTime:         float64(hostElapsed.Microseconds()) / 1000.0,
			sumPhases:        float64(sumPhases.Microseconds()) / 1000.0,
			interKernelGap:   float64(gap.Microseconds()) / 1000.0,
			attentionPhase:   float64(phaseTotals["attention"].Microseconds()) / 1000.0,
			compressionPhase: float64(phaseTotals["compression"].Microseconds()) / 1000.0,
			expertFFNPhase:   float64(phaseTotals["expert-ffn"].Microseconds()) / 1000.0,
			sharedFFNPhase:   float64(phaseTotals["shared-ffn"].Microseconds()) / 1000.0,
			attnOutPhase:     float64(phaseTotals["attention-output"].Microseconds()) / 1000.0,
			ffnPrePhase:            float64(phaseTotals["ffn-pre"].Microseconds()) / 1000.0,
			ffnPostPhase:           float64(phaseTotals["ffn-post"].Microseconds()) / 1000.0,
			indexerQBMatvecPhase:   float64(phaseTotals["indexer-qb-matvec"].Microseconds()) / 1000.0,
			indexerProjMatvecPhase: float64(phaseTotals["indexer-proj-matvec"].Microseconds()) / 1000.0,
			indexerQueryPhase:      float64(phaseTotals["indexer-query"].Microseconds()) / 1000.0,
			indexerScorePhase:      float64(phaseTotals["indexer-score"].Microseconds()) / 1000.0,
			indexerSelectPhase:     float64(phaseTotals["indexer-select"].Microseconds()) / 1000.0,
		}
	}

	finalAttnN := layerState.attn.n
	finalIndexerN := layerState.indexer.n
	t.Logf("post-decode live state at layer %d: attn.n = %d, indexer.n = %d (boundary >512 crossed = %t)",
		evenLayerIdx, finalAttnN, finalIndexerN, finalIndexerN > 512)

	// Sum-rule structural gate: every position must have strictly positive unattributed residue.
	for step := 0; step < numDecodeTokens; step++ {
		m := metrics[step]
		if m.sumPhases >= m.streamTime {
			t.Fatalf("position %d: phases sum to %.3f ms against a %.3f ms step; a phase was solved for, not measured",
				oracleWorkloadPrefill+step, m.sumPhases, m.streamTime)
		}
		if m.interKernelGap <= 0 {
			t.Fatalf("position %d: zero unattributed residue; every nanosecond of the step is claimed by a phase, which cannot be true",
				oracleWorkloadPrefill+step)
		}
	}

	// Per-position full phase progression
	fmt.Printf("\n=== PER-POSITION PHASE BREAKDOWN (ctx=%d, positions %d..%d) ===\n",
		oracleWorkloadContext, oracleWorkloadPrefill, oracleWorkloadPrefill+numDecodeTokens-1)
	fmt.Printf("%-5s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-8s\n",
		"Pos", "FullStep", "compress", "attn-out", "ffn-post", "exp-ffn", "ffn-pre", "sh-ffn", "attention", "residue", "sum-ph")
	fmt.Println("------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+---------")
	for step := 0; step < numDecodeTokens; step++ {
		m := metrics[step]
		fmt.Printf("%-5d | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms\n",
			oracleWorkloadPrefill+step, m.streamTime, m.compressionPhase, m.attnOutPhase, m.ffnPostPhase,
			m.expertFFNPhase, m.ffnPrePhase, m.sharedFFNPhase, m.attentionPhase, m.interKernelGap, m.sumPhases)
	}

	// Dedicated 5-way indexer breakdown per position
	fmt.Printf("\n=== PER-POSITION INDEXER FIVE-WAY BREAKDOWN (positions %d..%d) ===\n",
		oracleWorkloadPrefill, oracleWorkloadPrefill+numDecodeTokens-1)
	fmt.Printf("%-5s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
		"Pos", "compress", "idx-qb", "idx-proj", "idx-query", "idx-score", "idx-sel", "idx-5-sum")
	fmt.Println("------+-----------+-----------+-----------+-----------+-----------+-----------+----------")
	for step := 0; step < numDecodeTokens; step++ {
		m := metrics[step]
		idxSum := m.indexerQBMatvecPhase + m.indexerProjMatvecPhase + m.indexerQueryPhase + m.indexerScorePhase + m.indexerSelectPhase
		fmt.Printf("%-5d | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms | %6.2f ms\n",
			oracleWorkloadPrefill+step, m.compressionPhase, m.indexerQBMatvecPhase, m.indexerProjMatvecPhase,
			m.indexerQueryPhase, m.indexerScorePhase, m.indexerSelectPhase, idxSum)
	}

	calcStats := func(vals []float64) (median, min, max float64) {
		if len(vals) == 0 {
			return 0, 0, 0
		}
		sorted := append([]float64(nil), vals...)
		sort.Float64s(sorted)
		min = sorted[0]
		max = sorted[len(sorted)-1]
		n := len(sorted)
		if n%2 == 1 {
			median = sorted[n/2]
		} else {
			median = (sorted[n/2-1] + sorted[n/2]) / 2.0
		}
		return median, min, max
	}

	extract := func(field func(s stepMetrics) float64) []float64 {
		out := make([]float64, len(metrics))
		for i, s := range metrics {
			out[i] = field(s)
		}
		return out
	}

	fmt.Printf("\n=== CUDA DEEP-CONTEXT DECODE BREAKDOWN (ctx=%d, positions %d..%d, live indexer.n=%d..%d) ===\n",
		oracleWorkloadContext, oracleWorkloadPrefill, oracleWorkloadPrefill+numDecodeTokens-1, liveIndexerN, finalIndexerN)
	fmt.Printf("%-24s | %s\n", "Metric / Phase", "Duration ms: Median [Min, Max]")
	fmt.Println("-------------------------+---------------------------------")
	for _, row := range []struct {
		name  string
		field func(s stepMetrics) float64
	}{
		{"Full Step (Stream)", func(s stepMetrics) float64 { return s.streamTime }},
		{"Full Step (Host)", func(s stepMetrics) float64 { return s.hostTime }},
		{"Sum of Phases", func(s stepMetrics) float64 { return s.sumPhases }},
		{"Unattributed Residue", func(s stepMetrics) float64 { return s.interKernelGap }},
		{"---", nil},
		{"compression (base)", func(s stepMetrics) float64 { return s.compressionPhase }},
		{"indexer-qb-matvec", func(s stepMetrics) float64 { return s.indexerQBMatvecPhase }},
		{"indexer-proj-matvec", func(s stepMetrics) float64 { return s.indexerProjMatvecPhase }},
		{"indexer-query", func(s stepMetrics) float64 { return s.indexerQueryPhase }},
		{"indexer-score", func(s stepMetrics) float64 { return s.indexerScorePhase }},
		{"indexer-select", func(s stepMetrics) float64 { return s.indexerSelectPhase }},
		{"attention-output", func(s stepMetrics) float64 { return s.attnOutPhase }},
		{"ffn-post", func(s stepMetrics) float64 { return s.ffnPostPhase }},
		{"expert-ffn", func(s stepMetrics) float64 { return s.expertFFNPhase }},
		{"ffn-pre", func(s stepMetrics) float64 { return s.ffnPrePhase }},
		{"shared-ffn", func(s stepMetrics) float64 { return s.sharedFFNPhase }},
		{"attention", func(s stepMetrics) float64 { return s.attentionPhase }},
	} {
		if row.field == nil {
			fmt.Println("-------------------------+---------------------------------")
			continue
		}
		med, minV, maxV := calcStats(extract(row.field))
		fmt.Printf("%-24s | %6.2f [%6.2f, %6.2f]\n", row.name, med, minV, maxV)
	}
	fmt.Println("=========================================================")

	// Sum rule check on instrumented phases
	compMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.compressionPhase }))
	idxQBMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.indexerQBMatvecPhase }))
	idxProjMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.indexerProjMatvecPhase }))
	idxQryMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.indexerQueryPhase }))
	idxScoreMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.indexerScorePhase }))
	idxSelMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.indexerSelectPhase }))
	attnOutMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.attnOutPhase }))
	ffnPostMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.ffnPostPhase }))
	expFFNMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.expertFFNPhase }))
	ffnPreMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.ffnPrePhase }))
	shFFNMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.sharedFFNPhase }))
	attnMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.attentionPhase }))
	stepMed, _, _ := calcStats(extract(func(s stepMetrics) float64 { return s.streamTime }))

	sumAllPhases := compMed + idxQBMed + idxProjMed + idxQryMed + idxScoreMed + idxSelMed + attnOutMed + ffnPostMed + expFFNMed + ffnPreMed + shFFNMed + attnMed
	unattributedResidue := stepMed - sumAllPhases
	fmt.Printf("\nSum of 12 measured phase medians: %.2f ms vs Step median: %.2f ms (unattributed residue: %+.2f ms, %.1f%%)\n",
		sumAllPhases, stepMed, unattributedResidue, unattributedResidue/stepMed*100.0)
	fmt.Println("=================================================================")
}
