//go:build linux

package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type decodePhaseEvt struct {
	name     string
	position int
	layer    int
	event    *cuda.Event
}

type stepMetrics struct {
	hostTime         float64
	streamTime       float64
	attnOutPhase     float64
	expertFFNPhase   float64
	attentionPhase   float64
	ffnPostPhase     float64
	compressionPhase float64
	ffnPrePhase      float64
	sharedFFNPhase   float64
	sumPhases        float64
	interKernelGap   float64
}

func calcStats(vals []float64) (med, minVal, maxVal float64) {
	if len(vals) == 0 {
		return 0, 0, 0
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	minVal, maxVal = s[0], s[len(s)-1]
	if len(s)%2 == 1 {
		med = s[len(s)/2]
	} else {
		med = (s[len(s)/2-1] + s[len(s)/2]) / 2.0
	}
	return med, minVal, maxVal
}

func TestCUDADecodeBenchmark(t *testing.T) {
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}

	r, err := newCUDARuntime(modelPath, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	stream := r.device.DefaultStream()

	// Initial warmup on a few steps
	for i := 0; i < 4; i++ {
		_ = r.Eval(100+i, i)
	}
	_ = r.Reset()

	const (
		numTrials    = 10
		tokensPerRun = 16
	)

	// Fixed sequence of tokens to evaluate
	tokens := make([]int, tokensPerRun)
	for i := range tokens {
		tokens[i] = 1000 + i*13
	}

	runArm := func(disableBlockDiag bool) ([]stepMetrics, error) {
		r.disableBlockDiagAttnOut = disableBlockDiag
		if err := r.Reset(); err != nil {
			return nil, err
		}

		metrics := make([]stepMetrics, tokensPerRun)

		for pos := 0; pos < tokensPerRun; pos++ {
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
				return nil, err
			}
			endEvt, err := r.device.NewEvent()
			if err != nil {
				return nil, err
			}

			if err := startEvt.Record(stream); err != nil {
				return nil, err
			}
			hostStart := time.Now()
			if err := r.Eval(tokens[pos], pos); err != nil {
				return nil, fmt.Errorf("eval pos %d: %w", pos, err)
			}
			hostElapsed := time.Since(hostStart)
			if err := endEvt.Record(stream); err != nil {
				return nil, err
			}

			r.phaseMark = nil

			if err := endEvt.Synchronize(context.Background()); err != nil {
				return nil, err
			}

			streamElapsed, err := endEvt.ElapsedSince(startEvt)
			if err != nil {
				return nil, err
			}

			phaseTotals := make(map[string]time.Duration)
			for i := 0; i+1 < len(events); i++ {
				d, err := events[i+1].event.ElapsedSince(events[i].event)
				if err != nil {
					return nil, err
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

			metrics[pos] = stepMetrics{
				hostTime:         float64(hostElapsed.Microseconds()) / 1000.0,
				streamTime:       float64(streamElapsed.Microseconds()) / 1000.0,
				attnOutPhase:     float64(phaseTotals["attention-output"].Microseconds()) / 1000.0,
				expertFFNPhase:   float64(phaseTotals["expert-ffn"].Microseconds()) / 1000.0,
				attentionPhase:   float64(phaseTotals["attention"].Microseconds()) / 1000.0,
				ffnPostPhase:     float64(phaseTotals["ffn-post"].Microseconds()) / 1000.0,
				compressionPhase: float64(phaseTotals["compression"].Microseconds()) / 1000.0,
				ffnPrePhase:      float64(phaseTotals["ffn-pre"].Microseconds()) / 1000.0,
				sharedFFNPhase:   float64(phaseTotals["shared-ffn"].Microseconds()) / 1000.0,
				sumPhases:        float64(sumPhases.Microseconds()) / 1000.0,
				interKernelGap:   float64(gap.Microseconds()) / 1000.0,
			}
		}
		return metrics, nil
	}

	var allArmA, allArmB []stepMetrics

	// Run 10 interleaved trials alternating order to eliminate thermal bias
	for trial := 0; trial < numTrials; trial++ {
		if trial%2 == 0 {
			mA, err := runArm(true) // Arm A: 8-group loop
			if err != nil {
				t.Fatal(err)
			}
			mB, err := runArm(false) // Arm B: BlockDiag fused
			if err != nil {
				t.Fatal(err)
			}
			allArmA = append(allArmA, mA...)
			allArmB = append(allArmB, mB...)
		} else {
			mB, err := runArm(false) // Arm B: BlockDiag fused
			if err != nil {
				t.Fatal(err)
			}
			mA, err := runArm(true) // Arm A: 8-group loop
			if err != nil {
				t.Fatal(err)
			}
			allArmA = append(allArmA, mA...)
			allArmB = append(allArmB, mB...)
		}
	}

	extract := func(steps []stepMetrics, get func(s stepMetrics) float64) []float64 {
		out := make([]float64, len(steps))
		for i, s := range steps {
			out[i] = get(s)
		}
		return out
	}

	printRow := func(label string, field func(s stepMetrics) float64) {
		medA, minA, maxA := calcStats(extract(allArmA, field))
		medB, minB, maxB := calcStats(extract(allArmB, field))
		delta := medB - medA
		pct := 0.0
		if medA > 0 {
			pct = (delta / medA) * 100.0
		}
		fmt.Printf("%-18s | %6.2f [%6.2f, %6.2f] | %6.2f [%6.2f, %6.2f] | %+6.2f ms (%+5.1f%%)\n",
			label, medA, minA, maxA, medB, minB, maxB, delta, pct)
	}

	fmt.Printf("\n=== STRICT INTERLEAVED A/B BENCHMARK (N=%d matched steps per arm, positions 0..%d) ===\n",
		len(allArmA), tokensPerRun-1)
	fmt.Printf("%-18s | %-24s | %-24s | %-16s\n", "Metric", "Arm A (8-Group Loop)", "Arm B (BlockDiag Fused)", "Delta (B - A)")
	fmt.Println("-------------------+--------------------------+--------------------------+-----------------")
	printRow("Full Step (Stream)", func(s stepMetrics) float64 { return s.streamTime })
	printRow("Full Step (Host)", func(s stepMetrics) float64 { return s.hostTime })
	printRow("Sum of Phases", func(s stepMetrics) float64 { return s.sumPhases })
	printRow("Inter-kernel Gap", func(s stepMetrics) float64 { return s.interKernelGap })
	fmt.Println("-------------------+--------------------------+--------------------------+-----------------")
	printRow("attention-output", func(s stepMetrics) float64 { return s.attnOutPhase })
	printRow("expert-ffn", func(s stepMetrics) float64 { return s.expertFFNPhase })
	printRow("attention", func(s stepMetrics) float64 { return s.attentionPhase })
	printRow("ffn-post", func(s stepMetrics) float64 { return s.ffnPostPhase })
	printRow("compression", func(s stepMetrics) float64 { return s.compressionPhase })
	printRow("ffn-pre", func(s stepMetrics) float64 { return s.ffnPrePhase })
	printRow("shared-ffn", func(s stepMetrics) float64 { return s.sharedFFNPhase })
	fmt.Println("==========================================================================================")
}
