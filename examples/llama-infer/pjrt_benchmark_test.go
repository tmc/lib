package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/pprof"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// BenchmarkPJRTInference measures warm F32 adapter calls against a pinned
// checkpoint reference. Calls include token upload and last-row logit extraction.
func BenchmarkPJRTInference(b *testing.B) {
	dir, refDir, plugin := os.Getenv("PJRT_PRETRAINED_CHECKPOINT"), os.Getenv("PJRT_PRETRAINED_REFERENCE"), os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if dir == "" || refDir == "" || plugin == "" {
		b.Skip("set PJRT_PRETRAINED_CHECKPOINT, PJRT_PRETRAINED_REFERENCE and XLA_PJRT_CPU_PLUGIN")
	}
	ref := readPJRTReference(b, refDir)
	if ref.TeacherForced || (ref.ComputeDType != "" && ref.ComputeDType != "float32") {
		b.Fatal("benchmark requires a free-running F32 reference")
	}
	for name, hash := range ref.Files {
		checkPJRTReferenceHash(b, filepath.Join(dir, name), hash)
	}
	p, err := loadPJRTCheckpoint(dir, ref.Capacity, xla.F32)
	if err != nil {
		b.Fatal(err)
	}
	if p.config.VocabSize != ref.Vocab {
		b.Fatal("reference vocabulary mismatch")
	}
	engine, err := xla.NewEngine(plugin)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := engine.Close(); err != nil {
			b.Error(err)
		}
	})
	hostAccess := os.Getenv("PJRT_CPU_HOST_ACCESS") == "1"
	if hostAccess {
		checksum := os.Getenv("PJRT_CPU_HOST_ACCESS_SHA256")
		if checksum == "" {
			b.Fatal("host access requires PJRT_CPU_HOST_ACCESS_SHA256 for the qualified plugin")
		}
		checkPJRTReferenceHash(b, plugin, checksum)
		b.Logf("CPU host access plugin SHA256: %s", checksum)
		if err := engine.EnableCPUHostAccess(); err != nil {
			b.Fatal(err)
		}
	}
	model, err := newPJRTModel(context.Background(), engine, &p.config, p.weights, p.theta, xla.F32)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := model.close(); err != nil {
			b.Error(err)
		}
	})
	for caseIndex, c := range ref.Cases {
		b.Run(fmt.Sprintf("case%d", caseIndex), func(b *testing.B) {
			path := filepath.Join(refDir, c.Logits)
			checkPJRTReferenceHash(b, path, c.SHA256)
			raw, err := os.ReadFile(path)
			if err != nil {
				b.Fatal(err)
			}
			if len(raw) != len(c.Generated)*ref.Vocab*4 {
				b.Fatal("reference logit shape")
			}
			check := func(b *testing.B, logits []float32, step int) {
				b.Helper()
				if model.lastHostRead != hostAccess {
					b.Fatalf("readout route direct=%v, want %v", model.lastHostRead, hostAccess)
				}
				if len(logits) != ref.Vocab {
					b.Fatalf("got %d logits, want %d", len(logits), ref.Vocab)
				}
				if got := sampleArgmax(logits); got != c.Generated[step] {
					b.Fatalf("step %d token %d, want %d", step, got, c.Generated[step])
				}
				for i, got := range logits {
					want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(step*ref.Vocab+i)*4:]))
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(got-want)) > 1e-3+1e-4*math.Abs(float64(want)) {
						b.Fatalf("step %d logit %d: %g, want %g", step, i, got, want)
					}
				}
			}
			for _, phase := range []string{"prefill", "decode"} {
				b.Run(phase, func(b *testing.B) {
					if phase == "decode" && len(c.Generated) < 2 {
						b.Skip("decode requires at least two reference steps")
					}
					ctx := context.Background()
					input, position, step := c.Input, 0, 0
					if phase == "decode" {
						input, position, step = []int32{c.Generated[0]}, len(c.Input), 1
					}
					prepare := func() {
						if err := model.reset(); err != nil {
							b.Fatal(err)
						}
						if phase == "decode" {
							logits, err := model.forward(ctx, c.Input, 0)
							if err != nil {
								b.Fatal(err)
							}
							check(b, logits, 0)
						}
					}
					// Warm the exact chunk shape, and validate before timing.
					prepare()
					logits, err := model.forward(ctx, input, position)
					if err != nil {
						b.Fatal(err)
					}
					check(b, logits, step)
					b.ReportAllocs()
					b.ResetTimer()
					b.StopTimer()
					for i := 0; i < b.N; i++ {
						pprof.Do(ctx, pprof.Labels("pjrt.phase", "setup"), func(context.Context) { prepare() })
						logits = nil // A skipped call must not reuse the warm-up result.
						// Enter labels outside timing. StopTimer does not stop CPU profiling.
						pprof.Do(ctx, pprof.Labels("pjrt.phase", phase, "pjrt.dtype", "f32"), func(ctx context.Context) {
							b.StartTimer()
							logits, err = model.forward(ctx, input, position)
							b.StopTimer()
						})
						if err != nil {
							b.Fatal(err)
						}
						check(b, logits, step)
					}
					b.ReportMetric(float64(len(input)*b.N)/b.Elapsed().Seconds(), "tokens/s")
					if hostAccess {
						b.ReportMetric(1, "hostview/op")
					} else {
						b.ReportMetric(0, "hostview/op")
					}
				})
			}
		})
	}
}
